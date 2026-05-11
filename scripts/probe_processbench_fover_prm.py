#!/usr/bin/env python3
"""Evaluate FoVer/Math PRMs on Qwen/ProcessBench.

The FoVer branch mirrors the public FoVer PRM quickstart:
https://github.com/psunlpgroup/FoVer

The Qwen math PRM branch mirrors the Hugging Face model-card usage for
Qwen/Qwen2.5-Math-PRM-7B and Qwen/Qwen2.5-Math-7B-PRM800K.

It reports two views:
- fover_direct_*: FoVer-style direct metrics over ProcessBench per-step labels.
- first_error_*: ProcessBench first-error-index metrics, matching
  scripts/probe_processbench_fol_f1.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Literal


FOVER_FIRST_STEP_TEMPLATE = """** Problem **
{problem}

** Task **
Your task is to evaluate the accuracy of each step in the provided solution to the above question. For each step, respond with "correct" if the reasoning is logically valid and mathematically sound, or if the step is a general statement or transition that does not contain reasoning. Respond with "incorrect" if the step includes any errors or flawed logic.

** Sotluion **
{first_step}"""


def _fover_input_format(problem: str, solution_steps: list[str]) -> list[dict[str, str]]:
    if not solution_steps:
        return []
    labels = ["correct"] * len(solution_steps)
    conversation = [
        {
            "role": "user",
            "content": FOVER_FIRST_STEP_TEMPLATE.format(
                problem=problem,
                first_step=solution_steps[0],
            ),
        },
        {"role": "assistant", "content": labels[0]},
    ]
    for idx, step in enumerate(solution_steps[1:], start=1):
        conversation.append({"role": "user", "content": step})
        conversation.append({"role": "assistant", "content": labels[idx]})
    return conversation


def _qwen_math_prm_messages(problem: str, solution_steps: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Please reason step by step."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": "<extra_0>".join(solution_steps) + "<extra_0>"},
    ]


def _infer_model_family(model_name: str, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    lowered = model_name.lower()
    if "math-prm" in lowered or "prm800k" in lowered:
        return "qwen_math_prm"
    return "fover"


def _model_type(tokenizer: Any) -> Literal["llama", "qwen"]:
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    raise ValueError(f"Cannot infer FoVer model type from tokenizer name: {name}")


def _step_token_positions(
    tokenized_prompt: Any,
    tokenizer: Any,
    model_type: Literal["llama", "qwen"],
) -> Any:
    import numpy as np

    candidates: list[Any] = []

    if model_type == "llama":
        target_token_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
        candidates.append(tokenized_prompt == target_token_id)

        assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
        ids = np.where(tokenized_prompt == assistant_token_id)[0] + 2
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        ids = ids[ids < len(mask)]
        mask[ids] = True
        candidates.append(mask)

        end_header_id = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)[0]
        ids = np.where(tokenized_prompt == end_header_id)[0] + 1
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        ids = ids[ids < len(mask)]
        mask[ids] = True
        candidates.append(mask)

    elif model_type == "qwen":
        target_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        candidates.append(tokenized_prompt == target_token_id)

        assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
        ids = np.where(tokenized_prompt == assistant_token_id)[0] + 1
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        ids = ids[ids < len(mask)]
        mask[ids] = True
        candidates.append(mask)

        im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        ids = np.where(tokenized_prompt == im_start_id)[0] + 2
        mask = np.zeros_like(tokenized_prompt, dtype=bool)
        ids = ids[ids < len(mask)]
        mask[ids] = True
        candidates.append(mask)
    else:
        raise NotImplementedError(model_type)

    return np.where(np.logical_and.reduce(candidates))[0]


def _extract_fover_scores(
    tokenized_prompt: Any,
    logits: Any,
    tokenizer: Any,
    model_type: Literal["llama", "qwen"],
) -> list[float]:
    import torch

    positions = _step_token_positions(tokenized_prompt, tokenizer, model_type)
    if len(positions) == 0:
        return []

    positive_token_id = tokenizer.encode("correct", add_special_tokens=False)[0]
    negative_token_id = tokenizer.encode("incorrect", add_special_tokens=False)[0]
    selected_logits = logits[torch.as_tensor(positions, device=logits.device)]
    positive_logits = selected_logits[:, positive_token_id]
    negative_logits = selected_logits[:, negative_token_id]
    pair = torch.stack([negative_logits, positive_logits], dim=1)
    return torch.nn.functional.softmax(pair.float(), dim=1)[:, 1].detach().cpu().tolist()


def _dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _batched(iterable: list[Any], size: int) -> list[list[Any]]:
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]


def _first_error_prediction(scores: list[float], threshold: float) -> int:
    for idx, score in enumerate(scores):
        if score <= threshold:
            return idx
    return -1


def _processbench_label_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("pred_label") is not None]
    correct_rows = [row for row in valid_rows if row["gold_label"] == -1]
    error_rows = [row for row in valid_rows if row["gold_label"] != -1]
    exact = sum(row["pred_label"] == row["gold_label"] for row in valid_rows)
    correct_acc = (
        sum(row["pred_label"] == -1 for row in correct_rows) / len(correct_rows)
        if correct_rows
        else 0.0
    )
    error_acc = (
        sum(row["pred_label"] == row["gold_label"] for row in error_rows) / len(error_rows)
        if error_rows
        else 0.0
    )
    simple_f1 = (
        2 * correct_acc * error_acc / (correct_acc + error_acc)
        if correct_acc + error_acc
        else 0.0
    )
    return {
        "total": len(rows),
        "valid_total": len(valid_rows),
        "invalid_total": len(rows) - len(valid_rows),
        "exact_acc": exact / len(valid_rows) if valid_rows else 0.0,
        "correct_acc": correct_acc,
        "error_acc": error_acc,
        "simple_f1_score": simple_f1,
        "correct_count": len(correct_rows),
        "error_count": len(error_rows),
    }


def _binary_metrics(y_true: list[bool], y_score: list[float], threshold: float) -> dict[str, Any]:
    y_pred = [score > threshold for score in y_score]
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(y_true)
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _best_threshold_for_f1(y_true: list[bool], y_score: list[float]) -> tuple[float, float]:
    best_f1 = -1.0
    best_threshold = 0.0
    for threshold in sorted(set(y_score)):
        f1 = _binary_metrics(y_true, y_score, threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def _auroc(y_true: list[bool], y_score: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and y_score[order[end]] == y_score[order[idx]]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for j in range(idx, end):
            ranks[order[j]] = avg_rank
        idx = end
    rank_sum_pos = sum(rank for rank, label in zip(ranks, y_true) if label)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _fover_direct_metric_inputs(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    step_true: list[bool] = []
    step_score: list[float] = []
    instance_true: list[bool] = []
    instance_score: list[float] = []

    for row in rows:
        if row.get("scores") is None:
            continue
        scores = list(row["scores"])
        gold = int(row["gold_label"])
        labels: list[bool | None] = []
        for idx in range(len(scores)):
            if gold == -1:
                labels.append(True)
            elif idx < gold:
                labels.append(True)
            elif idx == gold:
                labels.append(False)
            else:
                labels.append(None)

        for label, score in zip(labels, scores):
            if label is None:
                continue
            step_true.append(label)
            step_score.append(float(score))

        known_labels = [label for label in labels if label is not None]
        if known_labels:
            instance_true.append(all(known_labels))
            instance_score.append(min(scores) if scores else -1e9)

    return {
        "step_true": step_true,
        "step_score": step_score,
        "instance_true": instance_true,
        "instance_score": instance_score,
    }


def _fover_direct_metrics(
    rows: list[dict[str, Any]],
    threshold: float,
    auto_threshold: bool,
) -> dict[str, Any]:
    inputs = _fover_direct_metric_inputs(rows)
    result: dict[str, Any] = {}
    for name, y_true_key, y_score_key in [
        ("step_level", "step_true", "step_score"),
        ("instance_level", "instance_true", "instance_score"),
    ]:
        y_true = inputs[y_true_key]
        y_score = inputs[y_score_key]
        selected_threshold = threshold
        best_f1 = None
        if auto_threshold and y_score:
            selected_threshold, best_f1 = _best_threshold_for_f1(y_true, y_score)
        metrics = _binary_metrics(y_true, y_score, selected_threshold) if y_score else {}
        metrics["auroc"] = _auroc(y_true, y_score) if y_score else None
        metrics["num_labels"] = len(y_true)
        if best_f1 is not None:
            metrics["best_f1"] = best_f1
        result[name] = metrics
    return result


def _score_rows(
    dataset_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from tqdm.auto import tqdm

    model_type = _model_type(tokenizer)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    conversations = [_fover_input_format(row["problem"], list(row["steps"])) for row in dataset_rows]
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False)
        for conv in conversations
    ]
    rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for batch_indices in tqdm(
        _batched(list(range(len(dataset_rows))), args.batch_size),
        desc=f"FoVerPRM/{args.split}",
        unit="batch",
    ):
        batch_texts = [texts[i] for i in batch_indices]
        batch_start = time.perf_counter()
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=args.max_length > 0,
            max_length=args.max_length if args.max_length > 0 else None,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)

        for local_idx, row_idx in enumerate(batch_indices):
            row = dataset_rows[row_idx]
            attention = encoded["attention_mask"][local_idx].detach().cpu().numpy().astype(bool)
            token_ids = encoded["input_ids"][local_idx].detach().cpu().numpy()[attention]
            logits = output.logits[local_idx, : len(token_ids)]
            scores = _extract_fover_scores(token_ids, logits, tokenizer, model_type)
            error: str | None = None
            if len(scores) != len(row["steps"]):
                error = f"score_step_count_mismatch:{len(scores)}!={len(row['steps'])}"
                pred_label: int | None = None
            else:
                pred_label = _first_error_prediction(scores, args.threshold)
            rows.append(
                {
                    "dataset": "processbench",
                    "subset": args.split,
                    "row_index": row_idx + args.offset,
                    "gold_label": int(row["label"]),
                    "pred_label": pred_label,
                    "scores": scores,
                    "num_steps": len(row["steps"]),
                    "error": error,
                    "elapsed_batch_s": time.perf_counter() - batch_start,
                }
            )
    elapsed = time.perf_counter() - start_all
    for row in rows:
        row["avg_elapsed_s"] = elapsed / len(rows) if rows else math.nan
    return rows


def _make_step_rewards(logits: Any, token_masks: Any) -> list[list[float]]:
    import torch

    probabilities = torch.nn.functional.softmax(logits.float(), dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores: list[list[float]] = []
    for idx in range(probabilities.size(0)):
        sample = probabilities[idx]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        all_scores.append(positive_probs.detach().cpu().tolist())
    return all_scores


def _truncate_qwen_math_prm_content(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_content_tokens: int,
) -> list[dict[str, str]]:
    if max_content_tokens <= 0:
        return messages
    messages = [dict(message) for message in messages]
    content = messages[-1]["content"]
    tokens = tokenizer(content)["input_ids"]
    if len(tokens) <= max_content_tokens:
        return messages
    messages[-1]["content"] = tokenizer.decode(tokens[:max_content_tokens])
    return messages


def _score_rows_qwen_math_prm(
    dataset_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import torch
    from tqdm.auto import tqdm

    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [
        _truncate_qwen_math_prm_content(
            tokenizer,
            _qwen_math_prm_messages(row["problem"], list(row["steps"])),
            args.qwen_prm_max_content_tokens,
        )
        for row in dataset_rows
    ]
    texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False,
        )
        for message in messages
    ]
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for batch_indices in tqdm(
        _batched(list(range(len(dataset_rows))), args.batch_size),
        desc=f"MathPRM/{args.split}",
        unit="batch",
    ):
        batch_start = time.perf_counter()
        batch_texts = [texts[i] for i in batch_indices]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=args.max_length > 0,
            max_length=args.max_length if args.max_length > 0 else None,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        logits = output[0] if isinstance(output, (tuple, list)) else output.logits
        token_masks = encoded["input_ids"] == step_sep_id
        batch_scores = _make_step_rewards(logits, token_masks)

        for local_idx, row_idx in enumerate(batch_indices):
            row = dataset_rows[row_idx]
            scores = batch_scores[local_idx]
            error: str | None = None
            if len(scores) != len(row["steps"]):
                error = f"score_step_count_mismatch:{len(scores)}!={len(row['steps'])}"
                pred_label: int | None = None
            else:
                pred_label = _first_error_prediction(scores, args.threshold)
            rows.append(
                {
                    "dataset": "processbench",
                    "subset": args.split,
                    "row_index": row_idx + args.offset,
                    "gold_label": int(row["label"]),
                    "pred_label": pred_label,
                    "scores": scores,
                    "num_steps": len(row["steps"]),
                    "error": error,
                    "elapsed_batch_s": time.perf_counter() - batch_start,
                }
            )
    elapsed = time.perf_counter() - start_all
    for row in rows:
        row["avg_elapsed_s"] = elapsed / len(rows) if rows else math.nan
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Qwen/ProcessBench")
    parser.add_argument("--split", default="gsm8k")
    parser.add_argument("--limit", type=int, default=0, help="0 means full split.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default="ryokamoi/Qwen-2.5-7B-FoVer-PRM-2026")
    parser.add_argument(
        "--model_family",
        choices=["auto", "fover", "qwen_math_prm"],
        default="auto",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--auto_threshold",
        action="store_true",
        help="Pick the threshold that maximizes FoVer-style F1 on this split.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_length", type=int, default=0, help="0 disables tokenizer truncation.")
    parser.add_argument(
        "--qwen_prm_max_content_tokens",
        type=int,
        default=3584,
        help="Truncate joined <extra_0> solution content before chat templating. 0 disables.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output", default="probe_outputs/processbench_gsm8k_fover_prm.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset(args.dataset, split=args.split)
    if args.offset:
        ds = ds.select(range(args.offset, len(ds)))
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    dataset_rows = [dict(row) for row in ds]

    model_family = _infer_model_family(args.model, args.model_family)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model_cls = AutoModel if model_family == "qwen_math_prm" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model,
        torch_dtype=_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()

    if model_family == "qwen_math_prm":
        rows = _score_rows_qwen_math_prm(dataset_rows, tokenizer, model, args)
    else:
        rows = _score_rows(dataset_rows, tokenizer, model, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "dataset": "processbench",
        "subset": args.split,
        "model": args.model,
        "model_family": model_family,
        "threshold": args.threshold,
        "auto_threshold": args.auto_threshold,
        "first_error_metrics": _processbench_label_metrics(rows),
        "fover_direct_metrics": _fover_direct_metrics(rows, args.threshold, args.auto_threshold),
        "avg_elapsed_s": sum(row["avg_elapsed_s"] for row in rows) / len(rows) if rows else math.nan,
        "output": str(output),
    }
    metrics_path = output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote rows to {output}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
