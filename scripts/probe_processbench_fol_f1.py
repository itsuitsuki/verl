#!/usr/bin/env python3
"""Probe the current FOL reward pipeline on Qwen/ProcessBench.

The benchmark label is the first erroneous step index, or -1 for a fully
correct solution. This script maps FOL step scores to the same label space:
the first step with score < threshold is predicted as the first error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from verl.utils.fol_utils.common import call_llm_structured
from verl.utils.reward_score.fol import compute_step_reward_fol


_CLAIM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["non_claim", "given_fact", "derived"],
                    },
                    "premises": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "conclusion": {"type": "string"},
                },
                "required": ["kind", "premises", "conclusion"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def _api_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "api_timeout": args.api_timeout,
        "timeout": args.fol_timeout,
        "max_tries": args.max_tries,
        "old_max_tries": args.old_max_tries,
        "fol_task_type": "math",
        "fol_cumulative_mode": "current_only",
        "fol_judge_use_outlines": args.outlines,
        "api_context_shrink_min_tokens": args.shrink_min_tokens,
        "api_context_shrink_retries": args.shrink_retries,
    }


def _claim_extraction_api_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _api_config(args)
    cfg.update(
        {
            "temperature": 0.0,
            "max_tokens": args.claim_max_tokens,
        }
    )
    return cfg


def _problem_prompt(problem: str) -> str:
    return (
        f"<Context>\n{problem.strip()}\n</Context>\n\n"
        "<Question>\nVerify the mathematical correctness of each solution step.\n</Question>"
    )


def _xml_step(problem: str, previous_steps: list[str], current_step: str, max_previous_steps: int) -> str:
    premises = [f"Problem: {problem.strip()}"]
    if max_previous_steps != 0:
        selected = previous_steps if max_previous_steps < 0 else previous_steps[-max_previous_steps:]
        premises.extend(f"Previous step: {step.strip()}" for step in selected if step and step.strip())
    premise_lines = "\n".join(f"<premise>{premise}</premise>" for premise in premises)
    return f"<step>\n{premise_lines}\n<conclusion>{current_step.strip()}</conclusion>\n</step>"


def _claim_xml_step(problem: str, previous_steps: list[str], claim: dict[str, Any], max_previous_steps: int) -> str:
    premises = [f"Problem: {problem.strip()}"]
    if max_previous_steps != 0:
        selected = previous_steps if max_previous_steps < 0 else previous_steps[-max_previous_steps:]
        premises.extend(f"Previous step: {step.strip()}" for step in selected if step and step.strip())
    premises.extend(str(p).strip() for p in claim.get("premises", []) if str(p).strip())
    premise_lines = "\n".join(f"<premise>{premise}</premise>" for premise in premises)
    conclusion = str(claim.get("conclusion", "")).strip()
    return f"<step>\n{premise_lines}\n<conclusion>{conclusion}</conclusion>\n</step>"


def _safe_reason(debug: dict[str, Any]) -> str:
    if debug.get("declaration_failed_closed"):
        return "declaration_failed"
    if debug.get("format_failed_closed"):
        return "format_failed"
    if debug.get("student_premise_conclusion_duplicate"):
        return "student_duplicate"
    if debug.get("invalid_translation_reason"):
        return str(debug["invalid_translation_reason"])
    if debug.get("z3_error"):
        return "z3_error"
    if debug.get("z3_output"):
        output = str(debug["z3_output"])
        if "SUCCESS_ENTAILED" in output:
            return "entailed"
        if "FAILED_NOT_ENTAILED" in output:
            return "not_entailed"
        if "FAILED_INVALID_TRANSLATION" in output:
            return "invalid_translation"
        if "FAILED_LEAKED_CONCLUSION" in output:
            return "leaked_conclusion"
        return output.strip().splitlines()[-1][:120]
    return ""


def _score_step(
    problem: str,
    previous_steps: list[str],
    current_step: str,
    api_config: dict[str, Any],
    max_previous_steps: int,
) -> dict[str, Any]:
    prompt_text = _problem_prompt(problem)
    step_text = _xml_step(problem, previous_steps, current_step, max_previous_steps)
    start = time.perf_counter()
    result = compute_step_reward_fol(
        step_text,
        prompt_text,
        [step_text],
        api_config=api_config,
        extra_info={
            "fol_context": problem,
            "fol_question": "Verify the mathematical correctness of each solution step.",
            "fol_options": "",
        },
        return_debug=True,
    )
    elapsed = time.perf_counter() - start
    debug = result.get("debug", {}) if isinstance(result, dict) else {}
    score = float(result.get("score", 0.0)) if isinstance(result, dict) else float(result)
    return {
        "score": score,
        "elapsed_s": elapsed,
        "reason": _safe_reason(debug),
        "debug": {
            "cache_hit": bool(debug.get("cache_hit")),
            "judge_usage": debug.get("judge_usage", {}),
            "invalid_translation_reason": debug.get("invalid_translation_reason"),
            "z3_output": debug.get("z3_output"),
            "z3_error": debug.get("z3_error"),
        },
    }


def _score_claim(
    problem: str,
    previous_steps: list[str],
    claim: dict[str, Any],
    api_config: dict[str, Any],
    max_previous_steps: int,
) -> dict[str, Any]:
    if claim.get("kind") in {"non_claim", "given_fact"}:
        return {
            "score": 1.0,
            "elapsed_s": 0.0,
            "reason": f"{claim.get('kind')}_skipped",
            "debug": {
                "cache_hit": False,
                "judge_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "invalid_translation_reason": None,
                "z3_output": None,
                "z3_error": None,
            },
        }

    prompt_text = _problem_prompt(problem)
    step_text = _claim_xml_step(problem, previous_steps, claim, max_previous_steps)
    start = time.perf_counter()
    result = compute_step_reward_fol(
        step_text,
        prompt_text,
        [step_text],
        api_config=api_config,
        extra_info={
            "fol_context": problem,
            "fol_question": "Verify the mathematical correctness of each solution step.",
            "fol_options": "",
        },
        return_debug=True,
    )
    elapsed = time.perf_counter() - start
    debug = result.get("debug", {}) if isinstance(result, dict) else {}
    score = float(result.get("score", 0.0)) if isinstance(result, dict) else float(result)
    return {
        "score": score,
        "elapsed_s": elapsed,
        "reason": _safe_reason(debug),
        "debug": {
            "cache_hit": bool(debug.get("cache_hit")),
            "judge_usage": debug.get("judge_usage", {}),
            "invalid_translation_reason": debug.get("invalid_translation_reason"),
            "z3_output": debug.get("z3_output"),
            "z3_error": debug.get("z3_error"),
        },
    }


def _claim_cache_path(cache_dir: Path, payload: dict[str, Any]) -> Path:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.json"


def _extract_atomic_claims(
    problem: str,
    previous_steps: list[str],
    current_step: str,
    args: argparse.Namespace,
    api_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_dir = Path(args.claim_cache_dir)
    payload = {
        "version": 3,
        "problem": problem,
        "previous_steps": previous_steps,
        "current_step": current_step,
        "max_previous_steps": args.max_previous_steps,
    }
    path = _claim_cache_path(cache_dir, payload)
    if path.exists() and not args.no_claim_cache:
        with path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached.get("claims", []), {"cache_hit": True, "raw": cached}

    selected_previous = previous_steps
    if args.max_previous_steps >= 0:
        selected_previous = previous_steps[-args.max_previous_steps :] if args.max_previous_steps else []
    prompt = f"""You are extracting atomic mathematical claims for process-supervision evaluation.

Rules:
- Only extract claims stated in the current step. Do not correct arithmetic.
- Preserve the current step's exact entities, quantities, and numeric results. If the step says "18 pink" even though the correct value is "12 pink", the conclusion must say "18 pink".
- Do not replace a wrong current-step claim with a corrected claim inferred from the problem or previous steps.
- If the current step includes an explicit equation or calculation result, extract it as a "derived" claim, even when the operands are given facts.
- Mark "given_fact" only for a direct restatement of a fact from the original problem with no calculation, no transformation, and no new result.
- Mark "derived" for any computed value, comparison, total, difference, product, quotient, remainder, or final answer.
- Mark "non_claim" for procedural text with no checkable mathematical assertion, such as "let's break down the problem" or "we need to calculate the total".
- A claim must contain one single conclusion.
- Keep premises short and faithful to the problem, previous steps, or current step.
- If a sentence contains multiple calculations or numeric assertions, split it into multiple claims.
- Do not use facts from later current-step sentences to repair earlier claims. Preserve the local reasoning order.
- Return JSON only.

Problem:
{problem}

Previous steps:
{json.dumps(selected_previous, ensure_ascii=False, indent=2)}

Current step:
{current_step}
"""
    start = time.perf_counter()
    response = call_llm_structured(
        prompt,
        api_config=api_config,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "processbench_atomic_claims",
                "schema": _CLAIM_EXTRACTION_SCHEMA,
                "strict": True,
            },
        },
    )
    elapsed = time.perf_counter() - start
    claims = response.get("claims", []) if isinstance(response, dict) else []
    normalized = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        kind = str(claim.get("kind", "")).strip()
        if kind not in {"non_claim", "given_fact", "derived"}:
            continue
        conclusion = str(claim.get("conclusion", "")).strip()
        if not conclusion:
            continue
        premises = claim.get("premises", [])
        if not isinstance(premises, list):
            premises = [str(premises)]
        normalized.append(
            {
                "kind": kind,
                "premises": [str(p).strip() for p in premises if str(p).strip()],
                "conclusion": conclusion,
            }
        )
    if not normalized:
        normalized = [{"kind": "non_claim", "premises": [], "conclusion": current_step.strip()}]

    cache_payload = {
        "claims": normalized,
        "elapsed_s": elapsed,
    }
    if not args.no_claim_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(cache_payload, f, ensure_ascii=False, indent=2)
    return normalized, {"cache_hit": False, "elapsed_s": elapsed}


def _is_invalid_claim_score(scored: dict[str, Any]) -> bool:
    reason = str(scored.get("reason", ""))
    return reason in {
        "declaration_failed",
        "format_failed",
        "unknown_identifier",
        "invalid_translation",
        "z3_error",
    }


def _score_atomic_step(
    problem: str,
    previous_steps: list[str],
    current_step: str,
    args: argparse.Namespace,
    api_config: dict[str, Any],
    claim_api_config: dict[str, Any],
) -> dict[str, Any]:
    claims, extraction_debug = _extract_atomic_claims(
        problem,
        previous_steps,
        current_step,
        args,
        claim_api_config,
    )
    claim_scores = []
    step_score = 1.0
    elapsed = float(extraction_debug.get("elapsed_s", 0.0) or 0.0)
    for claim_idx, claim in enumerate(claims):
        scored = _score_claim(
            problem,
            previous_steps,
            claim,
            api_config,
            args.max_previous_steps,
        )
        elapsed += float(scored.get("elapsed_s", 0.0) or 0.0)
        claim_scores.append({"claim_index": claim_idx, "claim": claim, **scored})
        if claim.get("kind") == "derived" and scored["score"] < args.threshold:
            if args.invalid_claim_policy == "ignore" and _is_invalid_claim_score(scored):
                continue
            step_score = 0.0
            if args.stop_at_first_claim_error:
                break
    return {
        "score": step_score,
        "elapsed_s": elapsed,
        "reason": "atomic_claim_fail" if step_score < args.threshold else "atomic_claims_pass",
        "adapter_debug": extraction_debug,
        "claim_scores": claim_scores,
    }


def _processbench_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct_rows = [row for row in rows if row["gold_label"] == -1]
    error_rows = [row for row in rows if row["gold_label"] != -1]
    exact = sum(row["pred_label"] == row["gold_label"] for row in rows)
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
        "total": total,
        "exact_acc": exact / total if total else 0.0,
        "correct_acc": correct_acc,
        "error_acc": error_acc,
        "simple_f1_score": simple_f1,
        "correct_count": len(correct_rows),
        "error_count": len(error_rows),
        "avg_elapsed_s": sum(row["elapsed_s"] for row in rows) / total if total else 0.0,
        "avg_steps_scored": sum(len(row["step_scores"]) for row in rows) / total if total else 0.0,
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


def _processbench_direct_metrics(
    rows: list[dict[str, Any]],
    threshold: float,
    auto_threshold: bool,
) -> dict[str, Any]:
    y_true: list[bool] = []
    y_score: list[float] = []
    reason_counts: dict[str, int] = {}
    for row in rows:
        gold_label = int(row["gold_label"])
        for step in row["step_scores"]:
            step_idx = int(step["step_index"])
            if gold_label != -1 and step_idx > gold_label:
                continue
            y_true.append(gold_label == -1 or step_idx < gold_label)
            y_score.append(float(step["score"]))
            reason = str(step.get("reason", ""))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    selected_threshold = threshold
    best_f1 = None
    if auto_threshold and y_score:
        selected_threshold, best_f1 = _best_threshold_for_f1(y_true, y_score)

    metrics = _binary_metrics(y_true, y_score, selected_threshold) if y_score else {}
    metrics["auroc"] = _auroc(y_true, y_score) if y_score else None
    metrics["num_step_labels"] = len(y_true)
    metrics["reason_counts"] = reason_counts
    if best_f1 is not None:
        metrics["best_f1"] = best_f1
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Qwen/ProcessBench")
    parser.add_argument(
        "--split",
        default="gsm8k",
        help="ProcessBench subset split: gsm8k, math, olympiadbench, or omnimath.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--adapter",
        choices=["raw", "atomic_llm"],
        default="raw",
        help="raw verifies the whole paragraph as one conclusion; atomic_llm extracts atomic claims first.",
    )
    parser.add_argument(
        "--max_previous_steps",
        type=int,
        default=-1,
        help="-1 uses all previous steps, 0 uses only the problem, N uses the last N previous steps.",
    )
    parser.add_argument(
        "--stop_at_first_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop scoring later steps once the first failed step is found.",
    )
    parser.add_argument(
        "--score_until_gold_error",
        action="store_true",
        help="Score all steps up to the gold first-error step for FoVer/ProcessBench AUROC.",
    )
    parser.add_argument(
        "--auto_threshold",
        action="store_true",
        help="Also report the threshold that maximizes step-level F1 on this split.",
    )
    parser.add_argument("--output", default="probe_outputs/processbench_fol_probe.jsonl")
    parser.add_argument(
        "--write_incremental",
        action="store_true",
        help="Write each sample row immediately so long probes keep partial results.",
    )
    parser.add_argument("--claim_cache_dir", default="probe_outputs/processbench_claim_cache")
    parser.add_argument("--no_claim_cache", action="store_true")
    parser.add_argument("--claim_max_tokens", type=int, default=768)
    parser.add_argument(
        "--invalid_claim_policy",
        choices=["fail", "ignore"],
        default="fail",
        help="Whether invalid FOL translations count as failed claims in atomic_llm mode.",
    )
    parser.add_argument(
        "--stop_at_first_claim_error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--model", default=os.environ.get("FOL_MODEL", "Qwen3.6-35B-A3B"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:4874/v1"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--api_timeout", type=float, default=600)
    parser.add_argument("--fol_timeout", type=float, default=10)
    parser.add_argument("--max_tries", type=int, default=1)
    parser.add_argument("--old_max_tries", type=int, default=0)
    parser.add_argument("--outlines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shrink_min_tokens", type=int, default=16)
    parser.add_argument("--shrink_retries", type=int, default=6)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    if args.offset:
        ds = ds.select(range(args.offset, len(ds)))
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    api_config = _api_config(args)
    claim_api_config = _claim_extraction_api_config(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    output_file = output.open("w", encoding="utf-8") if args.write_incremental else None
    for row in tqdm(ds, desc=f"ProcessBench/{args.split}", unit="sample"):
        problem = row["problem"]
        steps = list(row["steps"])
        gold_label = int(row["label"])
        if args.score_until_gold_error and gold_label != -1:
            steps_to_score = steps[: gold_label + 1]
        else:
            steps_to_score = steps
        step_scores = []
        pred_label = -1
        sample_start = time.perf_counter()
        for step_idx, step in enumerate(steps_to_score):
            if args.adapter == "atomic_llm":
                scored = _score_atomic_step(
                    problem,
                    steps[:step_idx],
                    step,
                    args,
                    api_config,
                    claim_api_config,
                )
            else:
                scored = _score_step(
                    problem,
                    steps[:step_idx],
                    step,
                    api_config,
                    args.max_previous_steps,
                )
            step_scores.append({"step_index": step_idx, **scored})
            if pred_label == -1 and scored["score"] < args.threshold:
                pred_label = step_idx
                if args.stop_at_first_error and not args.score_until_gold_error:
                    break
        result_row = {
            "dataset": "processbench",
            "subset": args.split,
            "id": row.get("id"),
            "gold_label": gold_label,
            "pred_label": pred_label,
            "final_answer_correct": row.get("final_answer_correct"),
            "elapsed_s": time.perf_counter() - sample_start,
            "adapter": args.adapter,
            "step_scores": step_scores,
        }
        rows.append(result_row)
        if output_file is not None:
            output_file.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            output_file.flush()

    if output_file is not None:
        output_file.close()
    else:
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = _processbench_metrics(rows)
    metrics.update(
        {
            "dataset": "processbench",
            "subset": args.split,
            "threshold": args.threshold,
            "adapter": args.adapter,
            "score_until_gold_error": args.score_until_gold_error,
            "direct_step_metrics": _processbench_direct_metrics(
                rows,
                args.threshold,
                args.auto_threshold,
            ),
        }
    )
    metrics_path = output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote per-sample rows to {args.output}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
