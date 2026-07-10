#!/usr/bin/env python3
"""Probe the current FOL reward pipeline on FOLIO and FLDx2.

This script intentionally measures only entailment-style behavior:
given a problem context and a hypothesis/conclusion, does the current
FOL translator + Z3 verifier return an entailed step?
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from verl.utils.reward_score.formal_verify import compute_step_reward_fol


def _api_config(args: argparse.Namespace, task_type: str = "logic") -> dict[str, Any]:
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
        "fol_task_type": task_type,
        "fol_cumulative_mode": "current_only",
        "fol_judge_use_outlines": args.outlines,
        "api_context_shrink_min_tokens": args.shrink_min_tokens,
        "api_context_shrink_retries": args.shrink_retries,
    }


def _problem_prompt(context: str, question: str, options: str = "") -> str:
    options_block = f"\n\n<Options>\n{options}\n</Options>" if options else ""
    return (
        f"<Context>\n{context.strip()}\n</Context>\n\n"
        f"<Question>\n{question.strip()}\n</Question>{options_block}"
    )


def _xml_step(premises: list[str], conclusion: str) -> str:
    premise_lines = "\n".join(f"<premise>{p.strip()}</premise>" for p in premises if p and p.strip())
    return f"<step>\n{premise_lines}\n<conclusion>{conclusion.strip()}</conclusion>\n</step>"


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
        return output.strip().splitlines()[-1][:120]
    return ""


def _score_one(
    context: str,
    question: str,
    premises: list[str],
    conclusion: str,
    api_config: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_text = _problem_prompt(context, question)
    step_text = _xml_step(premises, conclusion)
    start = time.perf_counter()
    result = compute_step_reward_fol(
        step_text,
        prompt_text,
        [step_text],
        api_config=api_config,
        extra_info=extra_info or {
            "fol_context": context,
            "fol_question": question,
            "fol_options": "",
        },
        return_debug=True,
    )
    elapsed = time.perf_counter() - start
    debug = result.get("debug", {}) if isinstance(result, dict) else {}
    score = float(result.get("score", 0.0)) if isinstance(result, dict) else float(result)
    return {
        "score": score,
        "pred_entailed": score >= 0.5,
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


def _limit_dataset(ds, limit: int, offset: int = 0):
    if offset:
        ds = ds.select(range(offset, len(ds)))
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def probe_folio(args: argparse.Namespace) -> dict[str, Any]:
    ds = load_dataset(args.folio_dataset, split=args.folio_split)
    ds = _limit_dataset(ds, args.folio_limit, args.folio_offset)
    api_config = _api_config(args, task_type="logic")
    rows = []
    for row in tqdm(ds, desc="FOLIO", unit="sample"):
        premises_text = row.get("premises") or ""
        premises = [p for p in premises_text.splitlines() if p.strip()]
        conclusion = row.get("conclusion") or ""
        gold_label = str(row.get("label", "")).strip()
        scored = _score_one(
            context=premises_text,
            question=conclusion,
            premises=premises,
            conclusion=conclusion,
            api_config=api_config,
            extra_info={
                "fol_context": premises_text,
                "fol_question": conclusion,
                "fol_options": "",
            },
        )
        gold_entailed = gold_label.lower() == "true"
        rows.append(
            {
                "dataset": "folio",
                "id": row.get("example_id"),
                "gold_label": gold_label,
                "gold_entailed": gold_entailed,
                **scored,
            }
        )
    return _summarize_binary("folio", rows)


def probe_fldx2(args: argparse.Namespace) -> dict[str, Any]:
    ds = load_dataset(args.fldx2_dataset, split=args.fldx2_split)
    ds = _limit_dataset(ds, args.fldx2_limit, args.fldx2_offset)
    api_config = _api_config(args, task_type="logic")
    rows = []
    for idx, row in enumerate(tqdm(ds, desc="FLDx2", unit="sample")):
        facts_text = row.get("facts") or ""
        hypothesis = row.get("hypothesis") or ""
        proof_label = str(row.get("proof_label", "")).strip().upper()
        facts = [part.strip() for part in facts_text.split(" fact") if part.strip()]
        if facts_text.startswith("fact") and facts:
            facts = [facts[0], *[f"fact{part}" for part in facts[1:]]]
        if args.fldx2_max_facts and args.fldx2_max_facts > 0:
            facts = facts[: args.fldx2_max_facts]
        context = " ".join(facts) if facts else facts_text
        scored = _score_one(
            context=context,
            question=hypothesis,
            premises=facts or [facts_text],
            conclusion=hypothesis,
            api_config=api_config,
            extra_info={
                "fol_context": context,
                "fol_question": hypothesis,
                "fol_options": "",
            },
        )
        gold_entailed = proof_label == "PROVED"
        rows.append(
            {
                "dataset": "fldx2",
                "id": idx,
                "gold_label": proof_label,
                "gold_entailed": gold_entailed,
                **scored,
            }
        )
    return _summarize_binary("fldx2", rows)


def _summarize_binary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid_reasons = {"", "entailed", "not_entailed"}
    valid_rows = [row for row in rows if row["reason"] in valid_reasons]
    invalid = total - len(valid_rows)
    correct = sum(row["pred_entailed"] == row["gold_entailed"] for row in valid_rows)
    positives = [row for row in valid_rows if row["gold_entailed"]]
    negatives = [row for row in valid_rows if not row["gold_entailed"]]
    tp = sum(row["pred_entailed"] for row in positives)
    tn = sum(not row["pred_entailed"] for row in negatives)
    fp = sum(row["pred_entailed"] for row in negatives)
    fn = sum(not row["pred_entailed"] for row in positives)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "dataset": name,
        "total": total,
        "valid_total": len(valid_rows),
        "coverage": len(valid_rows) / total if total else 0.0,
        "accuracy": correct / len(valid_rows) if valid_rows else 0.0,
        "positive_recall": recall,
        "positive_precision": precision,
        "positive_f1": f1,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "invalid_or_error_rate": invalid / total if total else 0.0,
        "avg_elapsed_s": sum(row["elapsed_s"] for row in rows) / total if total else 0.0,
        "rows": rows,
    }
    return summary


def _write_jsonl(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for summary in summaries:
            for row in summary["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["folio", "fldx2", "both"], default="both")
    parser.add_argument("--folio_dataset", default="tasksource/folio")
    parser.add_argument("--folio_split", default="validation")
    parser.add_argument("--folio_limit", type=int, default=10)
    parser.add_argument("--folio_offset", type=int, default=0)
    parser.add_argument("--fldx2_dataset", default="hitachi-nlp/FLDx2")
    parser.add_argument("--fldx2_split", default="validation")
    parser.add_argument("--fldx2_limit", type=int, default=10)
    parser.add_argument("--fldx2_offset", type=int, default=0)
    parser.add_argument("--fldx2_max_facts", type=int, default=12)
    parser.add_argument("--output", default="probe_outputs/folio_fldx2_fol_probe.jsonl")
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

    summaries = []
    if args.bench in {"folio", "both"}:
        summaries.append(probe_folio(args))
    if args.bench in {"fldx2", "both"}:
        summaries.append(probe_fldx2(args))

    _write_jsonl(Path(args.output), summaries)
    printable = []
    for summary in summaries:
        printable.append({k: v for k, v in summary.items() if k != "rows"})
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"Wrote per-sample rows to {args.output}")


if __name__ == "__main__":
    main()
