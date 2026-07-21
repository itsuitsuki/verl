"""Analyze DAPO policy rollouts with the production format and outcome rules.

The report separates policy generation behavior, boxed-answer scoring, and the
availability of valid XML steps. It never interprets these measurements as
translator faithfulness or Isabelle false-positive/false-negative rates.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verl.utils.reward_score.math_dapo import last_boxed_only_string
from verl.utils.reward_score.math_verify import compute_score_boxed
from verl.utils.step_splitter import (
    analyze_xml_steps,
    boxed_spans,
    find_stray_xml_tags,
    parse_xml_steps,
)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _score_key(text: str, ground_truth: str) -> tuple[str | None, str]:
    return last_boxed_only_string(text), ground_truth.strip()


def _grade_key(key: tuple[str | None, str]) -> tuple[tuple[str | None, str], float]:
    boxed, ground_truth = key
    if boxed is None:
        return key, 0.0
    return key, float(compute_score_boxed(boxed, ground_truth))


def _ratio(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _counter_dict(counter: Counter) -> dict:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _summarize_group(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        counts["rows"] += 1
        counts[f"finish_{row['finish_reason']}"] += 1
        counts[f"boxed_{row['boxed_class']}"] += 1
        counts["raw_outcome_correct"] += int(row["raw_outcome_score"] == 1.0)
        counts["committed_outcome_correct"] += int(row["committed_outcome_score"] == 1.0)
        counts["raw_verifier_parseable"] += int(row["raw_verifier_parseable"])
        counts["fully_valid_xml"] += int(row["fully_valid_xml"])
        counts["has_valid_xml_step"] += int(row["valid_xml_steps"] > 0)
        counts["training_verifier_has_valid_step"] += int(row["training_verifier_has_valid_step"])
        counts["has_invalid_xml_step"] += int(row["invalid_xml_steps"] > 0)
        counts["has_stray_xml_tag"] += int(row["stray_xml_tags"] > 0)
        counts["over_step_cap"] += int(row["xml_steps"] > 30)
        counts["step_after_first_boxed"] += int(row["step_after_first_boxed"])
    total = counts["rows"]
    return {
        **dict(counts),
        "raw_outcome_accuracy_pct": _ratio(counts["raw_outcome_correct"], total),
        "committed_outcome_accuracy_pct": _ratio(counts["committed_outcome_correct"], total),
        "raw_verifier_parseable_pct": _ratio(counts["raw_verifier_parseable"], total),
        "fully_valid_xml_pct": _ratio(counts["fully_valid_xml"], total),
        "has_valid_xml_step_pct": _ratio(counts["has_valid_xml_step"], total),
        "training_verifier_has_valid_step_pct": _ratio(counts["training_verifier_has_valid_step"], total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--detail-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = _load_rows(Path(args.rollouts))
    prepared = []
    grade_keys = set()
    for row in rows:
        response = str(row.get("response") or "")
        ground_truth = str(row.get("ground_truth") or "").strip()
        spans = boxed_spans(response)
        committed_text = response[:spans[0][1]] if len(spans) > 1 else response
        raw_key = _score_key(response, ground_truth)
        committed_key = _score_key(committed_text, ground_truth)
        grade_keys.update((raw_key, committed_key))
        prepared.append((row, response, spans, raw_key, committed_key))

    ordered_keys = sorted(grade_keys, key=lambda key: (key[0] is None, len(key[0] or ""), key[0] or "", key[1]))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        scored = dict(executor.map(_grade_key, ordered_keys))

    details = []
    defect_counts = Counter()
    step_count_distribution = Counter()
    invalid_step_distribution = Counter()
    for row, response, spans, raw_key, committed_key in prepared:
        reports = analyze_xml_steps(response)
        stray_tags = find_stray_xml_tags(response)
        defects = Counter(report.error for report in reports if report.error is not None)
        defect_counts.update(defects)
        valid_steps = sum(report.valid for report in reports)
        invalid_steps = len(reports) - valid_steps
        step_count_distribution[len(reports)] += 1
        invalid_step_distribution[invalid_steps] += 1
        first_boxed_start = spans[0][0] if spans else None
        step_after_first_boxed = bool(
            first_boxed_start is not None
            and any(report.start >= first_boxed_start for report in reports)
        )
        finish_reason = str(row.get("finish_reason") or "unknown")
        boxed_class = "none" if not spans else "single" if len(spans) == 1 else "multiple"
        detail = {
            "idx": int(row["idx"]),
            "sample": int(row.get("sample", 0)),
            "dataset": str(row.get("dataset") or "?"),
            "finish_reason": finish_reason,
            "boxed_count": len(spans),
            "boxed_class": boxed_class,
            "raw_outcome_score": scored[raw_key],
            "committed_outcome_score": scored[committed_key],
            "xml_steps": len(reports),
            "valid_xml_steps": valid_steps,
            "invalid_xml_steps": invalid_steps,
            "xml_defects": dict(defects),
            "stray_xml_tags": len(stray_tags),
            "raw_verifier_parseable": parse_xml_steps(response) is not None,
            "fully_valid_xml": bool(reports) and not defects and not stray_tags,
            "training_verifier_has_valid_step": finish_reason != "length" and valid_steps > 0,
            "step_after_first_boxed": step_after_first_boxed,
        }
        details.append(detail)

    details.sort(key=lambda row: (row["idx"], row["sample"]))
    detail_path = Path(args.detail_out)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")

    grouped = defaultdict(list)
    for detail in details:
        grouped[detail["finish_reason"]].append(detail)
    summary = {
        "rollouts": args.rollouts,
        "detail_file": str(detail_path),
        "unique_outcome_grade_inputs": len(ordered_keys),
        "overall": _summarize_group(details),
        "by_finish_reason": {
            reason: _summarize_group(group_rows)
            for reason, group_rows in sorted(grouped.items())
        },
        "xml_defect_counts": _counter_dict(defect_counts),
        "xml_step_count_distribution": _counter_dict(step_count_distribution),
        "invalid_xml_step_count_distribution": _counter_dict(invalid_step_distribution),
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
