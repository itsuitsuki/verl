"""Exhaustively test the production outcome scorer on a DAPO parquet file.

DAPO-Math answers in the current processed training pool are integers. For each row,
this script grades a canonical correct answer, a nearby wrong integer, and the same
answer without a boxed marker. It reports parser/reward failures separately from
policy-answer accuracy and writes one auditable JSON record per row.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from verl.utils.reward_score.math_verify import compute_score_boxed

_INTEGER_RE = re.compile(r"^-?\d+$")


def _assert_scorer_contract() -> None:
    """Fail before the exhaustive pass if boxed construction is malformed."""
    checks = (
        compute_score_boxed("\\boxed{34}", "34"),
        compute_score_boxed("\\boxed{35}", "34"),
        compute_score_boxed("34", "34"),
    )
    if checks != (1.0, 0.0, 0.0):
        raise RuntimeError(f"unexpected outcome scorer contract: {checks!r}")


def _ground_truth(row) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("math_final_answer") is not None:
        return str(extra["math_final_answer"]).strip()
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"]).strip()
    raise ValueError("row has no integer ground truth")


def _grade(answer: str) -> dict:
    try:
        correct = compute_score_boxed("\\boxed{" + answer + "}", answer)
        wrong = str(int(answer) + 1)
        wrong_score = compute_score_boxed("\\boxed{" + wrong + "}", answer)
        missing_score = compute_score_boxed(answer, answer)
        return {
            "answer": answer,
            "canonical_correct": float(correct),
            "wrong_integer": wrong,
            "wrong_rejected": float(wrong_score),
            "missing_boxed_rejected": float(missing_score),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "answer": answer,
            "canonical_correct": None,
            "wrong_integer": None,
            "wrong_rejected": None,
            "missing_boxed_rejected": None,
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    _assert_scorer_contract()
    frame = pd.read_parquet(args.parquet)
    payloads = []
    invalid = []
    for index, row in frame.iterrows():
        try:
            answer = _ground_truth(row)
            if not _INTEGER_RE.fullmatch(answer):
                raise ValueError(f"non-integer DAPO answer: {answer!r}")
            payloads.append((int(index), answer))
        except Exception as exc:  # noqa: BLE001
            invalid.append({"index": int(index), "error": repr(exc)})

    unique_answers = sorted({answer for _, answer in payloads}, key=lambda value: (len(value), value))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        unique_results = list(executor.map(_grade, unique_answers))
    result_by_answer = {row["answer"]: row for row in unique_results}
    results = [{"index": index, **result_by_answer[answer]}
               for index, answer in payloads]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in invalid:
            handle.write(json.dumps({"status": "invalid_input", **row}) + "\n")
        for row in results:
            handle.write(json.dumps({"status": "graded", **row}) + "\n")

    graded = [row for row in results if row["error"] is None]
    summary = {
        "parquet": str(args.parquet),
        "rows": len(frame),
        "integer_inputs": len(payloads),
        "unique_integer_answers": len(unique_answers),
        "invalid_inputs": len(invalid),
        "graded": len(graded),
        "canonical_correct_accept": sum(row["canonical_correct"] == 1.0 for row in graded),
        "wrong_integer_reject": sum(row["wrong_rejected"] == 0.0 for row in graded),
        "missing_boxed_reject": sum(row["missing_boxed_rejected"] == 0.0 for row in graded),
        "grader_errors": sum(row["error"] is not None for row in results),
        "detail_file": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
