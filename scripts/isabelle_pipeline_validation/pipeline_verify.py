"""Run DAPO policy rollouts through the production Isabelle reward path.

The script first applies the same training-time format policy as
``StepRewardManager``: token-limit truncation skips verification, malformed XML
steps and steps after the configured cap are penalized separately and removed,
and a response with several boxed answers is cut after the first one. The
remaining valid steps then pass through ``process_one_response``.

The output retains policy-level format metadata so later analysis can separate
policy formatting, translator acceptance, Isabelle proof capability, and
process rewards. Existing output is resumed automatically by ``response_id``.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verl.experimental.reward_loop.reward_manager.format_penalty import (
    assess_response,
    verifier_response,
)
from verl.utils.step_splitter import analyze_xml_steps, boxed_spans


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _prepare_training_response(row: dict, penalty_max_steps: int) -> tuple[str | None, dict]:
    response = str(row.get("response") or "")
    reports = analyze_xml_steps(response)
    spans = boxed_spans(response)
    finish_reason = str(row.get("finish_reason") or "unknown")
    metadata = {
        "policy_finish_reason": finish_reason,
        "policy_boxed_count": len(spans),
        "policy_xml_steps": len(reports),
        "policy_valid_xml_steps": sum(report.valid for report in reports),
        "policy_invalid_xml_steps": sum(not report.valid for report in reports),
        "policy_xml_defects": {
            reason: sum(report.error == reason for report in reports)
            for reason in sorted({report.error for report in reports if report.error is not None})
        },
    }

    if finish_reason == "length":
        metadata.update({
            "training_skip_verifier": True,
            "training_penalty_reason": "truncated",
            "training_penalty_step_indices": list(range(len(reports))),
            "training_verifier_policy_step_indices": [],
        })
        return None, metadata

    decision = assess_response(
        response,
        [(report.text, index) for index, report in enumerate(reports)],
        use_xml=True,
        valid_response_length=1,
        response_length=2,
        penalty_score=-1.0,
        penalty_max_steps=penalty_max_steps,
        penalty_on_truncated=True,
        penalty_on_multi_boxed=True,
        penalty_on_bad_format=True,
    )
    penalty_indices = set(decision.penalty_step_indices)
    verifier_indices = [index for index in range(len(reports)) if index not in penalty_indices]
    metadata.update({
        "training_skip_verifier": bool(decision.skip_verifier),
        "training_penalty_reason": "|".join(decision.reasons),
        "training_penalty_step_indices": sorted(penalty_indices),
        "training_verifier_policy_step_indices": verifier_indices,
    })
    if decision.skip_verifier:
        return None, metadata

    # Strip every penalized block, including valid blocks beyond the step cap.
    # The production manager discards those results; removing them here avoids
    # spending translator and prover time on results that cannot become rewards.
    penalized_ranges = {(reports[index].start, reports[index].end) for index in penalty_indices}
    decision.invalid_char_ranges = sorted(set(decision.invalid_char_ranges) | penalized_ranges)
    return verifier_response(response, decision), metadata


def _skipped_record(response_id: int, row: dict, metadata: dict) -> dict:
    return {
        "response_id": response_id,
        "dataset": row.get("dataset", "?"),
        "idx": row.get("idx"),
        "sample": row.get("sample", 0),
        "format_ok": False,
        "givens_ok": False,
        "steps_ok": False,
        "n_steps": 0,
        "pattern": "",
        "steps": [],
        "wall": 0.0,
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--judges", default="http://127.0.0.1:4873/v1,http://127.0.0.1:4874/v1")
    parser.add_argument("--pool-workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--penalty-max-steps", type=int, default=30)
    args = parser.parse_args()

    rollout_path = Path(args.rollouts)
    rows = _load_jsonl(rollout_path)
    prepared = [
        _prepare_training_response(row, max(0, args.penalty_max_steps))
        for row in rows
    ]
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.out)
    completed = set()
    if output_path.exists():
        for record in _load_jsonl(output_path):
            completed.add(int(record["response_id"]))
    pending = [index for index in range(len(rows)) if index not in completed]
    print(
        f"rollouts={len(rows)} completed={len(completed)} pending={len(pending)} "
        f"training_skipped={sum(response is None for response, _ in prepared)}",
        flush=True,
    )
    if not pending:
        print(f"DONE verify -> {output_path} (already complete)", flush=True)
        return

    engine_indices = [index for index in pending if prepared[index][0] is not None]
    engine = None
    process_one_response = None
    if engine_indices:
        api_config = {
            "base_url": args.judges,
            "model": "Qwen3.6-35B-A3B",
            "timeout": 60,
            "api_timeout": 200,
            "isabelle_pool_workers": args.pool_workers,
            "isabelle_worker_rss_cap_gb": 12,
        }
        from verl.utils.reward_score.formal_verify import _get_isabelle_engine
        from verl.utils.isabelle_utils.engine import process_one_response as process_response

        engine = _get_isabelle_engine(api_config)
        process_one_response = process_response

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = output_path.open("a", encoding="utf-8")
    write_lock = threading.Lock()
    started = time.time()
    done = [0]

    def work(index: int) -> None:
        row = rows[index]
        verifier_response_text, metadata = prepared[index]
        item_started = time.time()
        if verifier_response_text is None:
            record = _skipped_record(index, row, metadata)
        else:
            item = {
                "problem": row["problem"],
                "response": verifier_response_text,
                "ground_truth": row["ground_truth"],
                "dataset": row["dataset"],
                "idx": row["idx"],
                "sample": row["sample"],
            }
            try:
                record = process_one_response(
                    index,
                    item,
                    engine.pool,
                    engine.config,
                    outdir=debug_dir,
                    max_steps=0,
                )
            except Exception as exc:  # noqa: BLE001
                record = {"error": repr(exc), "format_ok": None}
            record.update(metadata)
            record["response_id"] = index
            record["dataset"] = row["dataset"]
            record["idx"] = row["idx"]
            record["sample"] = row["sample"]
            record["wall"] = round(time.time() - item_started, 2)

        with write_lock:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            output.flush()
            done[0] += 1
            current = done[0]
        print(
            "[%d/%d pending; %d/%d total] id=%d pattern=%r n_steps=%s wall=%.1fs elapsed=%.0fm"
            % (
                current,
                len(pending),
                len(completed) + current,
                len(rows),
                index,
                record.get("pattern"),
                record.get("n_steps"),
                record["wall"],
                (time.time() - started) / 60,
            ),
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        list(executor.map(work, pending))
    output.close()
    print(
        "DONE verify -> %s (%.0f min, %d responses)"
        % (output_path, (time.time() - started) / 60, len(rows)),
        flush=True,
    )
    if engine is not None:
        engine.shutdown()


if __name__ == "__main__":
    main()
