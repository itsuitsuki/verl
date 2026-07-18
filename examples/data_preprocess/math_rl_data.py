"""Shared helpers for auditable mathematical RL dataset preprocessing."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import datasets
import pandas as pd

from bigmath_quality import build_group_flags, clean_record, exclusion_flags


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(records: list[dict], path: Path) -> None:
    if records:
        datasets.Dataset.from_list(records).to_parquet(str(path))
    else:
        pd.DataFrame().to_parquet(path, index=False)


def load_prompt_file(path_or_name: str | None, prompt_dir: Path) -> str | None:
    if not path_or_name:
        return None
    path = Path(path_or_name)
    if not path.is_absolute():
        path = prompt_dir / path
    return path.read_text(encoding="utf-8").strip()


def make_verl_row(record: dict, index: int, data_source: str,
                  system_prompt: str | None, instruction: str) -> dict:
    question = record["prompt"].strip()
    answer = record["answer"].strip()
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({
        "role": "user",
        "content": f"<Question>\n{question}\n</Question>\n\n{instruction}",
    })
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        # This struct must stay identical across all files in train_files.
        "extra_info": {
            "split": "train",
            "index": index,
            "answer": answer,
            "question": question,
            "math_question": question,
            "math_solution": "",
            "math_final_answer": answer,
            "fol_context": "",
            "fol_question": question,
            "fol_options": "",
        },
    }


def process_records(records: list[dict], save_dir: Path, data_source: str,
                    source_description: str, system_prompt: str | None,
                    instruction: str, stage_counts: dict | None = None) -> dict:
    """Quality-filter canonical records and write one dataset's outputs."""
    group_flags = build_group_flags(records)
    clean_records = []
    quarantine = []
    flag_counts = Counter()
    for record in records:
        flags = clean_record(record, group_flags)
        flag_counts.update(flags)
        excluded = exclusion_flags(flags)
        if excluded:
            quarantine.append({**record, "flags": flags, "decision": "DROP",
                               "reason": "|".join(excluded)})
        else:
            clean_records.append(record)

    train = [make_verl_row(record, i, data_source, system_prompt, instruction)
             for i, record in enumerate(clean_records)]
    # Dataset-specific metadata stays out of the training schema so concatenating
    # these parquets with the unchanged GSM8K/MATH files remains valid.
    metadata = [{"index": i, **record} for i, record in enumerate(clean_records)]

    save_dir.mkdir(parents=True, exist_ok=True)
    train_path = save_dir / "train.parquet"
    quarantine_path = save_dir / "quarantine.parquet"
    metadata_path = save_dir / "metadata.parquet"
    write_dataset(train, train_path)
    write_dataset(quarantine, quarantine_path)
    write_dataset(metadata, metadata_path)

    report = {
        "source": source_description,
        "data_source": data_source,
        "counts": {
            **(stage_counts or {}),
            "quality_input": len(records),
            "clean_output": len(train),
            "quarantine": len(quarantine),
        },
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "train_sha256": sha256(train_path),
        "quarantine_sha256": sha256(quarantine_path),
        "metadata_sha256": sha256(metadata_path),
    }
    (save_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
