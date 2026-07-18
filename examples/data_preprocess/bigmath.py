# New file for the verl-fol fork; follows examples/data_preprocess/math_lighteval.py.
"""Build a conservative, auditable Big-Math training dataset."""
import argparse
from pathlib import Path

import pandas as pd

from math_rl_data import load_prompt_file, process_records, write_dataset

PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"
DATA_SOURCE = "open-r1/Big-Math-RL-Verified-Processed"
INSTRUCTION = r"Let's think step by step and output the final answer in \boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--local_save_dir", default="./data/bigmath_clean")
    parser.add_argument("--system_prompt_file", default="math_reasoning.txt")
    parser.add_argument("--solve_rate_min", type=float, default=0.0)
    parser.add_argument("--solve_rate_max", type=float, default=0.9)
    args = parser.parse_args()

    src = Path(args.raw_dir) / "all" / "train-00000-of-00001.parquet"
    raw = pd.read_parquet(src)

    # F1: these sources are already loaded as separate GSM8K/MATH parquets.
    f1_mask = raw["source"].isin(["gsm8k", "math"])
    f1_excluded = raw[f1_mask].copy()
    f1_excluded["source_index"] = f1_excluded.index
    f1_excluded["reason"] = "source_already_in_gsm8k_or_math"
    after_f1 = raw[~f1_mask]

    # F2: retain the configured RL difficulty band. This is not quality scoring.
    f2_keep = after_f1["llama8b_solve_rate"].between(
        args.solve_rate_min, args.solve_rate_max)
    f2_excluded = after_f1[~f2_keep].copy()
    f2_excluded["source_index"] = f2_excluded.index
    f2_excluded["reason"] = "solve_rate_outside_requested_range"
    selected = after_f1[f2_keep]

    records = []
    for source_index, row in selected.iterrows():
        records.append({
            "source_index": int(source_index),
            "prompt": str(row["prompt"] or ""),
            "answer": str(row["solution"] or ""),
            "source": str(row["source"] or ""),
            "domain": str(row["domain"]),
            "difficulty": float(row["llama8b_solve_rate"]),
            "topic": str(row["domain"]),
        })

    save_dir = Path(args.local_save_dir).expanduser()
    process_records(
        records=records,
        save_dir=save_dir,
        data_source=DATA_SOURCE,
        source_description=str(src),
        system_prompt=load_prompt_file(args.system_prompt_file, PROMPT_DIR),
        instruction=INSTRUCTION,
        stage_counts={
            "raw": len(raw),
            "f1_excluded": len(f1_excluded),
            "after_f1": len(after_f1),
            "f2_excluded": len(f2_excluded),
            "after_f2": len(selected),
        },
    )
    write_dataset(f1_excluded.to_dict("records"),
                  save_dir / "f1_excluded.parquet")
    write_dataset(f2_excluded.to_dict("records"),
                  save_dir / "f2_excluded.parquet")


if __name__ == "__main__":
    main()
