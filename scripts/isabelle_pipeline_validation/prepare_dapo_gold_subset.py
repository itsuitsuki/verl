"""Select a DAPO subset for independent gold-step annotation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet(args.parquet)
    if args.count <= 0:
        raise ValueError("count must be positive")
    selected = frame.sample(n=min(args.count, len(frame)), random_state=args.seed)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for position, (_, row) in enumerate(selected.iterrows()):
            extra = row["extra_info"]
            reward_model = row["reward_model"]
            handle.write(json.dumps({
                "annotation_id": position,
                "dataset_index": int(row["extra_info"]["index"]),
                "problem": str(extra.get("math_question") or extra.get("question") or ""),
                "ground_truth": str(
                    extra.get("math_final_answer")
                    or extra.get("answer")
                    or reward_model["ground_truth"]
                ),
                "domain": "",
                "difficulty": "",
                "gold_steps": [],
                "review_status": "unreviewed",
                "review_notes": "",
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(selected), "out": str(output)}))


if __name__ == "__main__":
    main()
