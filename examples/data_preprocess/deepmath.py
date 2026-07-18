# New file for the verl-fol fork; follows examples/data_preprocess/math_lighteval.py.
"""Preprocess DeepMath-103K for Isabelle Step-GDPO."""
import argparse
from pathlib import Path

from datasets import load_dataset

from math_rl_data import load_prompt_file, process_records

PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"
DATASET_ID = "zwhe99/DeepMath-103K"
DATA_SOURCE = "zwhe99/DeepMath-103K"
INSTRUCTION = r"Let's think step by step and output the final answer in \boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="./data/deepmath")
    parser.add_argument("--system_prompt_file", default="math_reasoning.txt")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--difficulty_min", type=float, default=None)
    parser.add_argument("--difficulty_max", type=float, default=None)
    args = parser.parse_args()

    if args.local_dataset_path:
        dataset = load_dataset("parquet", data_files=args.local_dataset_path,
                               split="train")
        source_description = args.local_dataset_path
    else:
        dataset = load_dataset(DATASET_ID, split="train")
        source_description = DATASET_ID

    records = []
    difficulty_excluded = []
    for source_index, row in enumerate(dataset):
        difficulty = row.get("difficulty")
        if ((args.difficulty_min is not None and difficulty < args.difficulty_min)
                or (args.difficulty_max is not None and difficulty > args.difficulty_max)):
            difficulty_excluded.append({
                "source_index": source_index,
                "prompt": str(row.get("question") or ""),
                "answer": str(row.get("final_answer") or ""),
                "difficulty": difficulty,
                "topic": str(row.get("topic") or ""),
                "reason": "difficulty_outside_requested_range",
            })
            continue
        records.append({
            "source_index": source_index,
            "prompt": str(row.get("question") or ""),
            "answer": str(row.get("final_answer") or ""),
            "source": DATA_SOURCE,
            "domain": str(row.get("topic") or ""),
            "difficulty": difficulty,
            "topic": str(row.get("topic") or ""),
            "r1_solution_1": str(row.get("r1_solution_1") or ""),
            "r1_solution_2": str(row.get("r1_solution_2") or ""),
            "r1_solution_3": str(row.get("r1_solution_3") or ""),
        })

    save_dir = Path(args.local_save_dir).expanduser()
    process_records(
        records=records,
        save_dir=save_dir,
        data_source=DATA_SOURCE,
        source_description=source_description,
        system_prompt=load_prompt_file(args.system_prompt_file, PROMPT_DIR),
        instruction=INSTRUCTION,
        stage_counts={"raw": len(dataset),
                      "difficulty_excluded": len(difficulty_excluded)},
    )
    if difficulty_excluded:
        from math_rl_data import write_dataset
        write_dataset(difficulty_excluded, save_dir / "difficulty_excluded.parquet")


if __name__ == "__main__":
    main()
