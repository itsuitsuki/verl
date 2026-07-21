"""Preprocess a selected Open-R1 DAPO-Math-17K dataset configuration."""
import argparse
from pathlib import Path

from datasets import load_dataset

from math_rl_data import load_prompt_file, process_records

PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"
DATASET_ID = "open-r1/DAPO-Math-17k-Processed"
DATA_SOURCE = DATASET_ID
INSTRUCTION = r"Let's think step by step and output the final answer in \boxed{}."
DATASET_CONFIGS = ("all", "en", "cn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="./data/dapo_math")
    parser.add_argument("--system_prompt_file", default="math_reasoning.txt")
    parser.add_argument(
        "--config",
        choices=DATASET_CONFIGS,
        default="en",
        help="Hugging Face configuration to preprocess; defaults to the historical English subset.",
    )
    parser.add_argument("--local_dataset_path", default=None)
    args = parser.parse_args()

    data_source = DATA_SOURCE
    if args.local_dataset_path:
        dataset = load_dataset("parquet", data_files=args.local_dataset_path,
                               split="train")
        source_description = args.local_dataset_path
    else:
        dataset = load_dataset(DATASET_ID, args.config, split="train")
        source_description = f"{DATASET_ID}:{args.config}"

    records = []
    for source_index, row in enumerate(dataset):
        records.append({
            "source_index": source_index,
            "prompt": str(row.get("prompt") or ""),
            "answer": str(row.get("solution") or ""),
            "source": str(row.get("data_source") or "math_dapo"),
            "domain": "MATH",
            "difficulty": None,
            "topic": "",
        })

    process_records(
        records=records,
        save_dir=Path(args.local_save_dir).expanduser(),
        data_source=data_source,
        source_description=source_description,
        system_prompt=load_prompt_file(args.system_prompt_file, PROMPT_DIR),
        instruction=INSTRUCTION,
        stage_counts={"raw": len(dataset), "dataset_config": args.config},
    )


if __name__ == "__main__":
    main()
