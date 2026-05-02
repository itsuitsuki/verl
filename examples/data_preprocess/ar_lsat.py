"""
Preprocess the AR-LSAT dataset to parquet format for verl.

The olegbask/AR-LSAT dataset follows the same multiple-choice schema as
ReClor: context, question, answers, and a numeric label. Unlike ReClor, its
test split is labeled, so this script emits train/validation/test by default.
"""

import argparse
import os
from pathlib import Path

import datasets
from tqdm.auto import tqdm


PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _load_prompt_file(path_or_name: str) -> str:
    """Load a prompt txt from an absolute path or verl/prompts/."""
    p = Path(path_or_name)
    if not p.is_absolute():
        p = PROMPT_DIR / p
    return p.read_text(encoding="utf-8").strip()


def make_map_fn(split, format="xml", system_prompt=None, user_prompt_suffix=None):
    def process_fn(example, idx):
        context = example.get("context", "")
        question_text = example.get("question", "")
        options = example.get("answers", [])

        answer_idx = int(example.get("label"))
        ground_truth = LABELS[answer_idx] if answer_idx < len(LABELS) else "A"

        options_str = "\n".join(
            f"Option ({LABELS[i]}): {opt}" for i, opt in enumerate(options) if i < len(LABELS)
        )
        instruction_following = (
            'Please reason step by step with steps separated by "\\n\\n", '
            "and put the letter of the correct option within \\boxed{{}}."
        )

        if format == "xml":
            prompt_content = (
                f"<Context>\n{context}\n</Context>\n\n"
                f"<Question>\n{question_text}\n</Question>\n\n"
                f"<Options>\n{options_str}\n</Options>\n\n"
                f"{instruction_following}"
            )
        else:
            prompt_content = (
                f"Context: {context}\n\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"{instruction_following}"
            )

        if user_prompt_suffix:
            prompt_content = f"{prompt_content}\n\n{user_prompt_suffix}"

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_content})

        return {
            "data_source": "ar_lsat",
            "prompt": prompt_messages,
            "ability": "logic",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": ground_truth,
                "fol_context": context,
                "fol_question": question_text,
                "fol_options": options_str,
            },
        }

    return process_fn


def map_with_progress(dataset, split, format, system_prompt=None, user_prompt_suffix=None):
    mapper = make_map_fn(
        split,
        format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
    )
    rows = []
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split}", unit="example")):
        rows.append(mapper(example, idx))
    return datasets.Dataset.from_list(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="./data/ar_lsat_prompt_v2",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument("--format", default="xml", choices=["flat", "xml"], help="Prompt format.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of train samples to keep.")
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Path or bare filename under verl/prompts/ for the system prompt.",
    )
    parser.add_argument(
        "--user_prompt_file",
        default=None,
        help="Path or bare filename under verl/prompts/ to append to the user prompt.",
    )
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    system_prompt = _load_prompt_file(args.system_prompt_file) if args.system_prompt_file else None
    user_prompt_suffix = _load_prompt_file(args.user_prompt_file) if args.user_prompt_file else None

    dataset = datasets.load_dataset("olegbask/AR-LSAT")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    if args.num_samples is not None and args.num_samples != -1:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.num_samples)))

    train_dataset = map_with_progress(
        train_dataset,
        "train",
        args.format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
    )
    val_dataset = map_with_progress(
        val_dataset,
        "validation",
        args.format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
    )
    test_dataset = map_with_progress(
        test_dataset,
        "test",
        args.format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
    )

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Dataset saved to {local_save_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
