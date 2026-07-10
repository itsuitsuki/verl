# Adapted from verl's examples/data_preprocess/gsm8k.py (Apache License 2.0).
# New file for the verl-fol fork; not an original Bytedance work.
"""
Preprocess the Hendrycks MATH dataset (DigitalLearningGmbH/MATH-lighteval)
to the Isabelle/FOL step-reward parquet format.

Mirrors examples/data_preprocess/gsm8k.py exactly (same extra_info schema,
same system-prompt injection, same reward_model.ground_truth) so the same
step reward manager, the same math_reasoning.txt XML-step system prompt, and
fol_task_type=math (Isabelle) work unchanged. The only differences vs gsm8k.py:
  - source dataset + fields (problem / solution instead of question / answer)
  - final-answer extraction pulls the last \\boxed{...} from the solution
The extra_info struct is kept BYTE-IDENTICAL to data/gsm8k (no extra fields):
pyarrow refuses to concat parquets whose extra_info structs differ, which
breaks multi-parquet data.train_files lists. Filter by MATH level/subject at
preprocess time here if ever needed (fields available in the raw dataset).

MATH train = 7,500 / test = 5,000. MATH-500 is a 500-problem subset of the
TEST split (Lightman et al. PRM800K) and is therefore disjoint from train, so
training on this train split does not leak MATH-500. Build the MATH-500 eval
parquet separately from HuggingFaceH4/MATH-500 if you want the fast eval.

Usage (matches prepare_gsm8k.sh style):
  python examples/data_preprocess/math_lighteval.py \
      --local_save_dir ./data/math \
      --format xml --answer_format boxed \
      --system_prompt_file math_reasoning.txt
"""

import argparse
import os
from pathlib import Path

import datasets
from tqdm.auto import tqdm

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"


def _load_prompt_file(path_or_name: str) -> str:
    p = Path(path_or_name)
    if not p.is_absolute():
        p = PROMPT_DIR / p
    return p.read_text(encoding="utf-8").strip()


def extract_solution(solution_str):
    """Final answer = contents of the last \\boxed{...} in the MATH solution."""
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def make_map_fn(split, format="flat", system_prompt=None, user_prompt_suffix=None, answer_format="boxed"):
    if answer_format == "boxed":
        instruction_following = r"Let's think step by step and output the final answer in \boxed{}."
    else:
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    def process_fn(example, idx):
        question_raw = example.pop("problem")
        solution_raw = example.pop("solution")
        final_answer = extract_solution(solution_raw)

        if format == "xml":
            prompt_content = f"<Question>\n{question_raw}\n</Question>\n\n{instruction_following}"
        else:
            prompt_content = f"{question_raw} {instruction_following}"

        if user_prompt_suffix:
            prompt_content = f"{prompt_content}\n\n{user_prompt_suffix}"

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_content})

        data_source = "DigitalLearningGmbH/MATH-lighteval"
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": final_answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": solution_raw,
                "question": question_raw,
                "math_question": question_raw,
                "math_solution": solution_raw,
                "math_final_answer": final_answer,
                # NOTE: no math_level / math_subject here — the extra_info
                # struct must be byte-identical to data/gsm8k so pyarrow can
                # concat them in a multi-parquet data.train_files list.
                # Filter by level/subject at preprocess time if ever needed.
                # Keep these structured fields compatible with the FOL extractor.
                "fol_context": "",
                "fol_question": question_raw,
                "fol_options": "",
            },
        }
        return data

    return process_fn


def map_with_progress(dataset, split, format="flat", system_prompt=None, user_prompt_suffix=None, answer_format="boxed"):
    mapper = make_map_fn(
        split,
        format=format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
        answer_format=answer_format,
    )
    rows = []
    skipped = 0
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split}", unit="example")):
        row = mapper(dict(example), idx)
        if row["reward_model"]["ground_truth"] is None:
            skipped += 1
            continue
        rows.append(row)
    if skipped:
        print(f"  [{split}] skipped {skipped} rows with no extractable \\boxed answer")
    return datasets.Dataset.from_list(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="Local path to the raw dataset, if it exists.")
    parser.add_argument("--local_save_dir", default="~/data/math", help="Save directory for the preprocessed dataset.")
    parser.add_argument("--format", default="xml", choices=["flat", "xml"], help="Prompt format.")
    parser.add_argument(
        "--answer_format",
        default="boxed",
        choices=["hash", "boxed"],
        help="Final answer format requested from the actor (MATH uses boxed).",
    )
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Path or bare filename under verl/prompts/ for the system prompt (use math_reasoning.txt for the XML-step format).",
    )
    parser.add_argument(
        "--user_prompt_file",
        default=None,
        help="Path or bare filename under verl/prompts/ to append to the user prompt.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    system_prompt = _load_prompt_file(args.system_prompt_file) if args.system_prompt_file else None
    user_prompt_suffix = _load_prompt_file(args.user_prompt_file) if args.user_prompt_file else None

    # 'lighteval/MATH' was removed from HF; the DigitalLearningGmbH mirror is the
    # standard replacement (train 7,500 / test 5,000).
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading {data_source} ...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = map_with_progress(
        train_dataset,
        "train",
        format=args.format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
        answer_format=args.answer_format,
    )
    test_dataset = map_with_progress(
        test_dataset,
        "test",
        format=args.format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
        answer_format=args.answer_format,
    )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Dataset saved to {local_save_dir}")
    print(f"Data source: {data_source}")
    print(f"Answer format: {args.answer_format}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
