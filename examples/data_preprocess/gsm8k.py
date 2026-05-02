# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
from pathlib import Path

import datasets
from tqdm.auto import tqdm

from verl.utils.hdfs_io import copy, makedirs

PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"


def _load_prompt_file(path_or_name: str) -> str:
    p = Path(path_or_name)
    if not p.is_absolute():
        p = PROMPT_DIR / p
    return p.read_text(encoding="utf-8").strip()


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def make_map_fn(split, format="flat", system_prompt=None, user_prompt_suffix=None, answer_format="hash"):
    if answer_format == "boxed":
        instruction_following = r"Let's think step by step and output the final answer in \boxed{}."
    else:
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    def process_fn(example, idx):
        question_raw = example.pop("question")
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)

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

        data_source = "openai/gsm8k"
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
                "math_question": question_raw,
                "math_solution": answer_raw,
                "math_final_answer": solution,
                # Keep these structured fields compatible with the FOL extractor.
                "fol_context": "",
                "fol_question": question_raw,
                "fol_options": "",
            },
        }
        return data

    return process_fn


def map_with_progress(dataset, split, format="flat", system_prompt=None, user_prompt_suffix=None, answer_format="hash"):
    mapper = make_map_fn(
        split,
        format=format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
        answer_format=answer_format,
    )
    rows = []
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split}", unit="example")):
        rows.append(mapper(dict(example), idx))
    return datasets.Dataset.from_list(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--format", default="flat", choices=["flat", "xml"], help="Prompt format.")
    parser.add_argument(
        "--answer_format",
        default="hash",
        choices=["hash", "boxed"],
        help="Final answer format requested from the actor.",
    )
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
    local_dataset_path = args.local_dataset_path
    system_prompt = _load_prompt_file(args.system_prompt_file) if args.system_prompt_file else None
    user_prompt_suffix = _load_prompt_file(args.user_prompt_file) if args.user_prompt_file else None

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

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
