# New file for the verl-fol fork; follows examples/data_preprocess/math_lighteval.py.
"""
Preprocess open-r1/Big-Math-RL-Verified-Processed into the verl-fol
step-reward parquet format (train split only -- Big-Math is training data;
validation runs on the 6-benchmark eval suite instead).

Upstream already did (SynthLabsAI paper + open-r1 processing):
  - dedup, MATH-500/Omni-MATH decontamination, non-English/multi-part/
    proof/yes-no removal (SynthLabsAI)
  - drop rows whose answer math-verify cannot parse; drop null solve rates
    (open-r1)

This script adds only what upstream does not know about our setup:
  F1  source dedup: drop source in {gsm8k, math} -- both datasets are
      already in our train list as data/gsm8k + data/math; keeping them
      here would double-sample those problems.
  F2  solve-rate band [0.1, 0.7]: outside the band GDPO group advantage
      collapses (all-fail or all-pass groups carry no signal). Band is
      Llama-3.1-8B based -- recalibrate against critic/score/mean if the
      4B/8B policy disagrees.

Schema matches data/gsm8k byte-identically (pyarrow multi-parquet concat).
data_source is fixed to "open-r1/Big-Math-RL-Verified-Processed", which
routes to math-verify boxed-gated scoring in reward_score/__init__.py.

Usage (dt2):
  HF_ENDPOINT=https://hf-mirror.com python examples/data_preprocess/bigmath.py \
      --raw_dir /2022533109/zhouchuyan/verl/data/raw_bigmath_processed \
      --local_save_dir ./data/bigmath \
      --system_prompt_file math_reasoning.txt
"""

import argparse
import os
from pathlib import Path

import datasets
import pandas as pd

PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"

DATA_SOURCE = "open-r1/Big-Math-RL-Verified-Processed"
INSTRUCTION = r"Let's think step by step and output the final answer in \boxed{}."


def _load_prompt_file(path_or_name: str) -> str:
    p = Path(path_or_name)
    if not p.is_absolute():
        p = PROMPT_DIR / p
    return p.read_text(encoding="utf-8").strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="local snapshot of the processed dataset")
    parser.add_argument("--local_save_dir", default="./data/bigmath")
    parser.add_argument("--system_prompt_file", default=None)
    # Band [0, 0.9] (2026-07-05 decision). min=0: unlike outcome-only GRPO
    # (all-fail group -> zero advantage -> dead signal), Step-GDPO's process
    # reward ranks rollouts by verified steps even when every rollout misses
    # the final answer -- hard problems are exactly where dense step credit
    # pays off. Big-Math already removed problems unsolvable by Llama-405B
    # x8 and 8B x64, so sub-0.1 rows still have reproduced answer labels.
    # max=0.9: solve rates are Llama-3.1-8B based and our Qwen policies run
    # higher; all-correct groups only waste compute (advantage collapses to
    # 0). Watch bigmath o_rate vs outcome divergence for process-reward
    # hacking on unsolved problems. Shipped train.parquet = 161,037 rows.
    parser.add_argument("--solve_rate_min", type=float, default=0.0)
    parser.add_argument("--solve_rate_max", type=float, default=0.9)
    args = parser.parse_args()

    system_prompt = _load_prompt_file(args.system_prompt_file) if args.system_prompt_file else None

    src = os.path.join(args.raw_dir, "all", "train-00000-of-00001.parquet")
    df = pd.read_parquet(src)
    n0 = len(df)

    # F1: source dedup against our existing train parquets
    df = df[~df["source"].isin(["gsm8k", "math"])]
    n1 = len(df)

    # F2: solve-rate band
    df = df[(df["llama8b_solve_rate"] >= args.solve_rate_min) & (df["llama8b_solve_rate"] <= args.solve_rate_max)]
    n2 = len(df)
    print(f"rows: {n0} -> drop gsm8k/math sources -> {n1} -> solve_rate band -> {n2}")

    rows = []
    for idx, ex in enumerate(df.itertuples(index=False)):
        question = str(ex.prompt).strip()
        answer = str(ex.solution).strip()
        if not question or not answer:
            continue
        prompt_content = f"<Question>\n{question}\n</Question>\n\n{INSTRUCTION}"
        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_content})
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": prompt_messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "train",
                    "index": idx,
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
        )

    out = datasets.Dataset.from_list(rows)
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    out.to_parquet(os.path.join(save_dir, "train.parquet"))
    print(f"saved {len(out)} rows -> {save_dir}/train.parquet")
    print(f"data_source: {DATA_SOURCE}")
