# New file for the verl-fol fork; follows examples/data_preprocess/math_lighteval.py.
"""
Preprocess the 6-benchmark math eval suite to the verl-fol parquet format.

Benchmarks (gsm8k test is NOT built here -- data/gsm8k/test.parquet already
exists and stays the canonical copy):
  - HuggingFaceH4/MATH-500           (500,  'test')
  - math-ai/aime24                   (30,   'test')
  - math-ai/amc23                    (40,   'test')
  - math-ai/minervamath              (272,  'test')
  - math-ai/olympiadbench            (674,  'test', text-only maths subset)

Each becomes data/<name>/test.parquet with the SAME extra_info struct as
data/gsm8k (identical field set -- pyarrow refuses to concat parquets
whose extra_info structs differ, and validation runs on a list of these
files together with gsm8k's).

Scoring routes (verl/utils/reward_score/__init__.py):
  MATH-500        -> is_equiv + acc_mathverify (dual, same as MATH-lighteval)
  aime24/amc23/minervamath/olympiadbench -> math-verify boxed-gated
  (aime* previously hit the math_dapo route; the explicit math-ai/* route
  added alongside this script takes precedence.)

Usage (dt2, needs HF_ENDPOINT=https://hf-mirror.com on datatech):
  python examples/data_preprocess/eval_suite.py \
      --local_save_root ./data \
      --system_prompt_file math_reasoning.txt
"""

import argparse
import os
from pathlib import Path

import datasets

PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"

INSTRUCTION = r"Let's think step by step and output the final answer in \boxed{}."

# name -> (hf_id, data_source, question_field, answer_field, save_dir)
BENCHES = {
    "math500": ("HuggingFaceH4/MATH-500", "HuggingFaceH4/MATH-500", "problem", "answer", "math500"),
    "aime24": ("math-ai/aime24", "math-ai/aime24", "problem", "solution", "aime24"),
    # AIME25 (2025-02) postdates most of Qwen3's pretraining crawl and its
    # report's own eval targets -- the least contamination-suspect bench in
    # the suite; weigh it accordingly when reading val curves.
    "aime25": ("math-ai/aime25", "math-ai/aime25", "problem", "answer", "aime25"),
    "amc23": ("math-ai/amc23", "math-ai/amc23", "question", "answer", "amc23"),
    "minerva": ("math-ai/minervamath", "math-ai/minervamath", "question", "answer", "minervamath"),
    "olympiadbench": ("math-ai/olympiadbench", "math-ai/olympiadbench", "question", "final_answer", "olympiadbench"),
}


def _load_prompt_file(path_or_name: str) -> str:
    p = Path(path_or_name)
    if not p.is_absolute():
        p = PROMPT_DIR / p
    return p.read_text(encoding="utf-8").strip()


def _norm_answer(ans) -> str:
    """Ground-truth normalization: olympiadbench wraps answers in a list,
    aime24's solution field wraps the integer in \\boxed{...}."""
    if isinstance(ans, (list, tuple)):
        if len(ans) != 1:
            return ""  # multi-answer rows are dropped (fail-closed grading)
        ans = ans[0]
    a = str(ans).strip()
    # strip one layer of $...$ so math-verify's \boxed{gt} wrap stays valid
    if a.startswith("$") and a.endswith("$") and len(a) > 2:
        a = a[1:-1].strip()
    # strip a full-string \boxed{...} wrapper (aime24): the scorer wraps
    # ground truth in \boxed{} itself, nesting would depend on parser grace
    if a.startswith("\\boxed{") and a.endswith("}"):
        inner = a[len("\\boxed{"):-1]
        if inner.count("{") == inner.count("}"):
            a = inner.strip()
    return a


def build_rows(name, dataset, data_source, q_field, a_field, system_prompt):
    rows = []
    skipped = 0
    for idx, ex in enumerate(dataset):
        question = str(ex[q_field]).strip()
        answer = _norm_answer(ex[a_field])
        if not question or not answer:
            skipped += 1
            continue
        # olympiadbench: keep only text-only rows (image fields all empty)
        if any(ex.get(f"image_{i}") is not None for i in range(1, 6) if f"image_{i}" in ex):
            skipped += 1
            continue
        prompt_content = f"<Question>\n{question}\n</Question>\n\n{INSTRUCTION}"
        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_content})
        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt_messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "answer": str(ex.get("solution", answer)),
                    "question": question,
                    "math_question": question,
                    "math_solution": str(ex.get("solution", "")),
                    "math_final_answer": answer,
                    "fol_context": "",
                    "fol_question": question,
                    "fol_options": "",
                },
            }
        )
    if skipped:
        print(f"  [{name}] skipped {skipped} rows (empty/multi answer or image modality)")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_root", default="./data")
    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--only", default=None, help="Comma-separated bench keys to rebuild.")
    args = parser.parse_args()

    system_prompt = _load_prompt_file(args.system_prompt_file) if args.system_prompt_file else None
    selected = args.only.split(",") if args.only else list(BENCHES)

    for key in selected:
        hf_id, data_source, qf, af, save_dir = BENCHES[key]
        print(f"Loading {hf_id} ...", flush=True)
        ds = datasets.load_dataset(hf_id, split="test")
        rows = build_rows(key, ds, data_source, qf, af, system_prompt)
        out = datasets.Dataset.from_list(rows)
        out_dir = os.path.expanduser(os.path.join(args.local_save_root, save_dir))
        os.makedirs(out_dir, exist_ok=True)
        out.to_parquet(os.path.join(out_dir, "test.parquet"))
        print(f"  {key}: {len(out)} rows -> {out_dir}/test.parquet")
