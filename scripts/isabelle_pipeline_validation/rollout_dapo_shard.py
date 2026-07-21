"""Generate one stable shard of DAPO policy rollouts with one local GPU.

Rows are assigned by their positional index modulo the total shard count. The script
uses the parquet chat prompt verbatim and the training sampling settings, so shards
from multiple GPUs or nodes can be concatenated without overlap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/2022533109/zhouchuyan/models/Qwen3-4B")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")

    frame = pd.read_parquet(args.parquet)
    selected = [(position, row) for position, (_, row) in enumerate(frame.iterrows())
                if position % args.num_shards == args.shard_id]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [tokenizer.apply_chat_template(
        [{"role": message["role"], "content": message["content"]}
         for message in row["prompt"]],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) for _, row in selected]

    model = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(
        n=args.samples,
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )
    outputs = model.generate(prompts, sampling)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for (position, row), generated in zip(selected, outputs):
            extra = row["extra_info"]
            reward_model = row["reward_model"]
            ground_truth = str(
                extra.get("math_final_answer")
                or extra.get("answer")
                or reward_model["ground_truth"]
            )
            problem = str(extra.get("math_question") or extra.get("question") or "")
            for sample, candidate in enumerate(generated.outputs):
                handle.write(json.dumps({
                    "dataset": "dapo_math",
                    "idx": position,
                    "sample": sample,
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "response": candidate.text,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                }, ensure_ascii=False) + "\n")
                written += 1
    print(json.dumps({
        "parquet_rows": len(frame),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "problems": len(selected),
        "samples": args.samples,
        "responses": written,
        "out": str(output),
    }))


if __name__ == "__main__":
    main()
