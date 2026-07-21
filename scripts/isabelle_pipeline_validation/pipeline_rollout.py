"""Task #37 rollout generation: sample train problems per dataset and generate step-format
responses with the policy model exactly as training does (temperature 0.8, top_p 0.95,
max response 1536 tokens, enable_thinking=false; the parquet rows already carry the
math_reasoning system prompt inside their chat prompt).

Writes one JSONL row per (problem, sample): dataset, idx, sample, problem (the user message
content, which is exactly what the step reward manager passes to the engine as `problem`),
ground_truth (extra_info math_final_answer, the field the reward manager reads), response.

  python -u pipeline_rollout.py --out rollouts.jsonl   (needs the policy vLLM at --url)
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests


def rollout_one(url, model, messages, n, max_tokens):
    r = requests.post(url + "/chat/completions", json={
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "n": n,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=900)
    r.raise_for_status()
    return [c["message"]["content"] for c in r.json()["choices"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/2022533109/zhouchuyan/verl/data")
    ap.add_argument("--datasets", default="gsm8k,math,bigmath,deepmath,dapo_math")
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--url", default="http://127.0.0.1:4875/v1")
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filter-regex", default=None,
                    help="keep only problems whose user message matches (case-insensitive)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = []
    for ds in args.datasets.split(","):
        df = pd.read_parquet(Path(args.data_root) / ds / "train.parquet")
        if args.filter_regex:
            keep = df["prompt"].map(lambda p: bool(
                __import__("re").search(args.filter_regex, p[-1]["content"],
                                        __import__("re").IGNORECASE)))
            df = df[keep]
            print("%s: %d problems match filter" % (ds, len(df)), flush=True)
        sub = df.sample(n=min(args.n_problems, len(df)), random_state=args.seed)
        for idx, row in sub.iterrows():
            messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
            ei = row["extra_info"]
            gt = str(ei.get("math_final_answer") or ei.get("answer")
                     or row["reward_model"]["ground_truth"])
            tasks.append((ds, int(idx), messages, gt))
    print("total prompts: %d" % len(tasks), flush=True)

    out = open(args.out, "w", encoding="utf-8")

    def work(t):
        ds, idx, messages, gt = t
        texts = rollout_one(args.url, args.model, messages, args.n_samples, args.max_tokens)
        return [{"dataset": ds, "idx": idx, "sample": s,
                 "problem": messages[-1]["content"], "ground_truth": gt, "response": txt}
                for s, txt in enumerate(texts)]

    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for rows in ex.map(work, tasks):
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            if done % 25 == 0:
                print("prompts done: %d/%d" % (done, len(tasks)), flush=True)
    out.close()
    print("DONE rollouts -> %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
