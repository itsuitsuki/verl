#!/usr/bin/env python3
"""Benchmark local OpenAI-compatible judge latency.

This is intentionally small and dependency-free so it can run while training is
alive. It reports endpoint latency plus usage tokens returned by vLLM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def _post_chat(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        latency_s = time.perf_counter() - t0
        obj = json.loads(raw)
        return {
            "ok": True,
            "latency_s": latency_s,
            "usage": obj.get("usage", {}),
            "finish_reason": (obj.get("choices") or [{}])[0].get("finish_reason"),
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "latency_s": time.perf_counter() - t0, "error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - t0, "error": repr(exc)}


def _build_prompt(size: str) -> str:
    base = 'Return JSON only: {"ok": true}. Do not explain.\nContext:\n'
    chunks = {
        "small": 8,
        "medium": 160,
        "large": 520,
    }
    repeat = chunks[size]
    text = "All cats are animals. Fluffy is a cat. Therefore Fluffy is an animal. Some animals are pets.\n"
    return base + text * repeat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", default=[], help="Base URL like http://127.0.0.1:4874")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--sizes", default="small,medium", help="Comma-separated: small,medium,large")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    endpoints = args.endpoint or ["http://127.0.0.1:4872", "http://127.0.0.1:4873", "http://127.0.0.1:4874"]
    sizes = [item.strip() for item in args.sizes.split(",") if item.strip()]

    for endpoint in endpoints:
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        for size in sizes:
            prompt = _build_prompt(size)
            rows = []
            for _ in range(args.repeat):
                row = _post_chat(url, args.model, prompt, args.max_tokens, args.timeout)
                row.update({"endpoint": endpoint, "size": size})
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            ok_latencies = [row["latency_s"] for row in rows if row.get("ok")]
            if ok_latencies:
                print(
                    json.dumps(
                        {
                            "endpoint": endpoint,
                            "size": size,
                            "summary": True,
                            "n_ok": len(ok_latencies),
                            "latency_mean_s": statistics.mean(ok_latencies),
                            "latency_max_s": max(ok_latencies),
                            "latency_min_s": min(ok_latencies),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
