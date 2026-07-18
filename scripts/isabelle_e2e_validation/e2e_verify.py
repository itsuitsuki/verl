"""Task #37 verification: run every rollout through the ACTUAL reward pipeline and save the
full per-response verification records.

The call is process_one_response — the same function training's engine.verify_solution wraps —
with outdir set, so each fully-translated response also leaves the translation debug JSON
(nl steps + Isabelle step conclusions) that the zero-false-positive audit reads.
api_config mirrors the production training launch (verify_timeout=60, api_timeout=200,
rss cap 12 GB, both dt1 judges load-balanced) except pool_workers, a per-process sizing knob
(training: 3 workers x 4 reward processes; here: one process).

  bash with_env.sh python -u e2e_verify.py --rollouts rollouts.jsonl \
      --out records.jsonl --debug-dir debug
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-dir", required=True)
    ap.add_argument("--judges", default="http://127.0.0.1:4873/v1,http://127.0.0.1:4874/v1")
    ap.add_argument("--pool-workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.rollouts, encoding="utf-8")]
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    api_config = {
        "base_url": args.judges,
        "model": "Qwen3.6-35B-A3B",
        "timeout": 60,
        "api_timeout": 200,
        "isabelle_pool_workers": args.pool_workers,
        "isabelle_worker_rss_cap_gb": 12,
    }
    from verl.utils.reward_score.formal_verify import _get_isabelle_engine
    engine = _get_isabelle_engine(api_config)
    from verl.utils.isabelle_utils.engine import process_one_response

    out = open(args.out, "w", encoding="utf-8")
    wlock = threading.Lock()
    t0 = time.time()
    ndone = [0]

    def work(i):
        row = rows[i]
        item = {"problem": row["problem"], "response": row["response"],
                "ground_truth": row["ground_truth"], "dataset": row["dataset"],
                "idx": row["idx"], "sample": row["sample"]}
        t = time.time()
        try:
            rec = process_one_response(i, item, engine.pool, engine.config,
                                       outdir=debug_dir, max_steps=0)
        except Exception as e:  # noqa: BLE001 -- one bad response must not kill the sweep
            rec = {"error": repr(e), "format_ok": None}
        rec["response_id"] = i
        rec["dataset"] = row["dataset"]
        rec["idx"] = row["idx"]
        rec["sample"] = row["sample"]
        rec["wall"] = round(time.time() - t, 2)
        with wlock:
            out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            out.flush()
            ndone[0] += 1
            n = ndone[0]
        print("[%d/%d] id=%d ds=%s pattern=%r n_steps=%s wall=%.1fs elapsed=%.0fm"
              % (n, len(rows), i, row["dataset"], rec.get("pattern"),
                 rec.get("n_steps"), rec["wall"], (time.time() - t0) / 60), flush=True)

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        list(ex.map(work, range(len(rows))))
    out.close()
    print("DONE verify -> %s (%.0f min, %d responses)"
          % (args.out, (time.time() - t0) / 60, len(rows)), flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
