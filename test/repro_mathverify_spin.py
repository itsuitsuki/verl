"""Reproduce the reward-worker python/C spin by feeding pathological but
math-answer-shaped strings to math_verify.compute_score_boxed (the OUTCOME
reward that ONLY the math training uses; FOL never called it). Each candidate
runs in a subprocess with a 15s hard timeout -- a candidate that hits the
timeout IS the confirmed hang (math-verify has no in-thread timeout). This is
the definitive test of the 'what was the python spin' question."""
import multiprocessing as mp
import time


def work(s, gt, q):
    try:
        from verl.utils.reward_score import math_verify as mv
        q.put(mv.compute_score_boxed(s, gt))
    except Exception as e:
        q.put(f"ERR:{e!r}")


CANDS = {
    "big_power":       r"\boxed{2^{100000000}}",
    "big_factorial":   r"\boxed{100000000!}",
    "power_tower":     r"\boxed{2^{2^{2^{2^{2^{2^{30}}}}}}}",
    "nested_frac_300": r"\boxed{" + r"\frac{1}{" * 300 + "2" + "}" * 300 + "}",
    "nested_sqrt_500": r"\boxed{" + r"\sqrt{" * 500 + "2" + "}" * 500 + "}",
    "deep_leftright":  r"\boxed{" + r"\left(\frac{1}{" * 120 + "2" + r"}\right)" * 120 + "}",
    "long_digits_100k": r"\boxed{" + "1" * 100000 + "}",
    "gcd_bignum":      r"\boxed{\gcd(10^{100000}, 3^{100000})}",
}

if __name__ == "__main__":
    for name, s in CANDS.items():
        q = mp.Queue()
        p = mp.Process(target=work, args=(s, "5", q))
        t0 = time.time()
        p.start()
        p.join(15)
        if p.is_alive():
            p.terminate(); p.join()
            print(f"  {name:18s}: *** SPIN >15s -- CONFIRMED math-verify hang ***", flush=True)
        else:
            dt = time.time() - t0
            r = q.get() if not q.empty() else "(no result)"
            flag = "  <-- SLOW" if dt > 3 else ""
            print(f"  {name:18s}: ok {dt:5.2f}s -> {r}{flag}", flush=True)
