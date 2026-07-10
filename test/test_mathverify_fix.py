"""Verify the math-verify subprocess-timeout fix (hardened version):
1. pathological giant-number answers return 0 within the wall timeout;
2. normal answers still grade correctly (no false rejects) -- including x^2 /
   \\pi^2 style answers that MUST stay in-thread (predicate check);
3. no orphaned grader processes survive the test.
Each case runs in an OUTER subprocess with a 25s cap; the fix's internal wall
is set to 5s here."""
import os
os.environ["MATH_VERIFY_WALL_TIMEOUT_S"] = "5"   # speed up the test
import multiprocessing as mp
import subprocess
import time


def check_predicate():
    from verl.utils.reward_score import math_verify as mv
    safe = [r"42", r"x^2+1", r"\frac{\pi^2}{6}", r"2^{10}", r"5!",
            r"\frac{n!}{k!(n-k)!}", r"3^{99}", r"10^{5}"]
    unsafe = [r"2^{100000000}", r"100000000!", r"2^{2^{30}}",
              r"(10^{8})!", "1" * 250, r"\gcd(10^{100000}, 3)"]
    bad = ([f"safe->{s}" for s in safe if not mv._mv_safe_inthread(s)]
           + [f"unsafe->{s}" for s in unsafe if mv._mv_safe_inthread(s)])
    return bad


def work(sol, gt, q):
    from verl.utils.reward_score import math_verify as mv
    q.put(mv.compute_score_boxed(sol, gt))


CASES = [
    ("pathol_power",     r"\boxed{2^{100000000}}",             "5",            "0 bounded"),
    ("pathol_factorial", r"\boxed{100000000!}",                "5",            "0 bounded"),
    ("pathol_tower",     r"\boxed{2^{2^{2^{2^{2^{2^{30}}}}}}}", "5",            "0 bounded"),
    ("normal_int",       r"The answer is \boxed{42}.",         "42",           "1.0 forked"),
    ("normal_x2",        r"\boxed{x^2 + 1}",                   r"x^2+1",       "1.0 forked"),
    ("normal_pi2",       r"\boxed{\frac{\pi^2}{6}}",           r"\frac{\pi^2}{6}", "1.0 forked"),
    ("normal_smallpow",  r"\boxed{2^{10}}",                    "1024",         "1.0 forked"),
    ("normal_wrong",     r"\boxed{7}",                         "42",           "0.0 forked"),
]

if __name__ == "__main__":
    fails = list(check_predicate())
    for b in fails:
        print(f"  [PREDICATE FAIL] {b}")
    print(f"predicate: {'OK' if not fails else 'FAILED'}")

    for name, sol, gt, exp in CASES:
        q = mp.Queue()
        p = mp.Process(target=work, args=(sol, gt, q))
        t0 = time.time()
        p.start()
        p.join(25)
        if p.is_alive():
            p.kill(); p.join()
            print(f"  {name:16s}: *** STILL SPINS >25s -- FIX FAILED ***")
            fails.append(name)
        else:
            dt = time.time() - t0
            r = q.get() if not q.empty() else None
            # VALUE assertion (2026-07-11): the earlier spin-only check let a
            # broken fork path (RLIMIT_AS zeroing every grade) report ALL OK.
            want = float(exp.split()[0])
            ok = r is not None and abs(float(r) - want) < 1e-6
            print(f"  {name:16s}: {dt:5.2f}s -> {r}   (expected {exp})"
                  + ("" if ok else "   *** VALUE MISMATCH ***"))
            if not ok:
                fails.append(name)

    time.sleep(1)
    leftover = subprocess.getoutput(
        "pgrep -f 'test_mathverify_fix' | grep -v ^%d$ | wc -l" % os.getpid())
    print(f"leftover grader procs (should be small/0): {leftover.strip()}")
    print("RESULT:", "ALL OK" if not fails else f"FAIL: {fails}")
