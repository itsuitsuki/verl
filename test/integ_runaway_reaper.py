"""Integration test of the runaway-poly reaper against a REAL spinning process
named `poly` (no Isabelle needed). Validates the actual /proc reading
(_descendants + _poly_cpu_state) AND the real SIGKILL, end to end:

  parent bash --> child `poly` (a copy of `yes`, spins 100% CPU)

We point a pool worker's jvm_pid at the parent, drive the reaper a few ticks,
and confirm the reaper detects the sustained-100%-CPU poly and kills it, while
NOT killing anything if the child is idle. Bounded (~20s). Run on an IDLE node.

Usage: python integ_runaway_reaper.py /tmp/server_pool_new.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import time

_spec = importlib.util.spec_from_file_location("sp", sys.argv[1])
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

# speed the reaper up for the test: 3s tick, 6s sustained -> streak 2
sp.IsabelleServerPool.MONITOR_INTERVAL_S = 3.0
sp.IsabelleServerPool.RUNAWAY_CPU_S = 6.0

def poly_state(pid):
    return sp._poly_cpu_state(pid)


def run(spin: bool, label: str):
    # A process whose executable basename is `poly` has comm=="poly". spin=True
    # uses `yes` (100% CPU); spin=False uses `sleep` (idle, state S).
    d = f"/tmp/polytest_{label}"
    os.makedirs(d, exist_ok=True)
    poly_bin = f"{d}/poly"
    src = (shutil.which("yes") or "/usr/bin/yes") if spin else \
          (shutil.which("sleep") or "/usr/bin/sleep")
    shutil.copy(src, poly_bin)
    inner = f"{poly_bin} >/dev/null" if spin else f"{poly_bin} 600"
    parent = subprocess.Popen(["bash", "-c", inner], preexec_fn=os.setsid)
    time.sleep(1.0)
    desc = sp._descendants(parent.pid)
    polys = [pid for pid in desc if poly_state(pid) is not None]
    print(f"\n[{label}] parent={parent.pid} descendants={list(desc)} "
          f"poly-named-children={polys}")
    if spin and not polys:
        print("  [FAIL] no poly child found under parent via _descendants")
        parent.kill(); return False

    pool = sp.IsabelleServerPool(num_workers=1, base_dir=f"/tmp/integ_{label}")
    pool.workers[0].jvm_pid = parent.pid
    print(f"  runaway_streak={pool._runaway_streak} interval={pool.MONITOR_INTERVAL_S}s "
          f"clk_tck={pool._clk_tck}")

    killed_pid = None
    for i in range(7):
        pool._reap_runaway_polys_once()
        target = polys[0] if polys else None
        alive = poly_state(target) is not None if target else False
        streak = pool._poly_cpu.get(target, ("", "", 0))[2] if target else 0
        print(f"  tick{i}: poly {target} alive={alive} streak={streak} "
              f"reasons={pool.restart_reasons}")
        if target and not alive:
            killed_pid = target
            break
        time.sleep(3)

    # cleanup
    try:
        os.killpg(os.getpgid(parent.pid), 9)
    except OSError:
        pass
    try:
        pool.shutdown()
    except Exception:
        pass

    if spin:
        ok = killed_pid is not None and pool.restart_reasons.get("runaway_poly", 0) > 0
        print(f"  [{'PASS' if ok else 'FAIL'}] spinning poly {'killed' if ok else 'NOT killed'}")
        return ok
    else:
        ok = pool.restart_reasons.get("runaway_poly", 0) == 0
        print(f"  [{'PASS' if ok else 'FAIL'}] idle poly {'left alone' if ok else 'WRONGLY killed'}")
        return ok


r1 = run(spin=True, label="spin")
r2 = run(spin=False, label="idle")
shutil.rmtree("/tmp/polytest_spin", ignore_errors=True)
shutil.rmtree("/tmp/polytest_idle", ignore_errors=True)
print("\nRESULT:", "ALL PASS" if (r1 and r2) else "FAIL")
sys.exit(0 if (r1 and r2) else 1)
