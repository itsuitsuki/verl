"""Standalone (no-pytest) logic test for the runaway-poly reaper in
server_pool.py. Mocks /proc access (_descendants / _poly_cpu_state / _read_stat)
and os.kill so NO Isabelle is needed. Validates: a poly sustaining ~100% CPU for
>= _runaway_streak ticks is SIGKILLed; idle / intermittent polys are NOT; a
starttime change (pid reuse) resets the streak. TEMP probe -> delete with ./test."""
import importlib.util
import os
import sys
import tempfile

# Import the module under test either from an explicit file path (arg 1, so a
# /tmp copy can be checked without touching the shared package) or the package.
if len(sys.argv) > 1:
    _spec = importlib.util.spec_from_file_location("server_pool_uut", sys.argv[1])
    sp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sp)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from verl.utils.isabelle_utils import server_pool as sp

fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp()
pool = sp.IsabelleServerPool(num_workers=1, base_dir=tmp)
pool.workers[0].jvm_pid = 9999
HZ = pool._clk_tck
INT = pool.MONITOR_INTERVAL_S
STREAK = pool._runaway_streak
print(f"clk_tck={HZ} interval={INT}s runaway_streak={STREAK} "
      f"(=> ~{STREAK*INT:.0f}s sustained 100% CPU to kill)")

# module-level mocks
sp._read_stat = lambda pid: (1, 1, 1, 100) if pid == 9999 else None
sp._descendants = lambda root: {12345: (0, 0, 0)}
_orig_kill = os.kill
killed = []
os.kill = lambda pid, sig: killed.append(pid)

cpu = [0]
start = [500]
sp._poly_cpu_state = lambda pid: ("R", cpu[0], start[0])

try:
    # 1) sustained 100% spinner -> killed after streak
    cpu[0] = 0; start[0] = 500; pool._poly_cpu = {}; killed.clear()
    pool._reap_runaway_polys_once()                       # baseline tick
    for _ in range(STREAK):
        cpu[0] += int(INT * HZ)                           # 100% of the interval
        pool._reap_runaway_polys_once()
    check("sustained 100% spinner is SIGKILLed", 12345 in killed)

    # 2) idle poly (cpu never advances) -> never killed
    cpu[0] = 0; start[0] = 500; pool._poly_cpu = {}; killed.clear()
    for _ in range(STREAK + 3):
        pool._reap_runaway_polys_once()
    check("idle poly never killed", 12345 not in killed)

    # 3) 50% busy (below RUNAWAY_BUSY_FRAC) -> never killed
    cpu[0] = 0; start[0] = 500; pool._poly_cpu = {}; killed.clear()
    for _ in range(STREAK + 3):
        cpu[0] += int(0.5 * INT * HZ)
        pool._reap_runaway_polys_once()
    check("50%-busy poly never killed (below frac)", 12345 not in killed)

    # 4) starttime change (pid reuse) mid-run resets the streak
    cpu[0] = 0; start[0] = 500; pool._poly_cpu = {}; killed.clear()
    pool._reap_runaway_polys_once()                       # baseline @ start 500
    cpu[0] += int(INT * HZ); start[0] = 600                # SAME pid, NEW process
    pool._reap_runaway_polys_once()                        # streak resets to 0
    cpu[0] += int(INT * HZ)
    pool._reap_runaway_polys_once()                        # streak now 1, < STREAK
    check("pid-reuse (starttime change) resets streak, no false kill",
          12345 not in killed)

    # 5) after a real kill, a fresh long-busy poly still gets caught (state clean)
    cpu[0] = 0; start[0] = 700; pool._poly_cpu = {}; killed.clear()
    pool._reap_runaway_polys_once()
    for _ in range(STREAK):
        cpu[0] += int(INT * HZ)
        pool._reap_runaway_polys_once()
    check("second distinct spinner also killed", 12345 in killed)
finally:
    os.kill = _orig_kill

print("\nRESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
