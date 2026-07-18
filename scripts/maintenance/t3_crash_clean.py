"""Definitive T3: SIGKILL the process that owns an Isabelle pool; the detached
reaper must kill that process's poly grandchildren.

Counts poly by /proc/<pid>/comm == 'poly' (the correct method; the old shell
script used `pgrep -cf polyml` which matches nothing and mis-parses on 0).
Tracks the *specific* poly PIDs the child created, then proves those exact PIDs
die after the child is SIGKILLed -> not a count coincidence.
"""
import os
import signal
import subprocess
import sys
import time


def poly_pids():
    out = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/comm") as fh:
                if fh.read().strip() == "poly":
                    out.add(int(name))
        except OSError:
            pass
    return out


BASE_DIR = f"/tmp/t3_clean_{os.getpid()}"
CHILD = "/tmp/test_reaper_crash_child.py"

base = poly_pids()
print(f"baseline poly pids: {sorted(base)}")

proc = subprocess.Popen(
    [sys.executable, CHILD, BASE_DIR],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
print(f"child pid={proc.pid}, waiting for CHILD_READY ...")

ready = False
deadline = 180
t0 = time.monotonic()
while time.monotonic() - t0 < deadline:
    line = proc.stdout.readline()
    if not line:
        break
    line = line.rstrip()
    print(f"  [child] {line}")
    if "CHILD_READY" in line:
        ready = True
        break
if not ready:
    print("FAIL: child never became ready")
    proc.kill()
    sys.exit(1)

time.sleep(1)
childs = poly_pids() - base
print(f"\npoly created by child ({len(childs)}): {sorted(childs)}")
if not childs:
    print("FAIL: child created no poly (nothing to reap) -> test invalid")
    proc.kill()
    sys.exit(1)

print(f"\n=== SIGKILL child {proc.pid} (simulates OOM kill of the pool owner) ===")
os.kill(proc.pid, signal.SIGKILL)

# reaper waits on parent death (pidfd) then killpg's recorded groups.
survivors = childs
for i in range(30):
    time.sleep(1)
    survivors = childs & poly_pids()
    if not survivors:
        print(f"  +{i+1}s: all child poly reaped")
        break
    if i % 3 == 0:
        print(f"  +{i+1}s: child poly still alive: {sorted(survivors)}")

print("\n=== T3 RESULT ===")
if survivors:
    print(f"FAIL: reaper leaked {len(survivors)} poly: {sorted(survivors)}")
    for p in survivors:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    sys.exit(1)
else:
    print("PASS: detached reaper killed all child poly after SIGKILL")

import shutil
shutil.rmtree(BASE_DIR, ignore_errors=True)
sys.exit(0)
