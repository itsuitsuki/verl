#!/usr/bin/env python3
"""Clean Isabelle processes after the owning Python process dies.

The reaper runs in a separate session, waits for the owner identified by PID and start time, then reads `.reap` records and terminates their prover process groups and JVMs."""
import json
import os
import signal
import sys
import time


def _stat(pid):
    """Read parent, process-group, session, start-time, and state fields from `/proc/<pid>/stat`."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            d = fh.read()
    except OSError:
        return None
    r = d.rfind(b")")
    f = d[r + 2:].split() if r >= 0 else []
    if len(f) < 20:
        return None
    try:
        return int(f[1]), int(f[2]), int(f[3]), int(f[19]), f[0].decode()
    except (ValueError, UnicodeDecodeError):
        return None


def _wait_death(pid, start):
    """Wait until the recorded owner exits or becomes a zombie.

    A zombie has already released its children, so cleanup need not wait for its `/proc` entry to disappear."""
    while True:
        st = _stat(pid)
        if st is None or st[3] != start or st[4] == "Z":
            return
        time.sleep(1.0)


def main():
    if len(sys.argv) < 4:
        return
    base, ppid, pstart = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    _wait_death(ppid, pstart)
    time.sleep(0.5)  # let a clean shutdown() win the race before we reap
    try:
        names = os.listdir(base)
    except OSError:
        return
    for name in names:
        if not (name.startswith("worker_") and name.endswith(".reap")):
            continue
        try:
            with open(os.path.join(base, name)) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        for g, s in rec.get("groups", {}).items():
            try:
                pgid, start = int(g), int(s)
            except (ValueError, TypeError):
                continue
            st = _stat(pgid)
            if st is not None and st[3] == start:  # leader alive + not pid-reused
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
        try:
            jp, js = int(rec.get("jvm_pid", 0)), int(rec.get("jvm_start", 0))
        except (ValueError, TypeError):
            jp, js = 0, 0
        if jp:
            st = _stat(jp)
            if st is not None and st[3] == js and st[4] != "Z":
                try:
                    os.kill(jp, signal.SIGKILL)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
