#!/usr/bin/env python3
"""Detached crash-time reaper for the Isabelle server pool.

Spawned by IsabelleServerPool in its OWN session (start_new_session=True) so it
survives the training process being OOM-SIGKILLed. It blocks on the parent's
pidfd (or polls /proc as a fallback), and when the parent dies it SIGKILLs every
prover process group recorded under <base_dir>/worker_*.reap.

Why this exists: the PolyML `poly` heap process is a grandchild that Isabelle's
`bash_process` wrapper setsid's into its OWN session (pgid==sid==pid). PDEATHSIG
on the isabelle-server JVM cannot reach it, and atexit/stop() never run under
SIGKILL. On an OOM kill of the training process, without this reaper every poly
(~4-5GB) would orphan to init and keep pressuring the shared container's memory
cgroup until the next run's startup sweep. See server_pool.py for the layered
design (this is Layer 4).

Needs no systemd/cgroup (unavailable in the Singularity container) -- it keys
purely off the on-disk .reap registry and starttime-guarded killpg.

argv: reaper.py <base_dir> <parent_pid> <parent_starttime>
"""
import json
import os
import signal
import sys
import time


def _stat(pid):
    """(ppid, pgrp, session, starttime, state) from /proc/<pid>/stat, or None.
    comm may contain spaces/parens, so split AFTER the last ')'."""
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
        # after ') ': f[0]=state f[1]=ppid f[2]=pgrp f[3]=session f[19]=starttime
        return int(f[1]), int(f[2]), int(f[3]), int(f[19]), f[0].decode()
    except (ValueError, UnicodeDecodeError):
        return None


def _wait_death(pid, start):
    """Block until pid (with matching starttime) dies.

    A SIGKILLed process whose real parent never waitpid()s it lingers as an
    unreaped ZOMBIE: /proc/<pid>/stat and its starttime persist indefinitely,
    so waiting for the /proc entry to vanish can hang forever (observed on the
    Singularity container's kernel; pidfd POLLIN likewise does NOT fire on an
    unreaped zombie here). But a zombie is already terminated -- its children,
    the poly heaps, were orphaned the instant it died -- so we treat state 'Z'
    as death and reap immediately. A 1s /proc poll is ample for a crash reaper.
    """
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


if __name__ == "__main__":
    main()
