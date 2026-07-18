"""Linux process-tree and resource handling for Isabelle workers."""
import json
import logging
import os
import re
import shutil
import signal
from pathlib import Path

_POOL_LOG = logging.getLogger("verl.isabelle.server_pool")


def _die_with_parent():
    """Ask Linux to kill the direct child when its parent process dies.

    This does not reach the separate session created for Poly/ML, which is handled through recorded process groups."""
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass

def _read_stat(pid):
    """Read `(ppid, pgrp, session, starttime)` from `/proc/<pid>/stat`.

    Splitting after the final closing parenthesis handles process names containing spaces or parentheses."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    r = data.rfind(b")")
    if r < 0:
        return None
    f = data[r + 2:].split()
    if len(f) < 20:
        return None
    try:
        return int(f[1]), int(f[2]), int(f[3]), int(f[19])
    except ValueError:
        return None


def _descendants(root_pid):
    """Return every current descendant of `root_pid` with its process-group metadata."""
    children, meta = {}, {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return {}
    for name in names:
        if not name.isdigit():
            continue
        st = _read_stat(int(name))
        if st is None:
            continue
        ppid, pgrp, sess, start = st
        children.setdefault(ppid, []).append(int(name))
        meta[int(name)] = (pgrp, sess, start)
    out, seen, stack = {}, set(), list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out[pid] = meta.get(pid)
        stack.extend(children.get(pid, []))
    return out


def _prover_groups(jvm_pid, own_pgrp):
    """Return self-led process groups owned by the worker JVM.

    The training process group is excluded so cleanup cannot signal the caller."""
    out = {}
    for pid, meta in _descendants(jvm_pid).items():
        if not meta:
            continue
        pgrp, sess, start = meta
        if pid == pgrp == sess and pgrp != own_pgrp:
            out[pgrp] = start
    return out


_PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024


def _poly_tree_rss_kb(jvm_pid):
    """Return the total resident memory in KB of Poly/ML descendants.

    Isabelle's heap option does not bound the combined memory of abandoned and replacement Poly/ML processes, so the pool monitors the full process tree."""
    total = 0
    for pid in _descendants(jvm_pid):
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                d = fh.read()
        except OSError:
            continue
        r = d.rfind(b")")
        if r < 0 or d[d.find(b"(") + 1:r] != b"poly":
            continue
        f = d[r + 2:].split()
        if len(f) < 22:
            continue
        try:
            total += int(f[21]) * _PAGE_KB   # /proc stat field 24 = rss (pages)
        except ValueError:
            continue
    return total


def _poly_cpu_state(pid):
    """Return the state, accumulated CPU time, and start time of a Poly/ML process.

    The start time distinguishes a continuing process from PID reuse between monitor samples."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            d = fh.read()
    except OSError:
        return None
    r = d.rfind(b")")
    if r < 0 or d[d.find(b"(") + 1:r] != b"poly":
        return None
    f = d[r + 2:].split()
    if len(f) < 20:
        return None
    try:
        return f[0].decode(), int(f[11]) + int(f[12]), int(f[19])
    except (ValueError, IndexError):
        return None

def _find_jvm_pid(server_name):
    """Locate the daemonized Isabelle server JVM by an exact server-name argument.

    The launcher process exits after spawning the JVM, so the launcher's PID cannot be used for process-tree accounting."""
    want = server_name.encode()
    best_pid, best_start = None, -1
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/cmdline", "rb") as fh:
                cl = fh.read()
        except OSError:
            continue
        argv = cl.split(b"\0")
        if want not in argv:
            continue
        if not argv or not (argv[0].endswith(b"/java") or argv[0] == b"java"):
            continue
        st = _read_stat(int(name))
        start = st[3] if st else 0
        if start > best_start:
            best_pid, best_start = int(name), start
    return best_pid


def _kill_pgid(pgid, start, sig=signal.SIGKILL):
    """Signal a process group only while its leader has the recorded start time.

    The start-time check prevents a reused PID from identifying an unrelated process group."""
    st = _read_stat(pgid)
    if st is None or st[3] != start:
        return False  # leader gone or pid recycled
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _sweep_stale_pools(base_dir):
    """Clean process groups recorded by dead sibling pool owners.

    Only directories following this pool's PID-qualified naming scheme are considered. A registry is retained while any recorded process still survives, allowing a later cleanup attempt to finish the work."""
    base_dir = Path(base_dir)
    m = re.match(r"^(.*)_(\d+)$", base_dir.name)
    if not m:
        return
    stem, own = m.group(1), m.group(2)
    for d in base_dir.parent.glob(f"{stem}_*"):
        mm = re.match(r"^" + re.escape(stem) + r"_(\d+)$", d.name)
        if not mm or mm.group(1) == own:
            continue
        if _read_stat(int(mm.group(1))) is not None:
            continue  # owner pid still alive -> a concurrent sibling, leave it
        recs = []
        for reap in d.glob("worker_*.reap"):
            try:
                recs.append(json.loads(reap.read_text()))
            except (OSError, ValueError):
                continue
        for rec in recs:
            for g, s in rec.get("groups", {}).items():
                try:
                    _kill_pgid(int(g), int(s))
                except (ValueError, TypeError):
                    pass
            try:
                jp, js = int(rec.get("jvm_pid", 0)), int(rec.get("jvm_start", 0))
            except (ValueError, TypeError):
                jp, js = 0, 0
            if jp:
                st = _read_stat(jp)
                if st is not None and st[3] == js:
                    try:
                        os.kill(jp, signal.SIGKILL)
                    except OSError:
                        pass
        residual = False
        for rec in recs:
            for g, s in rec.get("groups", {}).items():
                try:
                    st = _read_stat(int(g))
                    if st is not None and st[3] == int(s):
                        residual = True
                except (ValueError, TypeError):
                    pass
            try:
                jp, js = int(rec.get("jvm_pid", 0)), int(rec.get("jvm_start", 0))
                st = _read_stat(jp) if jp else None
                if st is not None and st[3] == js:
                    residual = True
            except (ValueError, TypeError):
                pass
        if residual:
            _POOL_LOG.warning("stale-pool sweep: survivors under %s -- "
                              "keeping its .reap registry", d)
            continue
        try:
            shutil.rmtree(d)
        except OSError:
            pass
