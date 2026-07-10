#!/usr/bin/env python3
"""Long-running Isabelle server pool (PIDE protocol over TCP).

Replaces per-call `isabelle build` subprocess (~10s, of which ~8s is process
cold start + heap loading) with resident `isabelle server` workers that load
the HOL-Number_Theory session once (~8s each) and then check theories at
~0.1-2s per call.

Protocol facts (probed 2026-06-11 on Isabelle 2025, see probe_server.py):
- `isabelle server -n NAME` prints one banner line, then stays in the
  foreground as the server process:
      server "NAME" = 127.0.0.1:PORT (password "UUID")
- the client connects via TCP and sends the password as its first line
- commands: `name {json}`; immediate reply `OK {"task": id}`; async NOTE
  progress messages; terminal FINISHED/FAILED carrying the same task id
- replies may be length-prefixed: a line of pure digits N means "the next N
  bytes are one message"
- session_start {"session": S, "options": ["quick_and_dirty=true"]}
  -> FINISHED {"session_id": ...}   (quick_and_dirty confirmed: sorry passes)
- use_theories {"session_id", "theories": [name], "master_dir"}
  -> FINISHED {"ok": bool, "errors": [...], "nodes": [...]}
- a re-used theory name may be served from the session's loaded-theories
  cache (probe: purge_theories reported the theory as retained, and the
  follow-up call returned in 0.11s). Every check therefore uses a UNIQUE
  theory name; purge_theories {"all": true} runs periodically to bound
  session memory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from queue import Queue

# Worker-restart events (usually verify_timeout hits) go here at DEBUG instead
# of stdout, to keep training logs readable. Bump to DEBUG to see each restart.
_POOL_LOG = logging.getLogger("verl.isabelle.server_pool")

ISABELLE_HOME = os.environ.get(
    "ISABELLE_HOME", "/2022533109/zhouchuyan/isabelle/Isabelle2025")
ISABELLE_BIN = f"{ISABELLE_HOME}/bin/isabelle"


def _die_with_parent():
    """preexec_fn: make this subprocess receive SIGKILL when its parent
    Python process dies (any reason: SIGKILL, OOM, segfault, normal exit).

    Uses Linux prctl(PR_SET_PDEATHSIG, SIGKILL). No-op on non-Linux.
    """
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Orphan-prover reaping (2026-07-08). The PolyML `poly` heap process is a
# grandchild that Isabelle's `bash_process` wrapper setsid's into a NEW
# session: measured tree is
#     JVM (our pgrp) -> bash_process (our pgrp) -> bash (pid==pgrp==sess) -> poly
# The `bash` under bash_process is the session LEADER (pid==pgrp==sess) and
# poly shares its group. So killpg(that_leader_pgid) reaps bash AND poly in one
# shot, and it crosses the setsid boundary by construction -- neither PDEATHSIG
# on the JVM nor killpg on OUR group can reach it. We record each such leader
# pgid (+ starttime, to defeat pid reuse) at spawn and killpg them at
# stop/restart/shutdown, plus a startup sweep and a detached crash reaper.
# ---------------------------------------------------------------------------

def _read_stat(pid):
    """(ppid, pgrp, session, starttime) from /proc/<pid>/stat, or None.

    comm may contain spaces/parens, so split AFTER the last ')': in the tail
    the 0-indexed fields are state=0, ppid=1, pgrp=2, session=3, starttime=19
    (i.e. /proc stat fields 4, 5, 6, 22)."""
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
    """{pid: (pgrp, session, starttime)} for every ppid-descendant of root_pid
    right now (single /proc scan + BFS over the ppid map)."""
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
    """Self-led session leaders (pid==pgrp==sess) under the JVM, excluding our
    own training group -> the `bash` leaders each owning a poly. Returns
    {pgid: starttime}."""
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
    """Total RSS (KB) of every `poly` prover process descended from jvm_pid.
    One /proc scan (via _descendants) + a stat read per descendant. This is
    what actually bounds a worker's host-memory footprint: --maxheap does NOT
    cap poly RSS (measured >99% private-anon PolyML working memory), and
    aborted 'zombie proof' polys accumulate inside a still-live JVM until OOM.
    """
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
    """(state, cpu_jiffies, starttime) for a `poly` pid, else None. cpu_jiffies
    = utime+stime; sampled across monitor ticks to detect a poly spinning ~100%
    CPU (a native tactic that ignored Isabelle's cooperative verify timeout).
    starttime defeats pid reuse across ticks."""
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


# --- Theorem-verdict disk cache (2026-07-10). The in-memory theorem memo dies
# with the process, and this training restarts often (resume replays the same
# problems with the same fixed data seed) -- every restart re-proves thousands
# of already-settled theorems. Persist EXACTLY what the memory memo caches
# (success always; failure only when fast and not a consolidation flake), keyed
# on the full theorem text (which embeds the tactic string, so tactic changes
# change the key). Bump ISABELLE_THEOREM_CACHE_VERSION if SESSION/THEORY_IMPORTS
# change (those alter verdicts without changing theorem text). ---
_THM_CACHE_VERSION = os.environ.get("ISABELLE_THEOREM_CACHE_VERSION", "v1")


def _thm_disk_enabled():
    return os.environ.get("ISABELLE_THEOREM_DISK_CACHE", "1") not in (
        "0", "false", "False")


def _thm_disk_path(theorem_code):
    key = hashlib.sha1(
        (_THM_CACHE_VERSION + "\0" + theorem_code).encode("utf-8")).hexdigest()
    base = os.environ.get("ISABELLE_THEOREM_CACHE_DIR",
                          "/tmp/verl_isabelle_theorem_cache")
    return os.path.join(base, _THM_CACHE_VERSION, key[:2], f"{key}.json")


def _thm_disk_load(theorem_code):
    if not _thm_disk_enabled():
        return None
    try:
        with open(_thm_disk_path(theorem_code)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _thm_disk_store(theorem_code, value):
    if not _thm_disk_enabled():
        return
    path = _thm_disk_path(theorem_code)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(value, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _find_jvm_pid(server_name):
    """Locate the REAL java server JVM by its `-n <server_name>` cmdline arg.

    The `isabelle server` launcher daemonizes: the Popen'd bash exits (leaving a
    <defunct> zombie) and the real java server reparents to init. So worker.proc
    is NOT the JVM, and every pid-tree walk rooted at worker.proc.pid finds ZERO
    poly -- which silently defeated killpg reaping AND the RSS watchdog (they saw
    an empty tree, never reaped, poly leaked to OOM). Prefer the most recently
    started match in case a stale same-named JVM lingers. Returns pid or None.
    """
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
        if want not in cl or b"/java" not in cl:
            continue
        st = _read_stat(int(name))
        start = st[3] if st else 0
        if start > best_start:
            best_pid, best_start = int(name), start
    return best_pid


def _kill_pgid(pgid, start, sig=signal.SIGKILL):
    """SIGKILL a process GROUP by leader pid, ONLY if the leader still carries
    the recorded starttime (defeats pid reuse). Safe on a shared root
    container: callers only ever pass self-led groups (pid==pgrp==sess) that
    descended from our own JVM -- never the training group, never a tenant."""
    st = _read_stat(pgid)
    if st is None or st[3] != start:
        return False  # leader gone or pid recycled
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _sweep_stale_pools(base_dir):
    """Reap prover groups left by a CRASHED previous run of THIS engine.
    Sibling dirs are matched by our own naming scheme (<stem>_<pid>) so we
    never touch another tool's or another tenant's Isabelle jobs."""
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
        for reap in d.glob("worker_*.reap"):
            try:
                rec = json.loads(reap.read_text())
            except (OSError, ValueError):
                continue
            for g, s in rec.get("groups", {}).items():
                try:
                    _kill_pgid(int(g), int(s))
                except (ValueError, TypeError):
                    pass
        try:
            shutil.rmtree(d)
        except OSError:
            pass

SESSION = "HOL-Number_Theory"
SESSION_OPTIONS = [
    "quick_and_dirty=true",
    # headless PIDE consolidation polls at 2.0s by default, which puts a
    # ~2.1s floor under every use_theories call regardless of proof size
    "headless_consolidate_delay=0.05",
    "headless_check_delay=0.05",
    # a theory with a broken header (outer syntax error in the theorem
    # statement) never consolidates; the watchdog is the server-side abort
    # for exactly that case (default 600s -> wedges the worker for the whole
    # CHECK_DEADLINE and forces a restart)
    "headless_watchdog_timeout=15",
]
THEORY_IMPORTS = """  imports
    Complex_Main
    "HOL-Library.Sum_of_Squares"
    "HOL-Library.Code_Target_Numeral"
    "HOL-Number_Theory.Number_Theory"
    "HOL-Decision_Procs.Approximation"
"""
PURGE_EVERY = 10          # checks per worker between purge_theories all=true
CHECK_DEADLINE = 60.0     # seconds per use_theories call before worker restart
BANNER_RE = re.compile(r'server "(.+?)" = ([\d.]+):(\d+) \(password "(.+?)"\)')


class _Conn:
    """Blocking message connection to one isabelle server."""

    def __init__(self, host: str, port: int, password: str):
        self.sock = socket.create_connection((host, port))
        self.buf = b""
        self.send_line(password)
        reply = self.read_msg()
        if not reply.startswith("OK"):
            raise RuntimeError(f"server auth failed: {reply!r}")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def send_line(self, s: str):
        self.sock.sendall(s.encode() + b"\n")

    def _read_line(self) -> bytes:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("server closed connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def _read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("server closed connection")
            self.buf += chunk
        data, self.buf = self.buf[:n], self.buf[n:]
        return data

    def read_msg(self) -> str:
        line = self._read_line()
        if re.fullmatch(rb"\d+", line.strip()):
            return self._read_exact(int(line)).decode()
        return line.decode()

    def command(self, name: str, args=None):
        self.send_line(name if args is None else f"{name} {json.dumps(args)}")

    def request(self, name: str, args=None) -> str:
        """Send a command, return the immediate OK/ERROR reply."""
        self.command(name, args)
        return self.read_msg()

    def request_task(self, name: str, args=None) -> str:
        """Send an async command; parse `OK {"task": id}`, return the id.

        A non-OK or unparseable reply leaves the connection in an unknown
        message state, so this raises (callers restart the worker) instead of
        continuing on a possibly desynced stream.
        """
        reply = self.request(name, args)
        if not reply.startswith("OK"):
            raise RuntimeError(f"{name} rejected: {reply[:200]}")
        try:
            task = json.loads(reply[2:].strip()).get("task")
        except (json.JSONDecodeError, AttributeError):
            task = None
        if not task:
            raise RuntimeError(f"{name}: no task id in reply: {reply[:200]}")
        return task

    def wait_task(self, deadline: float, task: str | None = None):
        """Read until the terminal message OF THIS TASK; return (kind, payload).

        2026-06-12 soundness fix: previously the FIRST terminal message was
        returned regardless of its task id, so a one-message desync bound the
        previous theory's result to the current request (observed in the
        corrupt test as Isabelle "proving" arithmetically false propositions,
        e.g. 54 mod 7 = 0; standalone resubmission fails them all). A terminal
        message carrying a DIFFERENT task id now raises, which makes the pool
        restart the worker (fail-closed) instead of mis-binding results.
        """
        self.sock.settimeout(deadline)
        try:
            t0 = time.time()
            while time.time() - t0 < deadline:
                msg = self.read_msg()
                kind = msg.split(" ", 1)[0]
                if kind not in ("FINISHED", "FAILED", "ERROR"):
                    continue
                body = msg[len(kind) + 1:] if " " in msg else "{}"
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {"raw": body}
                got = payload.get("task") if isinstance(payload, dict) else None
                if task is not None and got is not None and got != task:
                    raise RuntimeError(
                        f"desync: terminal message for task {got!r} while "
                        f"waiting for {task!r}")
                return kind, payload
            raise TimeoutError("no terminal message within deadline")
        finally:
            self.sock.settimeout(None)


class IsabelleWorker:
    """One resident isabelle server with a started session."""

    def __init__(self, wid: int, base_dir: Path, purge_every: int = PURGE_EVERY):
        self.wid = wid
        self.name = f"folpool_{os.getpid()}_{wid}"
        self.base_dir = Path(base_dir)
        self.master_dir = self.base_dir / f"worker_{wid}"
        self.purge_every = purge_every
        self.proc: subprocess.Popen | None = None
        self.jvm_pid: int | None = None   # REAL java pid (proc is a daemonizing launcher)
        self.conn: _Conn | None = None
        self.mgmt_conn: _Conn | None = None     # second connection: async purge
        self.addr: tuple[str, int, str] | None = None
        self.session_id: str | None = None
        self.counter = 0
        self.pending_names: list[str] = []
        self.lock = threading.Lock()
        self.mgmt_lock = threading.Lock()
        # {pgid: starttime} of setsid'd prover session leaders (bash+poly) owned
        # by this worker's JVM; killpg'd at stop/restart. Persisted to disk so a
        # crash reaper / next-run sweep can act without our in-memory state.
        self.reap_targets: dict[int, int] = {}

    def start(self):
        self.start_server()
        self.start_session()

    def start_server(self):
        """Spawn the server process and authenticate.

        NOT safe to run concurrently across workers: server registration
        writes to a shared SQLite registry ($ISABELLE_HOME_USER), which on
        NFS raises SQLITE_BUSY under concurrent writers. The pool serializes
        this phase and retries on registry contention.
        """
        if self.master_dir.exists():
            shutil.rmtree(self.master_dir)
        self.master_dir.mkdir(parents=True)
        last_banner = ""
        for attempt in range(5):
            self.proc = subprocess.Popen(
                [ISABELLE_BIN, "server", "-n", self.name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                preexec_fn=_die_with_parent,
            )
            banner = self.proc.stdout.readline().strip()
            m = BANNER_RE.search(banner)
            if m:
                # drain remaining stdout so the pipe never fills
                threading.Thread(target=self._drain, daemon=True).start()
                self.addr = (m.group(2), int(m.group(3)), m.group(4))
                self.conn = _Conn(*self.addr)
                self.mgmt_conn = _Conn(*self.addr)
                # The launcher bash (self.proc) daemonizes -> the real JVM is a
                # separate, reparented process. Track it by server name for ALL
                # poly reaping / RSS accounting; fall back to proc.pid if the
                # server did not daemonize (then proc.pid IS the java).
                self.jvm_pid = _find_jvm_pid(self.name) or self.proc.pid
                return
            last_banner = banner
            self.proc.kill()
            if "SQLITE_BUSY" in banner or "locked" in banner:
                time.sleep(1.0 + attempt)
                continue
            break
        raise RuntimeError(f"worker {self.wid}: bad banner {last_banner!r}")

    def start_session(self):
        task = self.conn.request_task("session_start", {
            "session": SESSION, "options": SESSION_OPTIONS,
        })
        kind, payload = self.conn.wait_task(deadline=300.0, task=task)
        if kind != "FINISHED":
            raise RuntimeError(f"worker {self.wid}: session_start {kind}: {payload}")
        self.session_id = payload["session_id"]
        self._load_prelude()
        # poly is guaranteed alive now (_load_prelude ran a real use_theories);
        # snapshot its session-leader group so we can reap it later.
        self._record_reap_targets()

    def _load_prelude(self):
        """Load the import set once; check() theories then `imports Prelude`,
        so the heavy HOL-Library import elaboration is paid once per worker
        instead of once per check (~2s -> sub-second)."""
        (self.master_dir / "Prelude.thy").write_text(
            f"theory Prelude\n{THEORY_IMPORTS}begin\n\nend\n")
        task = self.conn.request_task("use_theories", {
            "session_id": self.session_id,
            "theories": ["Prelude"],
            "master_dir": str(self.master_dir),
        })
        kind, payload = self.conn.wait_task(deadline=300.0, task=task)
        if kind != "FINISHED" or not payload.get("ok"):
            detail = json.dumps(payload.get("errors", payload))[:400]
            raise RuntimeError(
                f"worker {self.wid}: prelude load failed: {kind} {detail}")

    def _drain(self):
        try:
            for _ in self.proc.stdout:
                pass
        except ValueError:
            pass

    def _record_reap_targets(self):
        """Snapshot the JVM's setsid'd prover session leaders (bash+poly).
        These share a group distinct from ours, so killpg reaps them without
        touching the training process."""
        if self.jvm_pid is None:
            return
        self.reap_targets = _prover_groups(self.jvm_pid, os.getpgrp())
        self._persist_reap()

    def _persist_reap(self):
        st = _read_stat(self.jvm_pid) if self.jvm_pid else None
        rec = {"wid": self.wid, "owner_pid": os.getpid(),
               "jvm_pid": self.jvm_pid if self.jvm_pid else 0,
               "jvm_start": st[3] if st else 0,
               "groups": {str(g): s for g, s in self.reap_targets.items()}}
        dst = self.base_dir / f"worker_{self.wid}.reap"
        tmp = self.base_dir / f"worker_{self.wid}.reap.tmp"
        try:
            tmp.write_text(json.dumps(rec))
            os.replace(tmp, dst)  # atomic overwrite
        except OSError:
            pass

    def _reap_leftover(self):
        """Opportunistic re-walk (catches a prover respawned with a new pid;
        no-op if the JVM is already dead), then killpg every recorded group."""
        if self.jvm_pid is not None:
            for pgid, start in _prover_groups(self.jvm_pid, os.getpgrp()).items():
                self.reap_targets.setdefault(pgid, start)
        for pgid, start in list(self.reap_targets.items()):
            _kill_pgid(pgid, start)
        # SIGKILL the daemonized JVM itself: it reparented to init and is NOT in
        # any prover group, so the killpg above never touches it. graceful=False
        # would otherwise leak one java (and its next poly) per recycle.
        if self.jvm_pid is not None:
            try:
                os.kill(self.jvm_pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def check(self, theorem_code: str) -> dict:
        """Verify one theorem; returns {success, elapsed, errors}."""
        with self.lock:
            self.counter += 1
            name = f"V{self.counter}_{uuid.uuid4().hex[:8]}"
            (self.master_dir / f"{name}.thy").write_text(
                f"theory {name}\n  imports Prelude\nbegin\n\n{theorem_code}\n\nend\n")
            t0 = time.time()
            task = self.conn.request_task("use_theories", {
                "session_id": self.session_id,
                "theories": [name],
                "master_dir": str(self.master_dir),
                # completion is detected by polling; the default check_delay
                # of 0.5s quantizes every call to ~2s. 0.05s cuts the idle wait
                "check_delay": 0.05,
            })
            kind, payload = self.conn.wait_task(CHECK_DEADLINE, task=task)
            elapsed = time.time() - t0
            if kind != "FINISHED":
                return {"success": False, "elapsed": elapsed,
                        "errors": [f"{kind}: {str(payload)[:200]}"]}
            errors = [e.get("message", str(e)) if isinstance(e, dict) else str(e)
                      for e in payload.get("errors", [])]
            # 2026-06-12 soundness fix #2: payload "ok" means "no errors SO
            # FAR", not "fully processed". A tactic still running when the
            # headless watchdog (15s) aborts the theory yields ok=true,
            # errors=[], node status {percentage: 99, running: 1,
            # consolidated: false} -- so every check slower than the watchdog
            # used to count as a successful proof (this is how the bogus
            # "sqrt metis rung" was born). Require the checked node to be
            # fully finished AND consolidated.
            node_ok = False
            for nd in payload.get("nodes", []):
                if nd.get("theory_name", "").endswith(name):
                    st = nd.get("status", {})
                    node_ok = bool(st.get("ok") and st.get("consolidated")
                                   and not st.get("canceled")
                                   and st.get("failed", 1) == 0
                                   and st.get("percentage") == 100)
                    if not node_ok and not errors:
                        errors = [f"node not consolidated: {st}"]
                    break
            self.pending_names.append(name)
            if self.counter % self.purge_every == 0:
                batch = list(self.pending_names)
                self.pending_names.clear()
                threading.Thread(target=self._purge_async, args=(batch,),
                                 daemon=True).start()
            return {"success": bool(payload.get("ok")) and node_ok,
                    "elapsed": elapsed, "errors": errors}

    def _purge_async(self, names: list[str]):
        """Unload checked theories via the management connection.

        Runs in a background thread so the verification path never blocks:
        purge_theories occasionally stalls ~60-70s when it coincides with the
        server's background document cleanup (headless_prune_delay /
        headless_commit_cleanup_delay).

        Purges only explicit V-theory names (+ master_dir, name resolution is
        master_dir-relative). NEVER purge-and-resubmit a reused node name:
        the server's document blob CONCATENATES the resubmitted content onto
        the old one ("Illegal theory header" at a line past EOF) — that is
        why Prelude stays loaded forever and V names are unique.
        """
        try:
            with self.mgmt_lock:
                self.mgmt_conn.request("purge_theories", {
                    "session_id": self.session_id,
                    "theories": names,
                    "master_dir": str(self.master_dir),
                })
            for old in names:
                try:
                    (self.master_dir / f"{old}.thy").unlink()
                except OSError:
                    pass
        except (EOFError, OSError) as e:
            print(f"[pool] worker {self.wid} async purge failed: {e!r}", flush=True)

    def stop(self, graceful: bool = True):
        # graceful=False is the FAST path for a wedged worker: the graceful
        # session_stop below would just block on the same wedged socket (up to
        # its 15s timeout), so skip straight to SIGKILL + killpg.
        if graceful:
            # Graceful path: let Isabelle reap its own prover via session_stop.
            # BOUNDED with a socket timeout so a wedged worker cannot stall the
            # restart forever (session_stop/command have no timeout of their own).
            try:
                if self.mgmt_conn is not None:
                    self.mgmt_conn.close()
                if self.conn is not None:
                    try:
                        self.conn.sock.settimeout(15.0)
                    except OSError:
                        pass
                    if self.session_id is not None:
                        self.conn.request("session_stop", {"session_id": self.session_id})
                        self.conn.wait_task(deadline=15.0)
                    self.conn.command("shutdown")
                    self.conn.close()
            except (OSError, EOFError, TimeoutError, RuntimeError):
                pass
        else:
            for c in (self.mgmt_conn, self.conn):
                try:
                    if c is not None:
                        c.close()          # unblocks any wedged reader on it
                except (OSError, EOFError):
                    pass
        # Kill the JVM (PDEATHSIG-armed; also cuts bash_process).
        if self.proc is not None:
            if not graceful:
                self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        # LAYER 1: killpg the setsid'd prover group(s) the graceful path
        # leaked. Runs UNCONDITIONALLY -- this is the fix for the OOM leak
        # (a wedged worker's session_stop times out and orphans poly otherwise).
        self._reap_leftover()


class IsabelleServerPool:
    """N resident workers; check() dispatches to an idle worker."""

    # In-memory theorem memo. Key = the exact theorem_code string (name-free:
    # make_theorem emits a fixed `theorem chk:`; the unique V-name is added by
    # the worker when writing the .thy). Verification is deterministic, so a
    # SUCCESS result can always be replayed. FAILURES are cached only when
    # they are genuine fast refusals — worker errors, watchdog aborts ("node
    # not consolidated") and anything near the 15s watchdog are load-dependent
    # flakes that must NOT be frozen into permanent `x` verdicts.
    CACHE_MAX_ENTRIES = 200_000          # ~300B each -> tens of MB, plenty
    CACHE_FAIL_FAST_S = 10.0             # failure cacheable only below this
    MONITOR_INTERVAL_S = 60.0           # orphan-sweep cadence
    WORKER_RESTART_WARN = 20            # log every N restarts of one worker
    # Per-check hard ceiling on ONE worker's poly-tree RSS (KB). Exceeding it
    # recycles the whole JVM (resets every poly). This is THE bound on host
    # memory (2026-07-08 OOM root cause: leaked zombie-proof polys accumulate,
    # --maxheap does not cap RSS). Total host poly footprint <= num_workers x
    # cap regardless of step count. 7GB x 8 workers = 56GB steady.
    WORKER_RSS_CAP_KB = int(
        os.environ.get("ISABELLE_WORKER_RSS_CAP_GB", "7")) * 1024 * 1024
    MAX_CHECK_ATTEMPTS = 2             # 1 retry on a FRESH worker after a
    #                                    worker_error (hard-timeout / crash):
    #                                    the wedge is transient (the theorem
    #                                    verifies on a healthy worker), so a
    #                                    single retry recovers it instead of
    #                                    losing the step to a fail-closed x.
    # Runaway-poly reaper (2026-07-09 step-135 ~130min wedge). A native tactic
    # (auto/sos/sledgehammer-style) can ignore Isabelle's COOPERATIVE interrupt
    # and spin a poly at 100% CPU forever -- verify_timeout / CHECK_DEADLINE are
    # both cooperative and never end it. Such leaked "zombie proof" polys are NOT
    # caught by the orphan sweep (their JVM is alive) nor the RSS cap (they burn
    # CPU, not RAM); they accumulate, starve cores, and make every worker.start()
    # (session reload) hang -> full-pool wedge. The monitor SIGKILLs any poly
    # that stays ~100% busy for longer than any legit proof (verify_timeout) can
    # run. Threshold in CPU-seconds; needs >= that long of sustained 100% CPU.
    RUNAWAY_BUSY_FRAC = 0.85   # >=85% of a monitor interval on CPU == spinning
    RUNAWAY_CPU_S = float(os.environ.get("ISABELLE_RUNAWAY_CPU_S", "90"))

    def __init__(self, num_workers: int = 4,
                 base_dir: Path | str = "/tmp/isabelle_pool",
                 purge_every: int = PURGE_EVERY):
        self.base_dir = Path(base_dir)
        self.num_workers = num_workers
        self.workers = [IsabelleWorker(i, self.base_dir, purge_every)
                        for i in range(num_workers)]
        self.idle: Queue[IsabelleWorker] = Queue()
        # Restart bookkeeping: worker wedges (usually verify_timeout hits)
        # are no longer printed per-request (log spam). Track counts so the
        # timeout frequency stays observable without flooding stdout.
        self.restart_count = 0
        self.restart_reasons: dict[str, int] = {}
        self.worker_restart_counts: dict[int, int] = {}
        # Orphan-reaper machinery (2026-07-08 OOM fix): periodic monitor thread
        # + detached crash reaper. See _reap_orphans_once / reaper.py.
        self._stop_monitor = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._watchdog: subprocess.Popen | None = None
        # Theorem memo + single-flight: with 16 rollouts of one prompt being
        # verified concurrently, identical theorems arrive at the same time;
        # followers wait for the leader's result instead of re-proving.
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        # Runaway-poly reaper state: {poly_pid: (cpu_jiffies, starttime, streak)}
        # of consecutive monitor ticks the poly spent ~100% on CPU.
        self._poly_cpu: dict[int, tuple] = {}
        self._clk_tck = os.sysconf("SC_CLK_TCK") or 100
        self._runaway_streak = max(2, int(
            (self.RUNAWAY_CPU_S + self.MONITOR_INTERVAL_S - 1)
            // self.MONITOR_INTERVAL_S))

    def start(self):
        t0 = time.time()
        # LAYER 2: reap prover groups left by a crashed previous run of THIS
        # engine (freed before we spawn, so the memory is available). Safe:
        # only sibling base_dirs whose owner pid is dead, matched by our naming.
        try:
            _sweep_stale_pools(self.base_dir)
        except Exception as e:
            _POOL_LOG.warning("startup sweep failed: %r", e)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # phase 1: spawn server processes sequentially (shared SQLite registry
        # on NFS cannot take concurrent registrations, see start_server)
        for w in self.workers:
            w.start_server()
        # phase 2: heavy session loading in parallel
        threads = [threading.Thread(target=w.start_session) for w in self.workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for w in self.workers:
            if w.session_id is None:
                raise RuntimeError(f"worker {w.wid} failed to start")
            self.idle.put(w)
        # LAYER 4: detached crash reaper + LAYER 3: periodic orphan monitor.
        self._spawn_watchdog()
        self._monitor_thread = threading.Thread(target=self._monitor_loop,
                                                 daemon=True)
        self._monitor_thread.start()
        print(f"[pool] {len(self.workers)} workers ready in {time.time()-t0:.1f}s",
              flush=True)

    def _spawn_watchdog(self):
        """Detached reaper (own session) that survives our OOM-SIGKILL and
        killpg's the recorded prover groups when this process dies."""
        try:
            reaper = Path(__file__).with_name("reaper.py")
            st = _read_stat(os.getpid())
            self._watchdog = subprocess.Popen(
                [sys.executable, str(reaper), str(self.base_dir),
                 str(os.getpid()), str(st[3] if st else 0)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except Exception as e:
            _POOL_LOG.warning("watchdog reaper not started: %r", e)
            self._watchdog = None

    def _monitor_loop(self):
        while not self._stop_monitor.wait(self.MONITOR_INTERVAL_S):
            try:
                self._reap_orphans_once()
            except Exception as e:
                _POOL_LOG.debug("orphan monitor error: %r", e)
            try:
                self._reap_runaway_polys_once()
            except Exception as e:
                _POOL_LOG.debug("runaway monitor error: %r", e)

    def _reap_orphans_once(self):
        """Kill self-led prover groups NOT owned by any live worker -- catches
        polys leaked by a restart whose parent JVM has since died."""
        live = set()
        for w in self.workers:
            if w.jvm_pid is not None and _read_stat(w.jvm_pid) is not None:
                live.update(_prover_groups(w.jvm_pid, os.getpgrp()).keys())
                w.reap_targets.update(_prover_groups(w.jvm_pid, os.getpgrp()))
        # any recorded target whose group leader is alive but no live worker
        # claims it (its JVM died) is an orphan -> reap.
        for w in self.workers:
            for pgid, start in list(w.reap_targets.items()):
                if pgid not in live and _kill_pgid(pgid, start):
                    _POOL_LOG.debug("monitor reaped orphan prover group %d", pgid)
                    w.reap_targets.pop(pgid, None)

    def _reap_runaway_polys_once(self):
        """SIGKILL abandoned "zombie proof" polys: a native tactic that ignored
        Isabelle's cooperative interrupt keeps spinning ~100% CPU inside a
        still-healthy JVM after the check already returned; these accumulate,
        starve cores, and hang every worker.start() -> full-pool wedge
        (2026-07-09 step-135, ~130min).

        DISCRIMINATOR (2026-07-10 fix): sustained 100% CPU alone is NOT enough
        -- a healthy ML poly grinding a QUEUE of checks back to back also
        sustains 100% for minutes, and v1 of this reaper mass-killed live
        provers mid-burst (8 kills in 12min incl. 3-at-once = a whole pool's
        provers) and wedged the run. A true zombie is distinguishable by its
        JVM's poly tree: a healthy worker runs ~2 polys; when a tactic thread
        is abandoned, the JVM spawns a REPLACEMENT for the next check, so the
        tree grows to >= 3. Reap only spinning polys in an oversized tree,
        always protecting the 2 newest-started (the live prover pair). A
        runaway that IS the current prover needs no reaping here: its check
        hits the hard timeout and the worker restart killpg's its group."""
        interval = self.MONITOR_INTERVAL_S
        denom = interval * self._clk_tck
        seen: dict[int, tuple] = {}
        runaways: list[int] = []
        for w in self.workers:
            if w.jvm_pid is None or _read_stat(w.jvm_pid) is None:
                continue
            tree: list[tuple] = []   # (pid, start, streak)
            for pid in _descendants(w.jvm_pid):
                ps = _poly_cpu_state(pid)
                if ps is None:
                    continue
                _state, cpu, start = ps
                prev = self._poly_cpu.get(pid)
                streak = 0
                if prev is not None and prev[1] == start and denom > 0:
                    busy = (cpu - prev[0]) / denom
                    streak = prev[2] + 1 if busy >= self.RUNAWAY_BUSY_FRAC else 0
                seen[pid] = (cpu, start, streak)
                tree.append((pid, start, streak))
            if len(tree) <= 2:
                continue             # healthy-sized tree: never reap
            tree.sort(key=lambda t: t[1], reverse=True)
            for pid, _start, streak in tree[2:]:   # protect 2 newest
                if streak >= self._runaway_streak:
                    runaways.append(pid)
        self._poly_cpu = seen
        for pid in runaways:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if runaways:
            self.restart_reasons["runaway_poly"] = (
                self.restart_reasons.get("runaway_poly", 0) + len(runaways))
            _POOL_LOG.warning(
                "runaway-poly reaper: SIGKILLed %d zombie poly %s (>= %.0fs "
                "sustained 100%% CPU in an oversized (>2) poly tree)",
                len(runaways), runaways, self._runaway_streak * interval)

    def check(self, theorem_code: str) -> dict:
        # ---- theorem memo (single-flight) ----
        while True:
            with self._cache_lock:
                cached = self._cache.get(theorem_code)
                if cached is not None:
                    self._cache.move_to_end(theorem_code)
                    self.cache_hits += 1
                    out = dict(cached)
                    out["cache_hit"] = True
                    out["queue_wait"] = 0.0
                    out["check_time"] = 0.0
                    return out
                ev = self._inflight.get(theorem_code)
                if ev is None:
                    self._inflight[theorem_code] = threading.Event()
                    self.cache_misses += 1
                    break                      # we are the leader: go prove it
            # follower: wait for the leader, then re-read the cache. If the
            # leader's result was uncacheable (timeout flake), loop and become
            # the new leader — re-proving is the correct fallback.
            # BOUNDED wait: a wedged leader must not deadlock its followers
            # forever (2026-07-08 step-135 wedge). If the leader has not
            # signalled within the hard budget, drop its stale in-flight marker
            # and loop to re-become the leader ourselves.
            if not ev.wait(CHECK_DEADLINE + 30.0):
                with self._cache_lock:
                    if self._inflight.get(theorem_code) is ev:
                        self._inflight.pop(theorem_code, None)
        try:
            # disk layer: a verdict persisted by a previous run/process (same
            # policy as the memory memo). Hit -> promote to memory and return.
            disk = _thm_disk_load(theorem_code)
            if disk is not None and "success" in disk:
                with self._cache_lock:
                    self._cache[theorem_code] = dict(disk)
                    while len(self._cache) > self.CACHE_MAX_ENTRIES:
                        self._cache.popitem(last=False)
                    self.cache_hits += 1
                out = dict(disk)
                out["cache_hit"] = True
                out["queue_wait"] = 0.0
                out["check_time"] = 0.0
                return out
            result = self._check_uncached(theorem_code)
            elapsed = float(result.get("elapsed", 0.0) or 0.0)
            cacheable = result.get("success") or (
                not result.get("worker_error")
                and elapsed < self.CACHE_FAIL_FAST_S
                and not any("not consolidated" in str(e)
                            for e in result.get("errors", [])))
            if cacheable:
                entry = {k: result[k] for k in ("success", "elapsed", "errors")
                         if k in result}
                with self._cache_lock:
                    self._cache[theorem_code] = entry
                    while len(self._cache) > self.CACHE_MAX_ENTRIES:
                        self._cache.popitem(last=False)
                _thm_disk_store(theorem_code, entry)
            return result
        finally:
            with self._cache_lock:
                ev = self._inflight.pop(theorem_code, None)
            if ev is not None:
                ev.set()

    def _book_restart(self, worker, reason: str):
        """Count a worker restart and log at WARN every WORKER_RESTART_WARN."""
        self.restart_count += 1
        self.restart_reasons[reason] = self.restart_reasons.get(reason, 0) + 1
        n = self.worker_restart_counts.get(worker.wid, 0) + 1
        self.worker_restart_counts[worker.wid] = n
        if n % self.WORKER_RESTART_WARN == 0:
            _POOL_LOG.warning("worker %d restarted %d times (reasons=%s)",
                              worker.wid, n, self.restart_reasons)
        else:
            _POOL_LOG.debug("worker %d restart (%s)", worker.wid, reason)

    def _run_one_check(self, worker, theorem_code: str) -> dict:
        """One check on `worker` behind a HARD wall-clock cap; force-restart the
        worker on wedge/error; ALWAYS return it to idle. This cap GUARANTEES a
        return even if worker.check() (or the PIDE server) wedges past its own
        CHECK_DEADLINE -- otherwise a wedged single-flight leader never sets its
        Event and every follower, then (once the idle queue drains) every other
        request, deadlocks forever (2026-07-08 step-135 wedge: whole reward step
        hung ~85min, every worker idle). The cap sits above CHECK_DEADLINE so
        wait_task's own timeout wins in the common case; this is the backstop.
        """
        box: dict = {}

        def _run():
            try:
                box["r"] = worker.check(theorem_code)
            except Exception as e:  # noqa: BLE001 -- any failure -> restart
                box["e"] = e

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(CHECK_DEADLINE + 15.0)
        try:
            if th.is_alive():
                # Wedged past the hard cap. stop() closes the socket + kills the
                # JVM, which unblocks the wedged thread's socket read so it can
                # exit and release worker.lock; then rebuild the worker.
                self._book_restart(worker, "hard_timeout")
                try:
                    worker.stop(graceful=False)  # fast kill + killpg reap
                except Exception:
                    pass
                th.join(10.0)
                try:
                    worker.start()
                except Exception as se:  # noqa: BLE001 -- a failed session
                    # restart must NOT propagate out of the check (it would
                    # bubble an exception through the reward path); the broken
                    # worker goes back to idle and the NEXT check on it fails
                    # fast and retries the restart.
                    _POOL_LOG.warning("worker %d restart failed: %r",
                                      worker.wid, se)
                return {"success": False, "elapsed": CHECK_DEADLINE + 15.0,
                        "worker_error": True,
                        "errors": ["hard timeout: worker force-restarted"]}
            if "e" in box:
                e = box["e"]
                self._book_restart(worker, type(e).__name__)
                try:
                    worker.stop(graceful=False)  # fast kill + killpg reap
                except Exception:
                    pass
                try:
                    worker.start()
                except Exception as se:  # noqa: BLE001 -- see above
                    _POOL_LOG.warning("worker %d restart failed: %r",
                                      worker.wid, se)
                return {"success": False, "elapsed": 0.0, "worker_error": True,
                        "errors": [f"worker error: {e!r}"]}
            return box["r"]
        finally:
            # Per-check RSS ceiling (2026-07-08 OOM root cause). A check can
            # leave a runaway "zombie proof" poly alive inside a still-live JVM
            # (Isabelle's headless watchdog aborts the theory + our check
            # returns "unproved", but the PolyML tactic thread keeps running and
            # eating RAM; the JVM then spawns a fresh poly for the next check).
            # Such leaked polys accumulate over steps until host-OOM, and
            # --maxheap does NOT cap their RSS. Recycling the whole JVM when its
            # poly-tree RSS exceeds the cap resets all poly and hard-bounds host
            # memory to num_workers x cap. Done PER-CHECK (not on a 60s idle
            # tick) so a step where every worker stays busy is still bounded.
            # A fresh worker (just restarted above) has tiny RSS -> no-op.
            try:
                if (worker.jvm_pid is not None
                        and _poly_tree_rss_kb(worker.jvm_pid)
                        > self.WORKER_RSS_CAP_KB):
                    self._book_restart(worker, "rss_cap")
                    worker.stop(graceful=False)
                    worker.start()
            except Exception as e:  # noqa: BLE001 -- recycle must never crash
                _POOL_LOG.debug("rss recycle failed w%d: %r", worker.wid, e)
            self.idle.put(worker)

    def _check_uncached(self, theorem_code: str) -> dict:
        # Retry a worker_error (transient wedge / crash) on a FRESH worker up to
        # MAX_CHECK_ATTEMPTS times: the wedge is not a property of the theorem
        # (it verifies fine on a healthy worker), so one retry recovers the
        # verification instead of losing the step to a fail-closed `x`. A real
        # proof failure or a success is final -- never retried.
        t0 = time.time()
        queue_wait = None
        result = None
        for attempt in range(self.MAX_CHECK_ATTEMPTS):
            worker = self.idle.get()
            if queue_wait is None:
                queue_wait = time.time() - t0
            t1 = time.time()
            result = self._run_one_check(worker, theorem_code)
            result["worker"] = worker.wid
            result["check_time"] = time.time() - t1
            result["attempts"] = attempt + 1
            if result.get("success") or not result.get("worker_error"):
                break
        result["queue_wait"] = queue_wait
        return result

    def shutdown(self):
        self._stop_monitor.set()
        for w in self.workers:
            w.stop()
        # A clean shutdown never trips the crash reaper's pidfd; terminate it.
        if self._watchdog is not None:
            try:
                self._watchdog.terminate()
            except OSError:
                pass
            self._watchdog = None
        # Registry no longer needed after a clean stop.
        for reap in self.base_dir.glob("worker_*.reap"):
            try:
                reap.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    # smoke test: one worker, purge path exercised every 2 checks
    pool = IsabelleServerPool(num_workers=1, purge_every=2)
    pool.start()
    good = '''theorem t1:
    assumes "(x::int) = 3"
    shows "x + 1 = 4"
proof -
    have "(3::int) + 1 = 4" sorry
    thus ?thesis using assms by simp
qed'''
    # only the shows changes: the sorry'd fact stays "3 + 1 = 4", so the final
    # step must derive "x + 1 = 5" from {x = 3, 3 + 1 = 4} -> tactic fails
    bad = good.replace('shows "x + 1 = 4"', 'shows "x + 1 = 5"')
    # broken theorem header: must fail fast via watchdog, not wedge 60s
    broken = 'theorem t2 ((( malformed\n    shows "True"\nproof - qed'
    for label, thm in [("good", good), ("bad", bad),
                       ("broken-header", broken), ("good-again", good)]:
        r = pool.check(thm)
        print(f"[smoke] {label}: success={r['success']} elapsed={r['elapsed']:.2f}s "
              f"errors={[e[:80] for e in r['errors'][:1]]}", flush=True)
    pool.shutdown()
    print("[smoke] done")
