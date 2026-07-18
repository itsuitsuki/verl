"""Resident Isabelle server worker and session lifecycle."""
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Full, Queue

from verl.utils.isabelle_utils import state_classes
from verl.utils.isabelle_utils._server_pool import (config, connection,
                                                    processes)


class IsabelleWorker:
    """One resident Isabelle server with one loaded session."""

    def __init__(self, wid: int, base_dir: Path, purge_every: int = config.PURGE_EVERY,
                 session: str | None = None, options=None,
                 imports: str | None = None, verify_timeout: float | None = None):
        self.wid = wid
        self.session = session or config.SESSION
        self.options = list(options) if options is not None else list(config.SESSION_OPTIONS)
        self.imports = imports if imports is not None else config.THEORY_IMPORTS
        self.verify_timeout = float(verify_timeout) if verify_timeout is not None else config.VERIFY_TIMEOUT
        self.name = f"folpool_{os.getpid()}_{wid}"
        self.base_dir = Path(base_dir)
        self.master_dir = self.base_dir / f"worker_{wid}"
        self.purge_every = purge_every
        self.proc: subprocess.Popen | None = None
        self.jvm_pid: int | None = None   # daemonized server JVM
        self.conn: connection._Conn | None = None
        self.mgmt_conn: connection._Conn | None = None     # second connection: async purge
        self.addr: tuple[str, int, str] | None = None
        self.session_id: str | None = None
        self.counter = 0
        self.pending_names: list[str] = []
        self.lock = threading.Lock()
        self.mgmt_lock = threading.Lock()
        self.reap_targets: dict[int, int] = {}
        self._purge_q: Queue = Queue(maxsize=8)
        self._purge_thread = threading.Thread(
            target=self._purge_loop, daemon=True, name=f"isa-purge-w{wid}")
        self._purge_thread.start()

    def start(self):
        self.start_server()
        self.start_session()

    def start_server(self):
        """Start and authenticate the server process.

        Server registration writes to a shared SQLite registry, so the pool calls this method sequentially across workers and this method retries registry contention."""
        if self.master_dir.exists():
            shutil.rmtree(self.master_dir)
        self.master_dir.mkdir(parents=True)
        last_banner = ""
        for attempt in range(5):
            self.proc = subprocess.Popen(
                [config.ISABELLE_BIN, "server", "-n", self.name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                preexec_fn=processes._die_with_parent,
            )
            banner = self.proc.stdout.readline().strip()
            m = config.BANNER_RE.search(banner)
            if m:
                threading.Thread(target=self._drain, daemon=True).start()
                self.addr = (m.group(2), int(m.group(3)), m.group(4))
                self.conn = connection._Conn(*self.addr)
                self.mgmt_conn = connection._Conn(*self.addr)
                self.jvm_pid = processes._find_jvm_pid(self.name) or self.proc.pid
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
            "session": self.session, "options": self.options,
        })
        kind, payload = self.conn.wait_task(deadline=300.0, task=task)
        if kind != "FINISHED":
            raise RuntimeError(f"worker {self.wid}: session_start {kind}: {payload}")
        self.session_id = payload["session_id"]
        self._load_prelude()
        self._record_reap_targets()

    def _load_prelude(self):
        """Load the configured imports once into a resident Prelude theory.

        Per-check theories import Prelude so repeated checks do not reload the full library set."""
        (self.master_dir / "Prelude.thy").write_text(
            f"theory Prelude\n{self.imports}begin\n\nend\n")
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
        """Record self-led prover process groups owned by this worker's JVM.

        The registry is persisted so cleanup remains possible after the Python process dies."""
        if self.jvm_pid is None:
            return
        self.reap_targets = processes._prover_groups(self.jvm_pid, os.getpgrp())
        self._persist_reap()

    def _persist_reap(self):
        st = processes._read_stat(self.jvm_pid) if self.jvm_pid else None
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
        """Refresh and terminate all recorded prover groups, then terminate the daemonized JVM."""
        if self.jvm_pid is not None:
            for pgid, start in processes._prover_groups(self.jvm_pid, os.getpgrp()).items():
                self.reap_targets.setdefault(pgid, start)
        for pgid, start in list(self.reap_targets.items()):
            processes._kill_pgid(pgid, start)
        if self.jvm_pid is not None:
            try:
                os.kill(self.jvm_pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def check(self, theorem_code: str) -> "state_classes.VerificationOutcome":
        """Check one theorem and classify the PIDE result.

        A proof succeeds only when exactly one target node is present, fully processed, and consolidated. Missing, duplicate, or canceled nodes are infrastructure failures; an unfinished node is incomplete."""
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
                "check_delay": 0.02,
            })
            kind, payload = self.conn.wait_task(self.verify_timeout, task=task)
            elapsed = time.time() - t0
            if kind != "FINISHED":
                return state_classes.VerificationOutcome(
                    outcome=state_classes.ProofOutcome.WORKER_ERROR,
                    elapsed=elapsed, errors=[f"{kind}: {str(payload)[:200]}"])
            errors = [e.get("message", str(e)) if isinstance(e, dict) else str(e)
                      for e in payload.get("errors", [])]
            self.pending_names.append(name)
            if self.counter % self.purge_every == 0:
                batch = list(self.pending_names)
                self.pending_names.clear()
                self._enqueue_purge(batch)
            node_st = None
            node_seen = 0
            for nd in payload.get("nodes", []):
                if nd.get("theory_name", "").endswith(name):
                    node_seen += 1
                    node_st = nd.get("status", {})
            if node_seen != 1:
                reason = ("target node missing from PIDE payload"
                          if node_seen == 0
                          else f"duplicate target nodes ({node_seen})")
                return state_classes.VerificationOutcome(
                    outcome=state_classes.ProofOutcome.WORKER_ERROR,
                    elapsed=elapsed, errors=[reason])
            st = node_st
            if st.get("canceled"):
                return state_classes.VerificationOutcome(
                    outcome=state_classes.ProofOutcome.WORKER_ERROR,
                    elapsed=elapsed, errors=[f"node canceled: {st}"])
            consolidated_full = bool(st.get("consolidated")
                                     and st.get("percentage") == 100)
            node_ok = bool(st.get("ok") and consolidated_full
                           and st.get("failed", 1) == 0)
            if not node_ok and not errors:
                errors = [f"node not consolidated: {st}"]
            if bool(payload.get("ok")) and node_ok:
                outcome = state_classes.ProofOutcome.PROVED
            elif not node_ok and not consolidated_full:
                outcome = state_classes.ProofOutcome.INCOMPLETE
            else:
                outcome = state_classes.ProofOutcome.UNPROVED
            return state_classes.VerificationOutcome(
                outcome=outcome, elapsed=elapsed, errors=errors)

    def _enqueue_purge(self, batch: list[str]):
        """Queue theory names for asynchronous removal.

        When the bounded queue is full, the oldest pending batch is merged into the new batch so no theory name is discarded."""
        while True:
            try:
                self._purge_q.put_nowait(batch)
                return
            except Full:
                try:
                    batch = self._purge_q.get_nowait() + batch
                except Empty:
                    continue   # raced with the purge thread; retry the put

    def _purge_loop(self):
        while True:
            batch = self._purge_q.get()
            try:
                self._purge_async(batch)
            except Exception as e:  # noqa: BLE001 -- the loop must never die
                print(f"[pool] worker {self.wid} purge loop error: {e!r}",
                      flush=True)

    def _purge_async(self, names: list[str]):
        """Remove checked theories through the management connection.

        Each check uses a unique theory name because resubmitting a purged name can reuse stale document content. Prelude remains loaded for the worker's lifetime."""
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
        if graceful:
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
        self._reap_leftover()
