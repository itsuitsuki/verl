"""Scheduling, caching, monitoring, and lifecycle for Isabelle workers."""
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Queue

from verl.utils.isabelle_utils import cache_lock, state_classes
from verl.utils.isabelle_utils._server_pool import (config, processes,
                                                    theorem_cache)
from verl.utils.isabelle_utils._server_pool.worker import IsabelleWorker

_POOL_LOG = logging.getLogger("verl.isabelle.server_pool")


class IsabelleServerPool:
    """Manage resident Isabelle workers behind `start`, `check`, `submit`, and `shutdown`.

    Stable proof outcomes are memoized and coordinated across processes. Infrastructure failures may be retried on a fresh worker, while incomplete outcomes fail closed and remain uncacheable.

    The pool also bounds each worker's Poly/ML process-tree memory and removes orphaned or abandoned prover processes."""

    CACHE_MAX_ENTRIES = 200_000          # ~300B each -> tens of MB, plenty
    CACHE_FAIL_FAST_S = 10.0             # failure cacheable only below this
    MONITOR_INTERVAL_S = 60.0           # orphan-sweep cadence
    WORKER_RESTART_WARN = 20            # log every N restarts of one worker
    TIMEOUT_HISTORY_TTL = 900.0           # seconds a repeated timeout short-circuits
    TIMEOUT_HISTORY_MAX = 8192            # bounded LRU of theorem timeout counts
    EACH_WORKER_PROC_TREE_MEM_MAX_KB = 12 * 1024 * 1024
    MAX_CHECK_ATTEMPTS = 2             # one retry on another worker
    RUNAWAY_BUSY_FRAC = 0.85   # >=85% of a monitor interval on CPU == spinning
    RUNAWAY_CPU_S = 90.0       # class default; per-instance override via __init__ (Hydra config)

    def __init__(self, num_workers: int = 4,
                 base_dir: Path | str = "/tmp/isabelle_pool",
                 purge_every: int = config.PURGE_EVERY,
                 each_worker_proc_tree_mem_max_gb: float | None = None,
                 session: str | None = None, options=None,
                 imports: str | None = None,
                 verify_timeout: float | None = None,
                 runaway_cpu_seconds: float | None = None):
        self.base_dir = Path(base_dir)
        self.num_workers = num_workers
        if runaway_cpu_seconds is not None:
            self.RUNAWAY_CPU_S = float(runaway_cpu_seconds)
        if each_worker_proc_tree_mem_max_gb is not None:
            self.EACH_WORKER_PROC_TREE_MEM_MAX_KB = int(
                float(each_worker_proc_tree_mem_max_gb) * 1048576
            )
        self.session = session or config.SESSION
        self.options = list(options) if options is not None else list(config.SESSION_OPTIONS)
        self.imports = imports if imports is not None else config.THEORY_IMPORTS
        self.verify_timeout = float(verify_timeout) if verify_timeout is not None else config.VERIFY_TIMEOUT
        self._thm_fprint = theorem_cache._thm_env_fprint(self.session, self.imports, self.options)
        self.workers = [IsabelleWorker(i, self.base_dir, purge_every,
                                       session=session, options=options,
                                       imports=imports,
                                       verify_timeout=self.verify_timeout)
                        for i in range(num_workers)]
        self.idle: Queue[IsabelleWorker] = Queue()
        self.restart_count = 0
        self.restart_reasons: dict[str, int] = {}
        self.worker_restart_counts: dict[int, int] = {}
        self._stop_monitor = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._watchdog: subprocess.Popen | None = None
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._pending_checks: dict[str, threading.Event] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._timeout_history: OrderedDict[str, list] = OrderedDict()
        self.premise_consistency_unknown_short_circuits = 0
        self._poly_cpu: dict[int, tuple] = {}
        self._clk_tck = os.sysconf("SC_CLK_TCK") or 100
        self._runaway_streak = max(2, int(
            (self.RUNAWAY_CPU_S + self.MONITOR_INTERVAL_S - 1)
            // self.MONITOR_INTERVAL_S))
        self._reqq: Queue = Queue()
        self._dispatchers: list[threading.Thread] = []
        self._disp_lock = threading.Lock()

    def start(self):
        t0 = time.time()
        try:
            processes._sweep_stale_pools(self.base_dir)
        except Exception as e:
            _POOL_LOG.warning("startup sweep failed: %r", e)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for w in self.workers:
            w.start_server()
        threads = [threading.Thread(target=w.start_session) for w in self.workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for w in self.workers:
            if w.session_id is None:
                raise RuntimeError(f"worker {w.wid} failed to start")
            self.idle.put(w)
        self._spawn_watchdog()
        self._monitor_thread = threading.Thread(target=self._monitor_loop,
                                                 daemon=True)
        self._monitor_thread.start()
        print(f"[pool] {len(self.workers)} workers ready in {time.time()-t0:.1f}s",
              flush=True)

    def _spawn_watchdog(self):
        """Start a detached reaper that cleans recorded prover groups if this process dies."""
        try:
            reaper = Path(__file__).with_name("reaper.py")
            st = processes._read_stat(os.getpid())
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
        """Remove recorded prover groups that are no longer owned by a live worker JVM."""
        live = set()
        for w in self.workers:
            if w.jvm_pid is not None and processes._read_stat(w.jvm_pid) is not None:
                groups = processes._prover_groups(w.jvm_pid, os.getpgrp())
                live.update(groups.keys())
                new = set(groups) - set(w.reap_targets)
                w.reap_targets.update(groups)
                if new:
                    w._persist_reap()
        for w in self.workers:
            for pgid, start in list(w.reap_targets.items()):
                if pgid not in live and processes._kill_pgid(pgid, start):
                    _POOL_LOG.debug("monitor reaped orphan prover group %d", pgid)
                    w.reap_targets.pop(pgid, None)

    def _reap_runaway_polys_once(self):
        """Remove abandoned busy Poly/ML processes from oversized worker process trees.

        A healthy worker may legitimately keep one process busy, so CPU use alone is insufficient. Cleanup applies only when the tree has more than two Poly/ML processes, and the two newest processes are always preserved."""
        interval = self.MONITOR_INTERVAL_S
        denom = interval * self._clk_tck
        seen: dict[int, tuple] = {}
        runaways: list[int] = []
        for w in self.workers:
            if w.jvm_pid is None or processes._read_stat(w.jvm_pid) is None:
                continue
            tree: list[tuple] = []   # (pid, start, streak)
            for pid in processes._descendants(w.jvm_pid):
                ps = processes._poly_cpu_state(pid)
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

    def _has_repeated_timeout(self, theorem_code: str) -> bool:
        """Return whether the theorem has reached the timeout limit within the history interval."""
        now = time.time()
        with self._cache_lock:
            rec = self._timeout_history.get(theorem_code)
            if rec is None:
                return False
            if now - rec[1] > self.TIMEOUT_HISTORY_TTL:
                self._timeout_history.pop(theorem_code, None)
                return False
            return rec[0] >= 2

    def _record_timeout(self, theorem_code: str):
        now = time.time()
        with self._cache_lock:
            rec = self._timeout_history.get(theorem_code)
            if rec is None or now - rec[1] > self.TIMEOUT_HISTORY_TTL:
                self._timeout_history[theorem_code] = [1, now]
            else:
                rec[0] += 1
                rec[1] = now
                self._timeout_history.move_to_end(theorem_code)
            while len(self._timeout_history) > self.TIMEOUT_HISTORY_MAX:
                self._timeout_history.popitem(last=False)

    @staticmethod
    def _incomplete_result(reason: str):
        """Return the fail-closed result used after repeated timeouts."""
        return state_classes.VerificationOutcome(
            outcome=state_classes.ProofOutcome.TIMEOUT,
            premise_consistency_unknown=True, errors=[reason])

    def check(self, theorem_code: str) -> "state_classes.VerificationOutcome":
        while True:
            if self._has_repeated_timeout(theorem_code):
                with self._cache_lock:
                    self.premise_consistency_unknown_short_circuits += 1
                return self._incomplete_result(
                    "premise consistency unknown: two prior timeouts")
            with self._cache_lock:
                cached = self._cache.get(theorem_code)
                if cached is not None:
                    self._cache.move_to_end(theorem_code)
                    self.cache_hits += 1
                    out = state_classes.VerificationOutcome.from_raw(cached)
                    out.cache_hit = True
                    out.queue_wait = 0.0
                    out.check_time = 0.0
                    return out
                event = self._pending_checks.get(theorem_code)
                owner = event is None
                if owner:
                    event = threading.Event()
                    self._pending_checks[theorem_code] = event
                    self.cache_misses += 1
                    break
            if not event.wait(self.verify_timeout + 30.0):
                with self._cache_lock:
                    if self._pending_checks.get(theorem_code) is event:
                        self._pending_checks.pop(theorem_code, None)
        cache_file_lock, owns_cache_file_lock = None, False
        try:
            disk = theorem_cache._thm_disk_load(theorem_code, self._thm_fprint)
            if disk is not None and "success" in disk:
                with self._cache_lock:
                    self._cache[theorem_code] = dict(disk)
                    while len(self._cache) > self.CACHE_MAX_ENTRIES:
                        self._cache.popitem(last=False)
                    self.cache_hits += 1
                out = state_classes.VerificationOutcome.from_raw(disk)
                out.cache_hit = True
                out.queue_wait = 0.0
                out.check_time = 0.0
                return out
            if theorem_cache._thm_disk_enabled():
                cache_file_lock = theorem_cache._thm_disk_path(theorem_code, self._thm_fprint) + ".lock"
                owns_cache_file_lock = cache_lock.acquire(
                    cache_file_lock, stale_s=3 * self.verify_timeout
                )
                if not owns_cache_file_lock:
                    got = cache_lock.wait(
                        cache_file_lock,
                        lambda: theorem_cache._thm_disk_load(theorem_code, self._thm_fprint),
                        deadline_s=self.verify_timeout + 45.0,
                        poll_s=0.25,
                    )
                    if got is not None and "success" in got:
                        with self._cache_lock:
                            self._cache[theorem_code] = dict(got)
                            while len(self._cache) > self.CACHE_MAX_ENTRIES:
                                self._cache.popitem(last=False)
                            self.cache_hits += 1
                        out = state_classes.VerificationOutcome.from_raw(got)
                        out.cache_hit = True
                        out.queue_wait = 0.0
                        out.check_time = 0.0
                        return out
            result = state_classes.VerificationOutcome.from_raw(
                self._check_uncached(theorem_code))
            if result.counts_toward_repeated_timeout:
                self._record_timeout(theorem_code)
            cacheable = result.cacheable(self.CACHE_FAIL_FAST_S)
            if cacheable:
                entry = result.to_cache_entry()
                with self._cache_lock:
                    self._cache[theorem_code] = entry
                    while len(self._cache) > self.CACHE_MAX_ENTRIES:
                        self._cache.popitem(last=False)
                theorem_cache._thm_disk_store(theorem_code, entry, self._thm_fprint)
            elif owns_cache_file_lock:
                cache_lock.mark_failed(cache_file_lock)
            return result
        finally:
            if owns_cache_file_lock:
                cache_lock.release(cache_file_lock)
            with self._cache_lock:
                if self._pending_checks.get(theorem_code) is event:
                    self._pending_checks.pop(theorem_code, None)
            event.set()

    def submit(self, theorem_code: str):
        """Queue a theorem check and return a `Future`.

        All callers share one FIFO. Memory-cache hits resolve immediately without entering the queue."""
        from concurrent.futures import Future
        fut = Future()
        with self._cache_lock:
            cached = self._cache.get(theorem_code)
            if cached is not None:
                self._cache.move_to_end(theorem_code)
                self.cache_hits += 1
                out = state_classes.VerificationOutcome.from_raw(cached)
                out.cache_hit = True
                out.queue_wait = 0.0
                out.check_time = 0.0
                fut.set_result(out)
                return fut
        self._ensure_dispatchers()
        self._reqq.put((theorem_code, fut, time.time()))
        return fut

    def _ensure_dispatchers(self):
        if self._dispatchers:
            return
        with self._disp_lock:
            if self._dispatchers:
                return
            for i in range(max(2 * self.num_workers, self.num_workers + 2)):
                t = threading.Thread(target=self._dispatch_loop, daemon=True,
                                     name=f"isa-dispatch-{i}")
                t.start()
                self._dispatchers.append(t)

    def _dispatch_loop(self):
        while not self._stop_monitor.is_set():
            try:
                thm, fut, t_enq = self._reqq.get(timeout=1.0)
            except Empty:
                continue
            t0 = time.time()
            try:
                r = self.check(thm)
                r.queue_wait += t0 - t_enq
                fut.set_result(r)
            except BaseException as e:  # noqa: BLE001 -- deliver, don't die
                fut.set_exception(e)

    def _book_restart(self, worker, reason: str):
        """Record a worker restart and periodically report accumulated reasons."""
        self.restart_count += 1
        self.restart_reasons[reason] = self.restart_reasons.get(reason, 0) + 1
        n = self.worker_restart_counts.get(worker.wid, 0) + 1
        self.worker_restart_counts[worker.wid] = n
        if n % self.WORKER_RESTART_WARN == 0:
            _POOL_LOG.warning("worker %d restarted %d times (reasons=%s)",
                              worker.wid, n, self.restart_reasons)
        else:
            _POOL_LOG.debug("worker %d restart (%s)", worker.wid, reason)

    def _restart_or_mark_unavailable(self, worker) -> bool:
        """Restart a stopped worker or keep it out of the idle queue until a background retry succeeds."""
        try:
            worker.start()
            return True
        except Exception as error:  # noqa: BLE001
            _POOL_LOG.warning(
                "worker %d restart failed; temporarily unavailable: %r",
                worker.wid,
                error,
            )
            worker._unavailable = True

            def _rebuild():
                delay = 5.0
                while not self._stop_monitor.is_set():
                    time.sleep(delay)
                    try:
                        worker.stop(graceful=False)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        worker.start()
                        worker._unavailable = False
                        self.idle.put(worker)
                        _POOL_LOG.warning("worker %d restored", worker.wid)
                        return
                    except Exception as restart_error:  # noqa: BLE001
                        _POOL_LOG.warning(
                            "worker %d restart retry failed: %r",
                            worker.wid,
                            restart_error,
                        )
                        delay = min(delay * 2, 60.0)

            threading.Thread(target=_rebuild, daemon=True).start()
            return False

    def _run_one_check(self, worker, theorem_code: str) -> "state_classes.VerificationOutcome":
        """Run one worker check behind a hard wall-clock limit.

        A wedged worker is stopped and rebuilt before it can return to the idle queue. Process-tree memory is checked before the worker is released for another request."""
        box: dict = {}

        def _run():
            try:
                box["r"] = state_classes.VerificationOutcome.from_raw(
                    worker.check(theorem_code))
            except Exception as e:  # noqa: BLE001 -- any failure -> restart
                box["e"] = e

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(self.verify_timeout + 15.0)
        try:
            if th.is_alive():
                self._book_restart(worker, "hard_timeout")
                _POOL_LOG.warning(
                    "hard timeout on worker %d, theorem head: %r",
                    worker.wid, theorem_code[:300])
                try:
                    worker.stop(graceful=False)  # fast kill + killpg reap
                except Exception:
                    pass
                th.join(10.0)
                self._restart_or_mark_unavailable(worker)
                return state_classes.VerificationOutcome(
                    outcome=state_classes.ProofOutcome.TIMEOUT,
                    elapsed=self.verify_timeout + 15.0,
                    errors=["hard timeout: worker force-restarted"])
            if "e" in box:
                e = box["e"]
                self._book_restart(worker, type(e).__name__)
                if isinstance(e, TimeoutError):
                    self._timeout_seen = getattr(self, "_timeout_seen", 0) + 1
                    if self._timeout_seen % 20 == 1:
                        _POOL_LOG.warning(
                            "verify TimeoutError #%d w%d goal=%r tac=%r",
                            self._timeout_seen, worker.wid,
                            theorem_code[:220], theorem_code[-90:])
                try:
                    worker.stop(graceful=False)  # fast kill + killpg reap
                except Exception:
                    pass
                self._restart_or_mark_unavailable(worker)
                return state_classes.VerificationOutcome(
                    outcome=(state_classes.ProofOutcome.TIMEOUT
                             if isinstance(e, TimeoutError)
                             else state_classes.ProofOutcome.WORKER_ERROR),
                    elapsed=0.0, errors=[f"worker error: {e!r}"])
            return box["r"]
        finally:
            try:
                worker._rss_tick = getattr(worker, "_rss_tick", 0) + 1
                r = box.get("r")
                slow = (isinstance(r, state_classes.VerificationOutcome)
                        and r.elapsed > 5.0)
                if ((worker._rss_tick % 5 == 0 or slow)
                        and worker.jvm_pid is not None
                        and processes._poly_tree_rss_kb(worker.jvm_pid)
                        > self.EACH_WORKER_PROC_TREE_MEM_MAX_KB):
                    self._book_restart(worker, "rss_cap")
                    worker.stop(graceful=False)
                    self._restart_or_mark_unavailable(worker)
            except Exception as e:  # noqa: BLE001 -- recycle must never crash
                _POOL_LOG.debug("rss recycle failed w%d: %r", worker.wid, e)
            if not getattr(worker, "_unavailable", False):
                self.idle.put(worker)

    def _check_uncached(self, theorem_code: str) -> "state_classes.VerificationOutcome":
        queue_wait = 0.0
        run_s = 0.0
        result = None
        for attempt in range(self.MAX_CHECK_ATTEMPTS):
            t0 = time.time()
            worker = self.idle.get()
            queue_wait += time.time() - t0
            t1 = time.time()
            result = self._run_one_check(worker, theorem_code)
            run_s += time.time() - t1
            result = state_classes.VerificationOutcome.from_raw(result)
            result.worker = worker.wid
            result.attempts = attempt + 1
            if not result.infrastructure_failure:
                break
        result.check_time = run_s      # sum of in-worker attempts
        result.queue_wait = queue_wait
        return result

    def shutdown(self):
        self._stop_monitor.set()
        for w in self.workers:
            w.stop()
        if self._watchdog is not None:
            try:
                self._watchdog.terminate()
            except OSError:
                pass
            self._watchdog = None
        for reap in self.base_dir.glob("worker_*.reap"):
            try:
                reap.unlink()
            except OSError:
                pass
