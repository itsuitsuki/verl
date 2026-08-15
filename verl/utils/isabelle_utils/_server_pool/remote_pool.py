"""HTTP client presenting the IsabelleServerPool interface for a pool hosted on another machine.

The training node talks to a CPU-only server (verl.utils.isabelle_utils.remote_server) that owns the real
IsabelleServerPool; every theorem check crosses the network as one POST /check. Caching, retry, timeout
short-circuits, and process reaping all stay server-side, so this client is deliberately thin: no memoization,
no worker management, no prover processes on the training node.
"""
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import requests

from verl.utils.isabelle_utils import state_classes

_REMOTE_LOG = logging.getLogger("verl.isabelle.remote_pool")


def outcome_to_wire(outcome: "state_classes.VerificationOutcome") -> dict:
    """JSON-safe dict carrying every VerificationOutcome field, diagnostic ones included.

    The wire format transports the outcome enum value explicitly, so deserialization never re-derives the
    classification from error-text heuristics (VerificationOutcome.from_raw's raw-dict inference exists for
    3-field disk cache entries, which cannot distinguish TIMEOUT from WORKER_ERROR without those heuristics).
    """
    return {
        "outcome": outcome.outcome.value,
        "elapsed": float(outcome.elapsed),
        "errors": [str(e) for e in outcome.errors],
        "premise_consistency_unknown": bool(outcome.premise_consistency_unknown),
        "cache_hit": bool(outcome.cache_hit),
        "queue_wait": float(outcome.queue_wait),
        "check_time": float(outcome.check_time),
        "worker": outcome.worker,
        "attempts": outcome.attempts,
    }


def outcome_from_wire(payload: dict) -> "state_classes.VerificationOutcome":
    """Rebuild the typed VerificationOutcome from its wire dict (exact inverse of outcome_to_wire)."""
    return state_classes.VerificationOutcome(
        outcome=state_classes.ProofOutcome(payload["outcome"]),
        elapsed=float(payload.get("elapsed") or 0.0),
        errors=list(payload.get("errors") or []),
        premise_consistency_unknown=bool(payload.get("premise_consistency_unknown")),
        cache_hit=bool(payload.get("cache_hit")),
        queue_wait=float(payload.get("queue_wait") or 0.0),
        check_time=float(payload.get("check_time") or 0.0),
        worker=payload.get("worker"),
        attempts=payload.get("attempts"),
    )


class RemoteIsabellePool:
    """Client-side stand-in for IsabelleServerPool: the same check/submit/shutdown surface, HTTP underneath.

    Any transport failure (connection refused, HTTP error status, undecodable payload) fails closed as an
    infrastructure outcome instead of raising: TIMEOUT for a request timeout, WORKER_ERROR for everything else.
    That mirrors how the local pool converts worker crashes into WORKER_ERROR results, so downstream reward
    logic treats the step as unverified and never rewards it.
    """

    # Above the dispatcher-thread count the pool sizing rule (2 * workers) yields for typical worker counts.
    _HTTP_POOL_MAXSIZE = 64

    def __init__(self, base_url: str, timeout: float = 300.0, health_timeout: float = 10.0):
        """`timeout` bounds one whole remote check: server-side idle-worker wait plus up to MAX_CHECK_ATTEMPTS
        prover runs, so it must sit well above the pool's per-run verify_timeout."""
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        # Engine code sizes per-response check parallelism from this; start() replaces it with the server's real count.
        self.num_workers = 4
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=self._HTTP_POOL_MAXSIZE)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._stats_cache: tuple[float, dict] = (0.0, {})

    def start(self):
        """Probe /health so a wrong URL or an unstarted server fails loudly at engine construction, matching the local pool's start() behavior on a dead worker."""
        try:
            payload = self._fetch_health()
        except Exception as error:
            raise RuntimeError(
                f"remote Isabelle pool at {self.base_url} is unreachable or unhealthy: {error!r}") from error
        if payload.get("status") != "ok":
            raise RuntimeError(f"remote Isabelle pool at {self.base_url} reported {payload!r}")
        self.num_workers = int(payload.get("num_workers") or self.num_workers)
        print(f"[remote-pool] connected to {self.base_url}: {self.num_workers} server workers", flush=True)

    def _fetch_health(self) -> dict:
        response = self._session.get(f"{self.base_url}/health", timeout=self.health_timeout)
        response.raise_for_status()
        return response.json()

    def check(self, theorem_code: str) -> "state_classes.VerificationOutcome":
        t0 = time.time()
        try:
            response = self._session.post(
                f"{self.base_url}/check", json={"theorem": theorem_code}, timeout=self.timeout)
            response.raise_for_status()
            return outcome_from_wire(response.json())
        except requests.exceptions.Timeout:
            return state_classes.VerificationOutcome(
                outcome=state_classes.ProofOutcome.TIMEOUT,
                elapsed=time.time() - t0,
                errors=[f"remote pool request timed out after {self.timeout:.0f}s"])
        except Exception as error:  # noqa: BLE001 -- any transport/payload failure fails closed
            _REMOTE_LOG.warning("remote check failed: %r", error)
            return state_classes.VerificationOutcome(
                outcome=state_classes.ProofOutcome.WORKER_ERROR,
                elapsed=time.time() - t0,
                errors=[f"remote pool error: {error!r}"])

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    # Mirror the local pool's dispatcher count so client-side concurrency tracks server capacity.
                    self._executor = ThreadPoolExecutor(
                        max_workers=max(2 * self.num_workers, self.num_workers + 2),
                        thread_name_prefix="isa-remote")
        return self._executor

    def submit(self, theorem_code: str) -> Future:
        """Run check() on a client-side executor; time spent waiting for an executor thread joins queue_wait, like enqueue-to-dispatch time does in the local pool's FIFO."""
        t_enqueue = time.time()

        def _run():
            t0 = time.time()
            result = self.check(theorem_code)
            result.queue_wait += t0 - t_enqueue
            return result

        return self._ensure_executor().submit(_run)

    # Gauge counters read by the reward debug metrics (formal_verify.py). They report SERVER-wide totals,
    # shared across every training process using this server; a stale or unreachable server yields zeros,
    # and the caller's gauge block already tolerates that.
    def _stat(self, name: str) -> int:
        cached_at, payload = self._stats_cache
        if time.time() - cached_at > 5.0:
            try:
                payload = self._fetch_health()
            except Exception:  # noqa: BLE001 -- gauges must never fail scoring
                payload = {}
            self._stats_cache = (time.time(), payload)
        return int(payload.get(name) or 0)

    @property
    def restart_count(self) -> int:
        return self._stat("restart_count")

    @property
    def external_solver_reaps(self) -> int:
        return self._stat("external_solver_reaps")

    @property
    def cache_hits(self) -> int:
        return self._stat("cache_hits")

    @property
    def cache_misses(self) -> int:
        return self._stat("cache_misses")

    def shutdown(self):
        """Release client-side resources only; the server owns the workers and manages its own lifecycle."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._session.close()
