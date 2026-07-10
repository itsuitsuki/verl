"""CPU-only unit tests for IsabelleServerPool's check-retry / hard-timeout
recovery. These mock ``_run_one_check`` so NO Isabelle server is started -- they
exercise only the control flow of ``_check_uncached``:

  * a transient ``worker_error`` (hard-timeout or worker crash) is retried on a
    fresh worker, up to ``MAX_CHECK_ATTEMPTS``, so the verification is recovered
    instead of lost to a fail-closed ``x``;
  * a persistent ``worker_error`` exhausts the attempts and returns fail-closed;
  * a genuine proof failure (no ``worker_error``) is returned immediately and is
    never retried.

Regression guard for the 2026-07-08 step-135 Isabelle-pool deadlock fix
(single-flight leader wedged -> followers waited forever). See
verl/utils/isabelle_utils/server_pool.py.
"""

import queue

from verl.utils.isabelle_utils import server_pool as sp

WORKER_ERROR = {"success": False, "worker_error": True, "errors": ["hard timeout"]}
OK = {"success": True, "elapsed": 1.0, "errors": []}
GENUINE_FAIL = {"success": False, "errors": ["not a theorem"]}  # no worker_error


class _FakeWorker:
    def __init__(self, wid):
        self.wid = wid


def _pool_with_mock(seq):
    """Pool whose ``_run_one_check`` returns ``seq[i]`` on the i-th call (clamped
    to the last entry). No ``start()`` -> no Isabelle. Returns (pool, counter)."""
    pool = sp.IsabelleServerPool(num_workers=2, base_dir="/tmp/retry_unit_test")
    pool.idle = queue.Queue()
    for wid in range(2):
        pool.idle.put(_FakeWorker(wid))
    calls = {"n": 0}

    def mock(worker, code):
        i = calls["n"]
        calls["n"] += 1
        pool.idle.put(worker)  # the real _run_one_check always re-idles the worker
        return dict(seq[min(i, len(seq) - 1)])

    pool._run_one_check = mock
    return pool, calls


def test_retry_recovers_transient_wedge():
    pool, calls = _pool_with_mock([WORKER_ERROR, OK])
    r = pool._check_uncached("dummy theorem")
    assert r["success"] is True
    assert r["attempts"] == 2
    assert calls["n"] == 2


def test_persistent_wedge_is_fail_closed():
    pool, calls = _pool_with_mock([WORKER_ERROR])
    r = pool._check_uncached("dummy theorem")
    assert r["success"] is False
    assert r["worker_error"] is True
    assert r["attempts"] == sp.IsabelleServerPool.MAX_CHECK_ATTEMPTS
    assert calls["n"] == sp.IsabelleServerPool.MAX_CHECK_ATTEMPTS


def test_genuine_proof_failure_not_retried():
    pool, calls = _pool_with_mock([GENUINE_FAIL, OK])
    r = pool._check_uncached("dummy theorem")
    assert r["success"] is False
    assert r["attempts"] == 1
    assert calls["n"] == 1


# ---- per-check RSS ceiling (host-memory bound; 2026-07-08 OOM root cause) ----

class _FakeProc:
    def __init__(self, pid):
        self.pid = pid


class _RecordingWorker:
    """Worker whose check() succeeds instantly; records stop()/start() so a
    test can assert whether _run_one_check recycled it. Mirrors the real
    IsabelleWorker contract: jvm_pid is set (start_server resolves the real
    java pid) and refreshed on restart."""
    def __init__(self, wid=0, pid=999000, elapsed=1.0):
        self.wid = wid
        self.proc = _FakeProc(pid)
        self.jvm_pid = pid
        self.elapsed = elapsed
        self.events = []

    def check(self, code):
        return {"success": True, "elapsed": self.elapsed, "errors": []}

    def stop(self, graceful=True):
        self.events.append(("stop", graceful))

    def start(self):
        self.events.append(("start",))
        self.proc = _FakeProc(self.proc.pid + 1)  # fresh JVM after recycle
        self.jvm_pid = self.proc.pid


def _bare_pool(base):
    import queue
    pool = sp.IsabelleServerPool(num_workers=1, base_dir=base)
    pool.idle = queue.Queue()
    return pool


def test_recycle_slow_check_scans_immediately(monkeypatch):
    # The RSS scan is gated (every 5th check OR after a slow one); a SLOW
    # check (>5s) scans right away, so a leak born of a heavy proof is caught
    # on the very next turnaround.
    pool = _bare_pool("/tmp/rss_unit_over")
    monkeypatch.setattr(sp, "_poly_tree_rss_kb",
                        lambda pid: pool.WORKER_RSS_CAP_KB + 1)
    w = _RecordingWorker(elapsed=6.0)
    r = pool._run_one_check(w, "thm")
    assert r["success"] is True            # result still delivered to caller
    assert ("stop", False) in w.events     # recycled with fast-kill (killpg)
    assert ("start",) in w.events          # JVM rebuilt -> all poly reset
    assert pool.idle.qsize() == 1          # returned to the pool afterwards


def test_recycle_fast_checks_scan_every_fifth(monkeypatch):
    # Fast checks skip the full-/proc scan until the 5th turnaround.
    pool = _bare_pool("/tmp/rss_unit_gate")
    monkeypatch.setattr(sp, "_poly_tree_rss_kb",
                        lambda pid: pool.WORKER_RSS_CAP_KB + 1)
    w = _RecordingWorker(elapsed=1.0)
    for i in range(4):
        pool._run_one_check(w, "thm")
        pool.idle.get()                    # drain for the next round
        assert w.events == [], f"scanned too early at check {i + 1}"
    pool._run_one_check(w, "thm")          # 5th check -> scan fires
    assert ("stop", False) in w.events and ("start",) in w.events


def test_no_recycle_when_poly_rss_under_cap(monkeypatch):
    pool = _bare_pool("/tmp/rss_unit_under")
    monkeypatch.setattr(sp, "_poly_tree_rss_kb", lambda pid: 1024)  # 1 MB
    w = _RecordingWorker(elapsed=6.0)      # slow -> scans every time
    for _ in range(6):
        r = pool._run_one_check(w, "thm")
        pool.idle.get()
        assert r["success"] is True
    assert w.events == []                  # healthy worker NOT recycled
