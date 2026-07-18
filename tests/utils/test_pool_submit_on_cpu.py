"""CPU-only tests for the unified Isabelle check scheduler.

pool.submit() places requests in one process-wide FIFO served by dispatcher threads. `_check_uncached` is mocked, so no Isabelle process is needed. The tests cover result delivery, memory-cache hits, queue-wait measurement, exception delivery, and concurrent identical-request deduplication.
"""
import os
import threading

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf at module level)",
                allow_module_level=True)

from verl.utils.isabelle_utils import server_pool as sp

THM = 'theorem chk:\n  shows "(2::nat) + 2 = 4"\n  using assms by (simp)'


def _pool(tmp_path, monkeypatch, num_workers=2):
    monkeypatch.setenv("ISABELLE_THEOREM_DISK_CACHE", "0")  # isolate memo
    p = sp.IsabelleServerPool(num_workers=num_workers, base_dir=str(tmp_path))
    return p


def test_submit_resolves_all_and_injects_queue_wait(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    calls = []

    def fake_uncached(code):
        calls.append(code)
        return {"success": True, "elapsed": 0.01, "errors": [],
                "queue_wait": 0.0, "check_time": 0.01}

    monkeypatch.setattr(p, "_check_uncached", fake_uncached)
    futs = [p.submit(THM.replace("4", str(4 + i))) for i in range(6)]
    outs = [f.result(timeout=10.0) for f in futs]
    assert all(o.proved for o in outs)
    assert len(calls) == 6                     # each distinct theorem proved
    assert all(o.queue_wait >= 0.0 for o in outs)


def test_fifo_delay_lands_in_queue_wait(tmp_path, monkeypatch):
    # This assertion must detect whether FIFO delay is added to queue_wait. Use one dispatcher with a slow check so later submissions remain in the request queue long enough to measure.
    import time as _time

    p = _pool(tmp_path, monkeypatch, num_workers=1)
    monkeypatch.setattr(p, "_ensure_dispatchers", lambda: None)

    def slow_uncached(code):
        _time.sleep(0.2)
        return {"success": True, "elapsed": 0.2, "errors": [],
                "queue_wait": 0.0, "check_time": 0.2}

    monkeypatch.setattr(p, "_check_uncached", slow_uncached)
    futs = [p.submit(THM.replace("4", str(40 + i))) for i in range(3)]
    t = threading.Thread(target=p._dispatch_loop, daemon=True)  # one lane
    t.start()
    outs = [f.result(timeout=15.0) for f in futs]
    # 3rd request sat behind two 0.2s checks in the FIFO
    assert outs[2].queue_wait >= 0.3


def test_submit_memo_fast_path(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    n = {"c": 0}

    def fake_uncached(code):
        n["c"] += 1
        return {"success": True, "elapsed": 0.01, "errors": []}

    monkeypatch.setattr(p, "_check_uncached", fake_uncached)
    assert p.submit(THM).result(timeout=10.0).proved
    r2 = p.submit(THM).result(timeout=10.0)
    assert r2.proved and r2.cache_hit and n["c"] == 1
    assert r2.queue_wait == 0.0                # memo hit bypasses the FIFO


def test_submit_delivers_exceptions(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)

    def boom(code):
        raise RuntimeError("prover exploded")

    monkeypatch.setattr(p, "_check_uncached", boom)
    with pytest.raises(RuntimeError, match="prover exploded"):
        p.submit(THM).result(timeout=10.0)


def test_queue_wait_and_check_time_disjoint(tmp_path, monkeypatch):
    # queue_wait must cover only idle-worker and FIFO delay, while check_time covers execution after a worker is obtained. A delayed worker and a separate execution delay verify that these intervals do not overlap.
    import time as _time

    p = _pool(tmp_path, monkeypatch)

    class _W:
        wid = 0

    def fake_run(worker, code):
        _time.sleep(0.1)
        return {"success": True, "elapsed": 0.1, "errors": []}

    monkeypatch.setattr(p, "_run_one_check", fake_run)

    def _feed():
        _time.sleep(0.25)
        p.idle.put(_W())

    threading.Thread(target=_feed).start()
    r = p._check_uncached(THM)
    assert r.queue_wait >= 0.2                    # saw the idle delay
    assert 0.05 <= r.check_time <= 0.2            # run only, no idle time


def test_submit_deduplicates_concurrent_identical_checks(tmp_path, monkeypatch):
    # Sixteen concurrent submissions of one theorem still perform one proof.
    p = _pool(tmp_path, monkeypatch, num_workers=3)
    n = {"c": 0}
    lock = threading.Lock()

    def fake_uncached(code):
        with lock:
            n["c"] += 1
        return {"success": True, "elapsed": 0.01, "errors": []}

    monkeypatch.setattr(p, "_check_uncached", fake_uncached)
    futs = [p.submit(THM) for _ in range(16)]
    outs = [f.result(timeout=15.0) for f in futs]
    assert all(o.proved for o in outs)
    assert n["c"] == 1
