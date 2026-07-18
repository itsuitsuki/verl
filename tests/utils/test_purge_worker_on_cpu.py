"""CPU-only tests for the bounded purge pipeline (server_pool.py, 2026-07-11
review #3). The old design spawned a NEW daemon thread every PURGE_EVERY
checks; a stalled purge_theories accumulated threads without bound. Now each
worker owns ONE long-lived purge thread + a bounded queue; overflow merges the
oldest batch into the newest (names are never dropped). No Isabelle needed:
_purge_async is monkeypatched."""
import os
import threading
import time

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf at module level)",
                allow_module_level=True)

from verl.utils.isabelle_utils._server_pool.worker import IsabelleWorker


def _worker(tmp_path):
    return IsabelleWorker(0, tmp_path)


def test_single_purge_thread_processes_batches(tmp_path):
    w = _worker(tmp_path)
    seen, done = [], threading.Event()

    def fake_purge(names):
        seen.append(list(names))
        if len(seen) == 3:
            done.set()

    w._purge_async = fake_purge
    for i in range(3):
        w._enqueue_purge([f"V{i}"])
    assert done.wait(5.0)
    assert seen == [["V0"], ["V1"], ["V2"]]
    # exactly one purge thread, alive and reusable
    assert w._purge_thread.is_alive()


def test_overflow_merges_instead_of_dropping(tmp_path):
    w = _worker(tmp_path)
    release = threading.Event()
    seen = []

    def slow_purge(names):
        release.wait(10.0)
        seen.append(list(names))

    w._purge_async = slow_purge
    w._enqueue_purge(["HEAD"])          # occupies the purge thread
    time.sleep(0.2)                      # let the thread pick HEAD up
    for i in range(12):                  # maxsize is 8 -> forces merges
        w._enqueue_purge([f"V{i}"])
    release.set()
    deadline = time.time() + 5.0
    while time.time() < deadline and sum(len(b) for b in seen) < 13:
        time.sleep(0.05)
    flat = [n for b in seen for n in b]
    # every name survived (merged, not dropped), each exactly once
    assert sorted(flat) == sorted(["HEAD"] + [f"V{i}" for i in range(12)])


def test_purge_loop_survives_exceptions(tmp_path):
    w = _worker(tmp_path)
    calls, ok = [], threading.Event()

    def flaky(names):
        calls.append(list(names))
        if len(calls) == 1:
            raise RuntimeError("mgmt_conn is None")  # e.g. mid-restart
        ok.set()

    w._purge_async = flaky
    w._enqueue_purge(["BOOM"])
    w._enqueue_purge(["NEXT"])
    assert ok.wait(5.0)                  # thread survived the first failure
    assert w._purge_thread.is_alive()
