"""CPU-only tests for cross-process single-flight (2026-07-11 review #4):
xlock primitives, plus integration into the translate disk cache (judge.py)
and the theorem disk cache (server_pool.py). "Another process" is simulated
by pre-creating lock/fail-marker files and by clearing in-process state."""
import os
import threading
import time

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf at module level)",
                allow_module_level=True)

from verl.utils.isabelle_utils import xlock
from verl.utils.isabelle_utils import judge
from verl.utils.isabelle_utils import server_pool as sp

THM = 'theorem chk:\n  shows "(2::nat) + 2 = 4"\n  using assms by (simp)'


# ---------- primitives ----------

def test_acquire_release_cycle(tmp_path):
    lock = str(tmp_path / "e" / "k.lock")
    assert xlock.acquire(lock, stale_s=60.0)
    assert not xlock.acquire(lock, stale_s=60.0)   # held
    xlock.release(lock)
    assert xlock.acquire(lock, stale_s=60.0)       # reusable


def test_stale_lock_is_stolen(tmp_path):
    lock = str(tmp_path / "k.lock")
    assert xlock.acquire(lock, stale_s=60.0)
    os.utime(lock, (time.time() - 120, time.time() - 120))
    assert xlock.acquire(lock, stale_s=60.0)       # stolen


def test_fail_marker_roundtrip(tmp_path):
    lock = str(tmp_path / "k.lock")
    xlock.mark_failed(lock)
    assert xlock.failed_recently(lock)
    # waiter stops polling immediately on a fail marker
    t0 = time.time()
    assert xlock.wait(lock, lambda: None, deadline_s=30.0, poll_s=0.1) is None
    assert time.time() - t0 < 5.0
    # a fresh leader supersedes the old failure
    assert xlock.acquire(lock, stale_s=60.0)
    assert not xlock.failed_recently(lock)


def test_wait_returns_leader_result(tmp_path):
    lock = str(tmp_path / "k.lock")
    assert xlock.acquire(lock, stale_s=60.0)
    box = {}

    def _leader():
        time.sleep(0.3)
        box["v"] = 42
        xlock.release(lock)

    threading.Thread(target=_leader).start()
    got = xlock.wait(lock, lambda: box.get("v"), deadline_s=10.0, poll_s=0.1)
    assert got == 42


# ---------- translate cache integration ----------

def _reset_judge():
    judge._TR_CACHE.clear()
    judge._TR_INFLIGHT.clear()
    judge._TR_STATS.update(hits=0, misses=0)


def _ok_parse(r):
    return r


def _ok_validate(_p):
    return []


def test_translate_foreign_fail_marker_falls_back(monkeypatch, tmp_path):
    # another process held the lock and failed -> we translate ourselves
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset_judge()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "T:" + prompt

    monkeypatch.setattr(judge, "call_judge", fake)
    key = judge._tr_key("P", "m", 12288, _ok_parse, _ok_validate, None)
    lock = judge._tr_disk_path(key) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    open(lock, "w").write("99999")
    xlock.mark_failed(lock)
    r = judge.translate("P", _ok_parse, _ok_validate,
                        judge_url="u", judge_model="m")
    assert r[0] == "T:P" and calls["n"] == 1


def test_translate_waits_for_foreign_result(monkeypatch, tmp_path):
    # another process holds the lock and stores its result mid-wait -> we
    # reuse it and never call the judge
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset_judge()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "T:" + prompt

    monkeypatch.setattr(judge, "call_judge", fake)
    key = judge._tr_key("Q", "m", 12288, _ok_parse, _ok_validate, None)
    lock = judge._tr_disk_path(key) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    open(lock, "w").write("99999")

    def _foreign():
        time.sleep(0.3)
        judge._tr_disk_store(key, "FOREIGN", False)
        xlock.release(lock)

    threading.Thread(target=_foreign).start()
    r = judge.translate("Q", _ok_parse, _ok_validate,
                        judge_url="u", judge_model="m")
    assert r[0] == "FOREIGN" and calls["n"] == 0
    assert r[1] and r[1][0].get("cache") in ("xproc", "disk")


# ---------- theorem cache integration ----------

def _pool(tmp_path, monkeypatch, cache_dir):
    monkeypatch.setenv("ISABELLE_THEOREM_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("ISABELLE_THEOREM_DISK_CACHE", "1")
    return sp.IsabelleServerPool(num_workers=1, base_dir=str(tmp_path))


def test_theorem_foreign_fail_marker_falls_back(tmp_path, monkeypatch):
    p = _pool(tmp_path / "a", monkeypatch, tmp_path / "thm")
    calls = {"n": 0}

    def fake_uncached(code):
        calls["n"] += 1
        return {"success": True, "elapsed": 0.01, "errors": []}

    monkeypatch.setattr(p, "_check_uncached", fake_uncached)
    lock = sp._thm_disk_path(THM) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    open(lock, "w").write("99999")
    xlock.mark_failed(lock)
    r = p.check(THM)
    assert r["success"] and calls["n"] == 1


def test_theorem_waits_for_foreign_verdict(tmp_path, monkeypatch):
    p = _pool(tmp_path / "a", monkeypatch, tmp_path / "thm")
    calls = {"n": 0}

    def fake_uncached(code):
        calls["n"] += 1
        return {"success": True, "elapsed": 0.01, "errors": []}

    monkeypatch.setattr(p, "_check_uncached", fake_uncached)
    lock = sp._thm_disk_path(THM) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    open(lock, "w").write("99999")

    def _foreign():
        time.sleep(0.3)
        sp._thm_disk_store(THM, {"success": False, "elapsed": 0.5,
                                 "errors": ["by fails"]})
        xlock.release(lock)

    threading.Thread(target=_foreign).start()
    r = p.check(THM)
    assert r.get("cache_hit") and not r["success"] and calls["n"] == 0
