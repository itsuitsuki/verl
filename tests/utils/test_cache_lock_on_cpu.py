"""CPU-only tests for cross-process disk-cache coordination.

The tests cover cache_lock primitives and their use by the translation and theorem caches. Another process is simulated by pre-creating lock and failure files and clearing in-process state.
"""
import os
import threading
import time

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip(
        "server_pool is Linux-only (os.sysconf at module level)",
        allow_module_level=True,
    )

from verl.utils.isabelle_utils import cache_lock
from verl.utils.isabelle_utils import server_pool as sp
from verl.utils.isabelle_utils._server_pool import theorem_cache
from verl.utils.isabelle_utils import translator

THM = 'theorem chk:\n  shows "(2::nat) + 2 = 4"\n  using assms by (simp)'
# The pool keys the theorem cache by the normalized theorem text, so a test that
# simulates another process must build the same disk lock/store path from THM_KEY.
THM_KEY = theorem_cache.normalize_theorem_text(THM)


# ---------- primitives ----------

def test_acquire_release_cycle(tmp_path):
    lock = str(tmp_path / "e" / "k.lock")
    assert cache_lock.acquire(lock, stale_s=60.0)
    assert not cache_lock.acquire(lock, stale_s=60.0)
    cache_lock.release(lock)
    assert cache_lock.acquire(lock, stale_s=60.0)


def test_stale_lock_is_removed(tmp_path):
    lock = str(tmp_path / "k.lock")
    assert cache_lock.acquire(lock, stale_s=60.0)
    os.utime(lock, (time.time() - 120, time.time() - 120))
    assert cache_lock.acquire(lock, stale_s=60.0)


def test_failure_marker_roundtrip(tmp_path):
    lock = str(tmp_path / "k.lock")
    cache_lock.mark_failed(lock)
    assert cache_lock.failed_recently(lock)
    # A waiting process stops immediately after observing a failure marker.
    t0 = time.time()
    assert cache_lock.wait(
        lock, lambda: None, deadline_s=30.0, poll_s=0.1
    ) is None
    assert time.time() - t0 < 5.0
    # A new owner supersedes the old failure.
    assert cache_lock.acquire(lock, stale_s=60.0)
    assert not cache_lock.failed_recently(lock)


def test_wait_returns_other_process_result(tmp_path):
    lock = str(tmp_path / "k.lock")
    assert cache_lock.acquire(lock, stale_s=60.0)
    box = {}

    def _store_result():
        time.sleep(0.3)
        box["v"] = 42
        cache_lock.release(lock)

    threading.Thread(target=_store_result).start()
    got = cache_lock.wait(
        lock, lambda: box.get("v"), deadline_s=10.0, poll_s=0.1
    )
    assert got == 42


# ---------- translation cache integration ----------

def _reset_translator():
    translator._TR_CACHE.clear()
    translator._TR_PENDING.clear()
    translator._TR_STATS.update(hits=0, misses=0)


def _ok_parse(reply):
    return reply


def _ok_validate(_parsed):
    return []


def test_translation_after_other_process_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset_translator()
    calls = {"n": 0}

    def fake(prompt, thinking, **kwargs):
        calls["n"] += 1
        return "T:" + prompt

    monkeypatch.setattr(translator, "call_judge", fake)
    key = translator._tr_key("P", "m", 12288, _ok_parse, _ok_validate, None)
    lock = translator._tr_disk_path(key) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w") as file:
        file.write("99999")
    cache_lock.mark_failed(lock)
    result = translator.translate(
        "P", _ok_parse, _ok_validate, translator_url="u", translator_model="m"
    )
    assert result[0] == "T:P" and calls["n"] == 1


def test_translation_waits_for_other_process_result(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset_translator()
    calls = {"n": 0}

    def fake(prompt, thinking, **kwargs):
        calls["n"] += 1
        return "T:" + prompt

    monkeypatch.setattr(translator, "call_judge", fake)
    key = translator._tr_key("Q", "m", 12288, _ok_parse, _ok_validate, None)
    lock = translator._tr_disk_path(key) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w") as file:
        file.write("99999")

    def _store_result():
        time.sleep(0.3)
        translator._tr_disk_store(key, "FOREIGN", False)
        cache_lock.release(lock)

    threading.Thread(target=_store_result).start()
    result = translator.translate(
        "Q", _ok_parse, _ok_validate, translator_url="u", translator_model="m"
    )
    assert result[0] == "FOREIGN" and calls["n"] == 0
    assert result[1] and result[1][0].get("cache") in ("xproc", "disk")


# ---------- theorem cache integration ----------

def _pool(tmp_path, monkeypatch, cache_dir):
    monkeypatch.setenv("ISABELLE_THEOREM_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("ISABELLE_THEOREM_DISK_CACHE", "1")
    return sp.IsabelleServerPool(num_workers=1, base_dir=str(tmp_path))


def test_theorem_after_other_process_failure(tmp_path, monkeypatch):
    pool = _pool(tmp_path / "a", monkeypatch, tmp_path / "thm")
    calls = {"n": 0}

    def fake_uncached(code):
        calls["n"] += 1
        return {"success": True, "elapsed": 0.01, "errors": []}

    monkeypatch.setattr(pool, "_check_uncached", fake_uncached)
    lock = theorem_cache._thm_disk_path(THM_KEY, pool._thm_fprint) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w") as file:
        file.write("99999")
    cache_lock.mark_failed(lock)
    result = pool.check(THM)
    assert result.proved and calls["n"] == 1


def test_theorem_waits_for_other_process_result(tmp_path, monkeypatch):
    pool = _pool(tmp_path / "a", monkeypatch, tmp_path / "thm")
    calls = {"n": 0}

    def fake_uncached(code):
        calls["n"] += 1
        return {"success": True, "elapsed": 0.01, "errors": []}

    monkeypatch.setattr(pool, "_check_uncached", fake_uncached)
    lock = theorem_cache._thm_disk_path(THM_KEY, pool._thm_fprint) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w") as file:
        file.write("99999")

    def _store_result():
        time.sleep(0.3)
        theorem_cache._thm_disk_store(
            THM_KEY,
            {"success": False, "elapsed": 0.5, "errors": ["by fails"]},
            pool._thm_fprint,
        )
        cache_lock.release(lock)

    threading.Thread(target=_store_result).start()
    result = pool.check(THM)
    assert result.cache_hit and not result.proved
    assert calls["n"] == 0
