"""CPU-only unit tests for the theorem-verdict disk cache (server_pool.py).
Mocks _check_uncached so NO Isabelle is needed. Verifies: a cacheable verdict
persists to disk and survives a cold memory cache AND a fresh pool instance
(process-restart simulation); an uncacheable slow failure is NOT persisted;
the kill-switch env disables the disk layer. Linux-only (os.sysconf)."""
import pytest

from verl.utils.isabelle_utils import server_pool as sp

THM = 'theorem chk:\n  shows "(2::nat) + 2 = 4"\n  using assms by (simp)'


def _pool(tmp_path, monkeypatch, result, cache_dir):
    monkeypatch.setenv("ISABELLE_THEOREM_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("ISABELLE_THEOREM_DISK_CACHE", "1")
    p = sp.IsabelleServerPool(num_workers=1, base_dir=str(tmp_path))
    calls = {"n": 0}

    def fake_uncached(code):
        calls["n"] += 1
        return dict(result)

    monkeypatch.setattr(p, "_check_uncached", fake_uncached)
    return p, calls


def test_disk_survives_cold_memory_and_new_pool(tmp_path, monkeypatch):
    ok = {"success": True, "elapsed": 0.5, "errors": []}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, ok, cache_dir)
    r1 = p1.check(THM)
    assert r1["success"] and c1["n"] == 1
    p1._cache.clear()                       # cold MEMORY, disk kept
    r2 = p1.check(THM)
    assert r2["success"] and r2.get("cache_hit") and c1["n"] == 1  # disk hit

    p2, c2 = _pool(tmp_path / "b", monkeypatch, ok, cache_dir)  # "new process"
    r3 = p2.check(THM)
    assert r3["success"] and r3.get("cache_hit") and c2["n"] == 0


def test_slow_failure_not_persisted(tmp_path, monkeypatch):
    bad = {"success": False, "elapsed": 30.0, "errors": ["timeout"]}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, bad, cache_dir)
    p1.check(THM)
    assert c1["n"] == 1
    p1._cache.clear()
    p1.check(THM)
    assert c1["n"] == 2                     # recomputed: nothing on disk


def test_fast_failure_persisted(tmp_path, monkeypatch):
    fastbad = {"success": False, "elapsed": 1.0, "errors": ["by fails"]}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, fastbad, cache_dir)
    p1.check(THM)
    p1._cache.clear()
    r = p1.check(THM)
    assert c1["n"] == 1 and r.get("cache_hit") and not r["success"]


def test_infra_failure_never_persisted(tmp_path, monkeypatch):
    # 2026-07-11 fix: a fast PIDE FAILED/protocol error (worker_error) must
    # NOT be cached as a permanent theorem-failure verdict.
    infra = {"success": False, "elapsed": 0.5, "worker_error": True,
             "errors": ["FAILED: protocol"]}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, infra, cache_dir)
    p1.check(THM)
    p1._cache.clear()
    p1.check(THM)
    assert c1["n"] == 2                     # recomputed both times


def test_incomplete_outcome_never_persisted(tmp_path, monkeypatch):
    # 2026-07-11: a found-but-not-consolidated node (watchdog abort mid-
    # proof) is structurally excluded from caching -- previously only a
    # string match on the error text kept it out, and an incomplete outcome
    # with EMPTY errors could be cached as a fast theorem failure.
    inc = {"success": False, "elapsed": 1.0, "incomplete": True, "errors": []}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, inc, cache_dir)
    p1.check(THM)
    p1._cache.clear()
    p1.check(THM)
    assert c1["n"] == 2                     # recomputed both times


def test_kill_switch(tmp_path, monkeypatch):
    ok = {"success": True, "elapsed": 0.5, "errors": []}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, ok, cache_dir)
    monkeypatch.setenv("ISABELLE_THEOREM_DISK_CACHE", "0")
    p1.check(THM)
    p1._cache.clear()
    p1.check(THM)
    assert c1["n"] == 2                     # disk disabled -> recomputed
