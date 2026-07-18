"""CPU-only tests for the theorem-result disk cache.

_check_uncached is mocked, so no Isabelle process is needed. The tests verify that a cacheable result survives a cold memory cache and a fresh pool instance, that slow and infrastructure failures are not stored, and that an environment setting can disable the disk cache.
"""
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
    assert r1.proved and c1["n"] == 1
    p1._cache.clear()                       # cold MEMORY, disk kept
    r2 = p1.check(THM)
    assert r2.proved and r2.cache_hit and c1["n"] == 1  # disk hit

    p2, c2 = _pool(tmp_path / "b", monkeypatch, ok, cache_dir)  # "new process"
    r3 = p2.check(THM)
    assert r3.proved and r3.cache_hit and c2["n"] == 0


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
    assert c1["n"] == 1 and r.cache_hit and not r.proved


def test_infra_failure_never_persisted(tmp_path, monkeypatch):
    # A fast PIDE or protocol infrastructure error must not be cached as a permanent theorem failure.
    infra = {"success": False, "elapsed": 0.5, "worker_error": True,
             "errors": ["FAILED: protocol"]}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, infra, cache_dir)
    p1.check(THM)
    p1._cache.clear()
    p1.check(THM)
    assert c1["n"] == 2                     # recomputed both times


def test_incomplete_outcome_never_persisted(tmp_path, monkeypatch):
    # A node that did not finish consolidation is incomplete and must not be cached, even when its error list is empty. The earlier string-only exclusion could store that empty-error result as a fast theorem failure.
    inc = {"success": False, "elapsed": 1.0, "incomplete": True, "errors": []}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, inc, cache_dir)
    p1.check(THM)
    p1._cache.clear()
    p1.check(THM)
    assert c1["n"] == 2 # recomputed both times


def test_kill_switch(tmp_path, monkeypatch):
    ok = {"success": True, "elapsed": 0.5, "errors": []}
    cache_dir = tmp_path / "thm"
    p1, c1 = _pool(tmp_path / "a", monkeypatch, ok, cache_dir)
    monkeypatch.setenv("ISABELLE_THEOREM_DISK_CACHE", "0")
    p1.check(THM)
    p1._cache.clear()
    p1.check(THM)
    assert c1["n"] == 2 # disk disabled -> recomputed
