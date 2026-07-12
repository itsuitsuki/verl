"""CPU-only tests for the two-strike shared timeout (server_pool.py,
2026-07-11). A pathological theorem times out on the first worker, is retried
once on a fresh worker, and if it times out AGAIN it is 'undetermined' --
returned immediately for the rest of the strike window WITHOUT another prover
call, so 16 rollouts of one theorem cannot each restart a worker. No Isabelle:
_check_uncached is mocked. Linux-only (os.sysconf at module level)."""
import os

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf)",
                allow_module_level=True)

from verl.utils.isabelle_utils import server_pool as sp

THM = 'theorem t:\n  shows "(2::int)^100000 = 0"\n  by (eval)'


def _pool(tmp_path, monkeypatch):
    monkeypatch.setenv("ISABELLE_THEOREM_DISK_CACHE", "0")
    return sp.IsabelleServerPool(num_workers=1, base_dir=str(tmp_path))


def test_third_call_short_circuits_to_undetermined(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    calls = {"n": 0}

    def timeout_uncached(code):
        calls["n"] += 1
        return {"success": False, "elapsed": 60.0, "worker_error": True,
                "errors": ["timeout"]}

    monkeypatch.setattr(p, "_check_uncached", timeout_uncached)
    r1 = p.check(THM)              # strike 1
    r2 = p.check(THM)              # strike 2
    r3 = p.check(THM)              # >=2 strikes -> short-circuit, NO prover call
    assert r1.get("worker_error") and r2.get("worker_error")
    assert r3.get("undetermined") and r3.get("worker_error")
    assert calls["n"] == 2         # third call never reached the prover
    assert p.undetermined_short_circuits == 1


def test_success_does_not_strike(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    calls = {"n": 0}

    def ok_uncached(code):
        calls["n"] += 1
        return {"success": True, "elapsed": 0.5, "errors": []}

    monkeypatch.setattr(p, "_check_uncached", ok_uncached)
    for _ in range(3):
        assert p.check(THM)["success"]
    # success is cached after the first call -> 1 prover call, no strikes
    assert calls["n"] == 1
    assert p.undetermined_short_circuits == 0


def test_strike_expires_after_ttl(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    p.TIMEOUT_STRIKE_TTL = 0.0     # everything is "expired" immediately
    calls = {"n": 0}

    def timeout_uncached(code):
        calls["n"] += 1
        return {"success": False, "elapsed": 60.0, "worker_error": True,
                "errors": ["timeout"]}

    monkeypatch.setattr(p, "_check_uncached", timeout_uncached)
    p.check(THM)
    p.check(THM)
    p.check(THM)                   # TTL=0 -> strikes always stale -> still runs
    assert calls["n"] == 3
    assert p.undetermined_short_circuits == 0
