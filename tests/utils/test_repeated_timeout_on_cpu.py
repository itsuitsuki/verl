"""CPU-only tests for repeated theorem timeouts.

After two timeouts within the configured interval, the same theorem returns an
incomplete result without another Isabelle call. The timeout history expires,
so later checks can try again.
"""
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


def test_third_call_short_circuits_to_incomplete_result(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    calls = {"n": 0}

    def timeout_uncached(code):
        calls["n"] += 1
        return {"success": False, "elapsed": 60.0, "worker_error": True,
                "errors": ["timeout"]}

    monkeypatch.setattr(p, "_check_uncached", timeout_uncached)
    r1 = p.check(THM)              # timeout 1
    r2 = p.check(THM)              # timeout 2
    r3 = p.check(THM)              # >=2 timeouts -> short-circuit, NO prover call
    assert r1.infrastructure_failure and r2.infrastructure_failure
    assert r3.premise_consistency_unknown and r3.infrastructure_failure
    assert calls["n"] == 2         # third call never reached the prover
    assert p.premise_consistency_unknown_short_circuits == 1


def test_success_does_not_timeout(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    calls = {"n": 0}

    def ok_uncached(code):
        calls["n"] += 1
        return {"success": True, "elapsed": 0.5, "errors": []}

    monkeypatch.setattr(p, "_check_uncached", ok_uncached)
    for _ in range(3):
        assert p.check(THM).proved
    # success is cached after the first call -> 1 prover call, no timeouts
    assert calls["n"] == 1
    assert p.premise_consistency_unknown_short_circuits == 0


def test_timeout_expires_after_ttl(tmp_path, monkeypatch):
    p = _pool(tmp_path, monkeypatch)
    p.TIMEOUT_HISTORY_TTL = 0.0     # everything is "expired" immediately
    calls = {"n": 0}

    def timeout_uncached(code):
        calls["n"] += 1
        return {"success": False, "elapsed": 60.0, "worker_error": True,
                "errors": ["timeout"]}

    monkeypatch.setattr(p, "_check_uncached", timeout_uncached)
    p.check(THM)
    p.check(THM)
    p.check(THM)                   # TTL=0 -> timeouts always stale -> still runs
    assert calls["n"] == 3
    assert p.premise_consistency_unknown_short_circuits == 0
