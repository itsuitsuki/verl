"""CPU-only contract tests for the engine config isolation (2026-07-16; domain-pool half removed by the 2026-07-17 merge).

The engine is a process-level singleton. The contract these tests pin: a later caller whose effective config MATCHES reuses the existing instance, and one whose config DIFFERS gets a RuntimeError naming the conflict, never a silent reuse under someone else's config (the old behavior was documented as "SILENTLY IGNORED"). No Isabelle process is started: IsabelleEngine is replaced with a recording fake.
"""
import os

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf at module level)", allow_module_level=True)

import verl.utils.isabelle_utils.engine as eng
import verl.utils.reward_score.formal_verify as fv


class _FakeEngine:
    def __init__(self, config):
        self.config = config


@pytest.fixture
def clean_singleton(monkeypatch):
    monkeypatch.setattr(eng, "IsabelleEngine", _FakeEngine)
    monkeypatch.setattr(fv, "_isabelle_engine", None)
    yield


def test_same_effective_config_reuses_the_engine(clean_singleton):
    first = fv._get_isabelle_engine({"isabelle_pool_workers": 3, "fol_timeout": 45})
    second = fv._get_isabelle_engine({"isabelle_pool_workers": 3, "fol_timeout": 45})
    assert first is second


def test_default_config_callers_share_one_engine(clean_singleton):
    first = fv._get_isabelle_engine(None)
    second = fv._get_isabelle_engine({})
    assert first is second


def test_different_config_is_rejected_loudly(clean_singleton):
    fv._get_isabelle_engine({"isabelle_pool_workers": 3, "fol_timeout": 45})
    with pytest.raises(RuntimeError) as excinfo:
        fv._get_isabelle_engine({"isabelle_pool_workers": 1, "fol_timeout": 120})
    message = str(excinfo.value)
    assert "pool_workers" in message
    assert "verify_timeout" in message


# The domain-pool verify_timeout contract tests lived here until the 2026-07-17 merge removed the per-domain pools; direct checks now run under the engine pool's own timeout, covered by the engine-level tests above.
