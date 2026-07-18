"""CPU-only pins for the merged direct-check routing (2026-07-17, 屎山代码4).

The separate per-domain HOL-Algebra pools are gone: Isa_Step_Base imports the HOL-Algebra theories, the single engine pool serves both verification paths, and the session runs quick_and_dirty=false. These tests pin what replaced the old lifecycle: direct checks route through the response's PoolVerifier (so they share the pool's workers and land in the prove profile), the engine module keeps no domain registry, and the theorem-cache identity still keys on the owning pool's session environment. No Isabelle process is started.
"""
import os

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf at module level)", allow_module_level=True)

import verl.utils.isabelle_utils.engine as engine
import verl.utils.isabelle_utils.stages.direct_verify as direct_verify
from verl.utils.isabelle_utils import state_classes
from verl.utils.isabelle_utils._server_pool import config as server_pool_config
from verl.utils.isabelle_utils._server_pool import theorem_cache


def test_no_domain_pool_machinery_remains():
    """Tombstone: the merge removed the per-domain pools AND the dict adapter between the stages and the pool; nothing may quietly reintroduce either."""
    for name in ("_domain_pools", "_domain_pool", "_domain_check",
                 "_shutdown_domain_pools", "_direct_check", "_direct_domain_verify"):
        assert not hasattr(engine, name)


def test_direct_verify_accepts_typed_pool_verifier_results():
    """verification passes the response's PoolVerifier to direct_verify.verify UNWRAPPED; direct_verify normalizes each typed result itself (VerificationOutcome.from_raw), so no adapter exists between the stages and the pool."""
    calls = []

    def verify(theorem):
        calls.append(theorem)
        if 'shows "False"' in theorem:
            return state_classes.VerificationOutcome.from_raw(
                {"success": False, "incomplete": False})
        return state_classes.VerificationOutcome.from_raw(
            {"success": "assumes" in theorem, "incomplete": False})

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        r"inv (a \<otimes> b) = inv b \<otimes> inv a",
        "The step states that a and b are elements of group G.",
        verify)
    assert rewarded is True and reason == "rewarded"
    assert calls and all("theorem (in group" in c for c in calls)


def test_direct_verify_maps_typed_fail_closed_to_rejection():
    def verify(theorem):
        return state_classes.VerificationOutcome.from_raw(
            {"success": False, "incomplete": True})

    rewarded, reason = direct_verify.verify(
        direct_verify.FIELD_SPEC,
        [r"x \<in> carrier R", r"x \<noteq> \<zero>"],
        r"x \<otimes> inv x = \<one>",
        "The step states that x is a nonzero element of the field R.",
        verify)
    assert rewarded is False and reason == "premise_consistency_unknown"


def test_session_serves_both_paths_with_kernel_checking():
    """The single session runs quick_and_dirty=false: the no-sorry guarantee is kernel-enforced for the general AND the direct path."""
    assert "quick_and_dirty=false" in server_pool_config.SESSION_OPTIONS
    assert server_pool_config.SESSION == "Isa_Step"


def test_theorem_cache_identity_keys_on_the_session_environment():
    default_fingerprint = theorem_cache._thm_env_fprint()
    other = theorem_cache._thm_env_fprint(
        "Other_Session", '  imports\n    "Other.Theory"\n', ["quick_and_dirty=false"])
    assert default_fingerprint != other
    # deterministic: the same environment always yields the same identity
    assert other == theorem_cache._thm_env_fprint(
        "Other_Session", '  imports\n    "Other.Theory"\n', ["quick_and_dirty=false"])


def test_theorem_disk_path_separates_different_identities():
    theorem = 'theorem chk: shows "(2::int) + 2 = 4" by simp'
    default_path = theorem_cache._thm_disk_path(theorem, theorem_cache._thm_env_fprint())
    other_path = theorem_cache._thm_disk_path(
        theorem, theorem_cache._thm_env_fprint(
            "Other_Session", '  imports\n    "Other.Theory"\n', ["quick_and_dirty=false"]))
    assert default_path != other_path
