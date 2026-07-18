"""CPU-only contract tests for the verification stage of the pipeline.

Pins the verify_claims_one_step() seam: the with-premises primary attempt with proven nonzero conditions joined, the canonical premise-free rescue with minimal sorted fixes, the restated-given short circuit (zero prover calls), the restricted attempt set under unknown premise consistency, and the decimal-tolerance fallback. Also pins the verify_response fate of a claim_repeats_given step (pattern x, no claim check). The prover is a scripted verify callable (marker convention from the characterization suite: 77 proves only WITH premises, 55 only premise-free, 88 never); no Isabelle, no pool.
"""
import os

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("engine imports server_pool (Linux-only os.sysconf)",
                allow_module_level=True)

from verl.utils.isabelle_utils import pyexpr, state_classes
from verl.utils.isabelle_utils.stages import verification


class _ScriptedVerify:
    submit = None    # verify_response's synchronous path (no futures, no executor)

    def __init__(self, decide):
        self.decide = decide
        self.calls = []

    def __call__(self, thm):
        self.calls.append(thm)
        decision = self.decide(thm)
        if isinstance(decision, dict):
            return state_classes.VerificationOutcome.from_raw(decision)
        return state_classes.VerificationOutcome.from_raw({"success": bool(decision)})


def _step(claims, problem_premises=None, gflags=None, nodes=None,
          nonzero=None,
          consistency=state_classes.PremiseConsistency.CONSISTENT, **over):
    kw = dict(
        k=0,
        pyexpr_premises=[], pyexpr_conclusion="x == 1",
        pyexpr_definitions=[], pyexpr_claims=nodes or [],
        claim_restates_given=gflags or [False] * len(claims),
        pyexpr_variable_types={"x": "real"},
        isabelle_fixes=[("pv_x", "real")],
        isabelle_problem_premises=list(problem_premises or []),
        isabelle_previous_conclusions=[], isabelle_step_premises=[],
        isabelle_definitions=[], isabelle_claims=list(claims),
        isabelle_nonzero_divisors=list(nonzero or []),
        numeral_carrier="real", nonlinear=False, linear_premises=False,
        linear_step=False, admitted_pyexpr_sources=(),
        trig_context=state_classes.TrigContext())
    kw.update(over)
    step = state_classes.IsabelleStep(**kw)
    step.premise_consistency = consistency
    return step


def test_primary_tactic_with_premises_and_proven_nonzero():
    step = _step(["(pv_x) = (77::real)"],
                 problem_premises=[("g0", "(pv_x) = (77::real)")],
                 nonzero=["(2::real) * pv_x"])

    def decide(thm):
        goal = thm.split("shows", 1)[-1]
        if "\\<noteq>" in goal:
            return True                       # nonzero side condition proven
        return "77" in goal and "assumes" in thm

    verify = _ScriptedVerify(decide)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is True and out["tolerance"] is False
    # the proven condition joined the primary attempt's assumptions as nz0
    assert any("nz0" in c and "77" in c for c in verify.calls)


def test_canonical_premise_free_rescue():
    step = _step(["(pv_x) = (55::real)"],
                 problem_premises=[("g0", "(pv_x) = (44::real)")])
    verify = _ScriptedVerify(lambda thm: "55" in thm and "assumes" not in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is True
    assert any("assumes" in c for c in verify.calls)      # primary tried first
    assert any("assumes" not in c for c in verify.calls)  # bare rescue proved


def test_restated_given_short_circuits_without_prover():
    step = _step(["(pv_x) = (44::real)"], gflags=[True])
    verify = _ScriptedVerify(lambda thm: False)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is True
    assert verify.calls == []


def test_unknown_consistency_restricts_attempts():
    step = _step(["(pv_x) = (88::real)"],
                 problem_premises=[("g0", "(pv_x) = (44::real)")],
                 nonzero=["pv_x"],
                 consistency=state_classes.PremiseConsistency.UNKNOWN)
    verify = _ScriptedVerify(lambda thm: False)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is False
    # one with-premises attempt, the canonical bare goal, premise-free eval -- nothing else
    assert len(verify.calls) == 3
    # nonzero synthesis and tolerance are disabled under unknown consistency
    assert not any("\\<noteq>" in c for c in verify.calls)


def test_tolerance_fallback_marks_result():
    step = _step(["(pv_x) = ((1::real) / (4::real))"],
                 nodes=[pyexpr.parse_expr("x == 0.25")])
    # tolerance proves from premises, so its success also carries the equal-strength guard: the scripted prover must NOT prove the guard's False twin
    verify = _ScriptedVerify(
        lambda thm: "(approximation 20)" in thm and '"False"' not in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is True and out["tolerance"] is True


def test_with_premises_success_is_voided_by_the_equal_strength_guard():
    """An ex-falso route: the tactic proves the claim WITH premises but the same premises and tactic also prove False, so the success is voided and the step stays unverified (the response probe alone runs weaker tactics and could not see this contradiction)."""
    step = _step(["(pv_x) = (77::real)"],
                 problem_premises=[("g0", "(pv_x) = (44::real)")])
    verify = _ScriptedVerify(
        lambda thm: "assumes" in thm and "unfolding" not in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is False
    assert any('shows "False"' in c for c in verify.calls)   # the guard twin ran


def test_guard_ambiguity_fails_closed():
    """A guard twin that times out is ambiguous: the with-premises success is voided, never trusted."""
    step = _step(["(pv_x) = (77::real)"],
                 problem_premises=[("g0", "(pv_x) = (44::real)")])

    def decide(thm):
        if '"False"' in thm:
            return {"success": False, "worker_error": True,
                    "errors": ["hard timeout: probe"]}
        return {"success": "assumes" in thm and "unfolding" not in thm}

    verify = _ScriptedVerify(decide)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is False


def test_wrong_log_premise_cannot_prove_via_the_log_facts():
    """The log-fact asymmetry degeneration check: an admitted wrong log premise (log 2 8 = 4) is invisible to the response probe, and the proven local fact lg2_8 (log 2 8 = 3) contradicts it, so the with-facts attempt can prove anything. The guard twin carries the SAME facts, derives False, and voids the success."""
    step = _step(["(pv_x) = (88::real)"],
                 problem_premises=[("g0", "(log (2::real) (8::real)) = (4::real)")])
    verify = _ScriptedVerify(lambda thm: "lg2_8" in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is False
    assert any("lg2_8" in c and '"False"' in c for c in verify.calls)


def test_log_facts_still_prove_an_honest_claim():
    step = _step(["(pv_x) = (88::real)"],
                 problem_premises=[("g0", "(log (2::real) (8::real)) = (3::real)")])
    verify = _ScriptedVerify(lambda thm: "lg2_8" in thm and '"False"' not in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is True


def _trig_sources():
    return (
        state_classes.AdmittedPyExprSource(
            source_kind=state_classes.PremiseSource.STEP, name="p0_0",
            pyexpr_source="tan(x) == 3/4",
            isabelle_term="((tan pv_x) = ((3::real) / (4::real)))"),
        state_classes.AdmittedPyExprSource(
            source_kind=state_classes.PremiseSource.STEP, name="p0_1",
            pyexpr_source="0 < x", isabelle_term="((0::real) < pv_x)"),
        state_classes.AdmittedPyExprSource(
            source_kind=state_classes.PremiseSource.STEP, name="p0_2",
            pyexpr_source="x < pi / 2", isabelle_term="(pv_x < pi / (2::real))"),
    )


def _trig_step(**over):
    from verl.utils.isabelle_utils import trigonometry
    sources = _trig_sources()
    context = trigonometry.extract_trig_context(sources, {"x": "real"})
    return _step(["sin pv_x = ((3::real) / (5::real))"],
                 admitted_pyexpr_sources=sources, trig_context=context,
                 isabelle_step_premises=[(s.name, s.isabelle_term) for s in sources],
                 **over)


def test_trig_attempt_success_requires_the_exact_twin_to_fail():
    """A guarded trig attempt rewards only when its byte-identical False twin completes UNPROVED; a twin that proves False (an inconsistency the attempt's own machinery can see) voids the attempt."""
    step = _trig_step()
    verify = _ScriptedVerify(
        lambda thm: "sin_from_tan_cpos" in thm and '"False"' not in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is True
    assert any('"False"' in c and "sin_from_tan_cpos" in c for c in verify.calls)

    step2 = _trig_step()
    verify2 = _ScriptedVerify(lambda thm: "sin_from_tan_cpos" in thm)
    out2 = verification.verify_claims_one_step(step2, verify2, {})
    assert out2["verified"] is False         # the twin also proved -> voided


def test_trig_guard_ambiguity_voids_the_attempt():
    step = _trig_step()

    def decide(thm):
        if '"False"' in thm:
            return {"success": False, "worker_error": True,
                    "errors": ["hard timeout: probe"]}
        return {"success": "sin_from_tan_cpos" in thm}

    verify = _ScriptedVerify(decide)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is False


def test_tangent_definedness_check_blocks_every_path():
    """A claim containing a tangent application verifies ONLY when its cos != 0 condition is proved first: a scripted prover that would prove the claim through any tactic family still yields no verification when the condition stays unproved (the tan(pi/2) == 0 totalization FP)."""
    step = _step(["tan (pi / (2::real)) = (0::real)"],
                 isabelle_tan_conditions=[("(pi / (2::real))", None)])
    # proves everything except the condition and the sign facts its twins rest on
    verify = _ScriptedVerify(lambda thm: "\\<noteq> 0" not in thm and "cos (pi" not in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is False
    assert all("cos (pi / (2::real))" in c or '"False"' in c
               for c in verify.calls)          # nothing past the definedness check ran
    assert not any("tan (pi" in c for c in verify.calls)   # no claim attempt was rendered

    step2 = _step(["tan (pi / (4::real)) = (1::real)"],
                  isabelle_tan_conditions=[("(pi / (4::real))", None)])
    verify2 = _ScriptedVerify(lambda thm: True)             # condition provable -> claim proceeds
    out2 = verification.verify_claims_one_step(step2, verify2, {})
    assert out2["verified"] is True


def test_trig_premise_free_attempts_run_under_unknown_consistency():
    """Premise-free identity and special-angle attempts carry no assumptions, so they need no guard and stay available under unknown premise consistency, while every premise-dependent trig attempt is skipped there."""
    step = _trig_step(consistency=state_classes.PremiseConsistency.UNKNOWN)
    verify = _ScriptedVerify(lambda thm: "sin_30" in thm)
    out = verification.verify_claims_one_step(step, verify, {})
    assert out["verified"] is True           # the special-angle group proved premise-free

    step2 = _trig_step(consistency=state_classes.PremiseConsistency.UNKNOWN)
    verify2 = _ScriptedVerify(lambda thm: "sin_from_tan_cpos" in thm)
    out2 = verification.verify_claims_one_step(step2, verify2, {})
    assert out2["verified"] is False
    assert not any("sin_from_tan_cpos" in c for c in verify2.calls)


def test_claim_repeats_given_earns_nothing_without_prover_time():
    """A pure given-restatement step (claim_repeats_given) is never checked or rewarded: only its consistency probe reaches the prover and the pattern shows x, the g*-side analog of the claim_repeats_earlier fate."""
    step = _step(["(pv_x) = (44::real)"], gflags=[True],
                 claim_repeats_given=True,
                 premise_consistency_theorem='theorem probe: shows "False" by linarith')
    verify = _ScriptedVerify(lambda thm: False)
    prepared = state_classes.PreparationOutput(
        steps=[step], isabelle_fixes=[("pv_x", "real")], function_arities={},
        isabelle_problem_conditions=[], isabelle_step_conclusions=[])
    out = verification.verify_response(prepared, verify)
    entry = out.steps[0]
    assert entry.claim_repeats_given is True
    assert entry.verified is False and entry.rewarded is False
    assert entry.to_dict()["claim_repeats_given"] is True
    assert out.pattern == "x"
    assert len(verify.calls) == 1     # the consistency probe only, no claim theorem
