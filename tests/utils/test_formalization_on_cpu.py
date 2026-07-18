"""CPU-only contract tests for the formalization stage of the verification pipeline.

Pins the formalize() seam: the FormalizationOutput flags and attempt logs at each early exit, the typed ordered `steps` list on success (PyExprStep for general steps, DirectDomainStep in its real list position, no "0 == 0" placeholder), and the injected `verify` callable being this stage's only prover access (the givens skeleton check). The translator is replaced at the module seam exactly as in the characterization suite; no Isabelle, no network.
"""
import os
from fractions import Fraction

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("engine imports server_pool (Linux-only os.sysconf)",
                allow_module_level=True)

from verl.utils.isabelle_utils import engine as eng
from verl.utils.isabelle_utils import state_classes
from verl.utils.isabelle_utils.stages import formalization

FIXES = [("a", "real"), ("answer", "real")]
GIVENS = ["answer == a"]
PROBLEM = "Numbers 44 and 999 appear here. Compute answer."


def _nl(*conclusions):
    return [state_classes.NaturalLanguageStep(
                nl_premises=[], nl_conclusion=c, nl_step_text="value %s" % c)
            for c in conclusions]


def _profile():
    return {"translator_http_time": 0.0, "translate_validate_time": 0.0}


def _fake_translate(sc):
    def fake_translate(prompt_base, parse_fn, validate_fn, **_kwargs):
        if getattr(parse_fn, "__name__", "") == "parse_givens_vars_to_pyexpr":
            if sc.get("fail_givens"):
                return None, [{"mock": "givens-failed"}], False
            return (state_classes.PyExprGiven(
                        pyexpr_variable_types=list(sc.get("fixes", FIXES)),
                        pyexpr_givens=list(sc.get("givens", GIVENS))),
                    [{"mock": "givens"}], False)
        if sc.get("fail_steps"):
            return None, [{"mock": "steps-failed"}], False
        parsed = {int(k): state_classes.PyExprStep(
                      pyexpr_conclusion=v["prop"],
                      pyexpr_premises=list(v.get("premises", [])))
                  for k, v in sc["props"].items()}
        return parsed, [{"mock": "steps"}], False
    return fake_translate


def _no_verify(_thm):
    pytest.fail("formalize touched the prover outside validate_givens")


def _formalize(sc, monkeypatch, nl_steps, problem=PROBLEM):
    monkeypatch.setattr(formalization.translator, "translate",
                        _fake_translate(sc))
    return formalization.formalize(problem, nl_steps, eng.IsabelleConfig(),
                                   _no_verify, _profile())


def test_givens_failure_leaves_stage_b_unset(monkeypatch):
    nl_steps = _nl("value 44")
    res = _formalize({"fail_givens": True}, monkeypatch, nl_steps)
    assert isinstance(res, state_classes.FormalizationOutput)
    assert res.givens_ok is False and res.steps_ok is False
    assert res.translation_record_from_problem == [{"mock": "givens-failed"}]
    assert res.translation_record_from_steps is None
    assert res.pyexpr_givens is None and res.steps is None
    assert res.nl_steps is nl_steps


def test_steps_failure_keeps_givens_payload(monkeypatch):
    res = _formalize({"fail_steps": True}, monkeypatch, _nl("value 44"))
    assert res.givens_ok is True and res.steps_ok is False
    assert res.pyexpr_givens == GIVENS
    assert res.pyexpr_variable_types_declared == dict(FIXES)
    assert res.translation_record_from_steps == [{"mock": "steps-failed"}]
    assert res.steps is None


def test_success_payload_and_problem_nums(monkeypatch):
    sc = {"props": {1: {"prop": "a == 44", "premises": ["answer > 0"]},
                    2: {"prop": "answer == 999", "premises": []}}}
    res = _formalize(sc, monkeypatch, _nl("value 44", "value 999 and 123"))
    assert res.givens_ok is True and res.steps_ok is True
    assert all(isinstance(s, state_classes.PyExprStep) for s in res.steps)
    assert [s.pyexpr_conclusion for s in res.steps] == ["a == 44", "answer == 999"]
    assert [s.pyexpr_premises for s in res.steps] == [["answer > 0"], []]
    # single chunk: the steps record is the attempts list itself, not a list of chunk lists
    assert res.translation_record_from_steps == [{"mock": "steps"}]
    assert {Fraction(44), Fraction(999), Fraction(0), Fraction(1),
            Fraction(2)} <= res.problem_nums
    # 999 is computed by step 2's proposition; 123 appears nowhere in it
    assert res.transcription_missing == [[], ["123"]]


def test_direct_domain_statement_takes_its_list_position(monkeypatch):
    claim = "inv x \\<otimes> x = \\<one>"
    sc = {"props": {1: {"prop": "a == 44", "premises": []},
                    2: {"prop": claim, "premises": ["x \\<in> carrier G"]}}}
    nl_steps = _nl("value 44", "group identity")
    res = _formalize(sc, monkeypatch, nl_steps)
    assert isinstance(res.steps[0], state_classes.PyExprStep)
    assert res.steps[0].pyexpr_conclusion == "a == 44"
    domain_step = res.steps[1]
    assert isinstance(domain_step, state_classes.DirectDomainStep)
    assert domain_step.spec is not None
    assert domain_step.premises == ["x \\<in> carrier G"]
    assert domain_step.claim == claim
    assert domain_step.nl_step_text == nl_steps[1].nl_step_text
    # the direct step occupies its real list position; no "0 == 0" placeholder exists anywhere
    assert all(getattr(s, "pyexpr_conclusion", None) != "0 == 0" for s in res.steps)
    assert res.transcription_missing == [[], []]


def test_validate_givens_routes_skeleton_through_injected_verify(monkeypatch):
    """The one prover access this stage has: translate calls validate_givens, whose skeleton typecheck must go through the injected callable and read the typed .proved."""
    theorems = []

    def verify(thm):
        theorems.append(thm)
        return state_classes.VerificationOutcome.from_raw({"success": True})

    def fake_translate(prompt_base, parse_fn, validate_fn, **_kwargs):
        if getattr(parse_fn, "__name__", "") == "parse_givens_vars_to_pyexpr":
            parsed = parse_fn("VARS: a real, answer real\n"
                              "GIVEN: a = 44\nGIVEN: answer = a")
            assert validate_fn(parsed) == []
            return parsed, [{"mock": "givens"}], False
        return ({1: state_classes.PyExprStep(pyexpr_conclusion="a == 44",
                                                 pyexpr_premises=[])},
                [{"mock": "steps"}], False)

    monkeypatch.setattr(formalization.translator, "translate", fake_translate)
    res = formalization.formalize(PROBLEM, _nl("value 44"),
                                  eng.IsabelleConfig(), verify, _profile())
    assert res.steps_ok is True
    assert len(theorems) == 1
    assert "g0" in theorems[0] and '"True"' in theorems[0]
