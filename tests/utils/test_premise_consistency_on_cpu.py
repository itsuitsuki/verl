"""CPU-only tests for premise-consistency handling.

When a premise-consistency check cannot decide whether accumulated premises are
consistent, the primary tactic gets one with-premises attempt, whose success
must additionally survive the equal-strength guard (the same tactic against
False); a guard as undecidable as the probe voids the success. If the attempt
fails or is voided, the engine tries the canonical premise-free theorem and
then premise-free evaluation. Other premise-dependent additional proof attempts
remain disabled. A proven inconsistency blocks the tail completely (symbol ``c``).

No Isabelle is used: a mock pool decides proof results from theorem text.
numbers in the goal: 55 = premise-free tautology; 77 = provable only with
premises. ALTERNATION contains ``(eval)``, so the mock uses the presence of
assumptions to distinguish the routes.
"""
import re

from verl.utils.isabelle_utils import engine as eng
from verl.utils.isabelle_utils.stages import formalization
from verl.utils.isabelle_utils.state_classes import (
    PremiseConsistency, PremiseConsistencyBoundaries)


class _MockPool:
    """Decide premise-consistency and claim results from theorem text."""

    num_workers = 1

    def __init__(self, consistency_by_step=None):
        self.consistency_by_step = consistency_by_step or {}
        self.claim_calls = []

    def submit(self, code):
        from concurrent.futures import Future

        future = Future()
        future.set_result(self.check(code))
        return future

    def check(self, code):
        if not re.search(r'shows\s+"False"', code):
            self.claim_calls.append(("assumes" in code, code))
        success, extra = self._decide(code)
        result = {
            "success": success,
            "elapsed": 0.01,
            "errors": [] if success else ["no"],
            "queue_wait": 0.0,
            "check_time": 0.01,
        }
        result.update(extra)
        return result

    def _decide(self, code):
        has_assumes = "assumes" in code
        if re.search(r'shows\s+"False"', code):
            k = len(set(re.findall(r"\bs(\d+)\b", code)))
            outcome = self.consistency_by_step.get(k, "consistent")
            if outcome == "inconsistent":
                return True, {}
            if outcome == "unknown":
                return False, {"worker_error": True, "premise_consistency_unknown": True}
            return False, {}
        goal = code.split("shows", 1)[1] if "shows" in code else code
        if "55" in goal:
            return not has_assumes, {}
        if "77" in goal:
            return has_assumes, {}
        return False, {}


def _run(pool, props, monkeypatch):
    fixes = [
        ["a", "real"],
        ["b", "real"],
        ["c", "real"],
        ["d", "real"],
        ["answer", "real"],
    ]
    givens = ["answer == a"]

    def mock_translate(
        prompt_base,
        parse_fn,
        validate_fn,
        **_kw,
    ):
        if "VARS line followed by GIVEN" in prompt_base:
            parsed = eng.state_classes.PyExprGiven(
                pyexpr_variable_types=[tuple(f) for f in fixes],
                pyexpr_givens=list(givens))
            return parsed, [{}], False
        parsed = {int(k): eng.state_classes.PyExprStep(
                      pyexpr_conclusion=v["prop"],
                      pyexpr_premises=list(v.get("premises", [])))
                  for k, v in props.items()}
        return parsed, [{}], False

    monkeypatch.setattr(formalization.translator, "translate", mock_translate)
    steps_xml = ""
    for value in props.values():
        number = "77" if "77" in value["prop"] else "55"
        steps_xml += (
            "<step><premise>-</premise>"
            f"<conclusion>value {number}</conclusion></step>"
        )
    item = {
        "problem": "Numbers 55 and 77 and 5 appear.",
        "response": steps_xml + " \\boxed{55}",
        "ground_truth": "55",
        "dataset": "mock",
        "idx": 0,
        "sample": 0,
    }
    return eng.process_one_response(0, item, pool, eng.IsabelleConfig())


def test_boundaries_return_premise_consistency_status():
    boundaries = PremiseConsistencyBoundaries(
        inconsistent_from_step=3,
        unknown_from_step=1,
    )
    assert boundaries.status_for(0) is PremiseConsistency.CONSISTENT
    assert boundaries.status_for(1) is PremiseConsistency.UNKNOWN
    assert boundaries.status_for(2) is PremiseConsistency.UNKNOWN
    assert boundaries.status_for(3) is PremiseConsistency.INCONSISTENT


def test_unknown_consistency_tries_with_premises_first(monkeypatch):
    props = {
        1: {"prop": "a == 55", "premises": []},
        2: {"prop": "b == 77", "premises": []},
    }
    pool = _MockPool({1: "unknown"})
    rec = _run(pool, props, monkeypatch)

    assert rec["premise_consistency_unknown_at"] == 1
    # The with-premises attempt runs FIRST and succeeds, but its equal-strength guard (the same tactic against False over the same premises) is exactly as undecidable as the probe here, and an ambiguous guard fails closed: the success is voided and the step scores u. The old `oo` pin rested on the claim that an unproven weaker probe protects a with-premises proof, which the guard replaced.
    assert rec["pattern"] == "ou"
    step_two_calls = [
        has_assumes
        for has_assumes, code in pool.claim_calls
        if "77" in code.split("shows", 1)[1]
    ]
    assert step_two_calls[0] is True                       # with premises first
    assert all(h is False for h in step_two_calls[1:])     # then premise-free fallbacks


def test_unknown_consistency_falls_back_to_premise_free(monkeypatch):
    props = {
        1: {"prop": "a == 55", "premises": []},
        2: {"prop": "b == 55", "premises": []},
    }
    pool = _MockPool({1: "unknown"})
    rec = _run(pool, props, monkeypatch)

    assert rec["premise_consistency_unknown_at"] == 1
    assert rec["pattern"] == "oo"
    step_two_calls = [
        has_assumes
        for has_assumes, code in pool.claim_calls
        if "55" in code.split("shows", 1)[1]
    ]
    assert True in step_two_calls
    assert False in step_two_calls


def test_inconsistent_premises_block_all_subsequent_steps(monkeypatch):
    props = {
        1: {"prop": "a == 55", "premises": []},
        2: {"prop": "b == 55", "premises": []},
        3: {"prop": "c == 55", "premises": []},
        4: {"prop": "d == 55", "premises": []},
    }
    rec = _run(_MockPool({2: "inconsistent"}), props, monkeypatch)

    assert rec["premise_consistency_inconsistent_at"] == 2
    assert rec["pattern"] == "oocc"


def test_inconsistency_takes_precedence_over_later_unknown(monkeypatch):
    props = {
        1: {"prop": "a == 55", "premises": []},
        2: {"prop": "b == 55", "premises": []},
        3: {"prop": "c == 55", "premises": []},
        4: {"prop": "d == 55", "premises": []},
    }
    rec = _run(
        _MockPool({1: "inconsistent", 3: "unknown"}),
        props,
        monkeypatch,
    )

    assert rec["premise_consistency_inconsistent_at"] == 1
    assert rec["pattern"] == "occc"
