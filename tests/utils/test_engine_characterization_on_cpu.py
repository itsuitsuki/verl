"""Deterministic characterization baseline for engine.process_one (2026-07-16).

Pins the CURRENT observable semantics of the whole per-response verification pipeline ahead of the planned process_one refactor (phase-1 remaining item): record schema, per-step result fields, the o/x/c/u/m/g pattern, the premise-inconsistency hard cut, guard/transcription behavior, translation early returns, max_steps truncation, direct-domain routing (success, failure, and its interaction with the inconsistency cut), and the theorem-call order on the synchronous path. No Isabelle, JVM, or network: the translator is replaced and the prover pool is a scripted fake whose result is a pure function of the theorem text.

Marker legend (matched as substrings of the goal after `shows`): 77 = provable only WITH premises; 88 = never provable; 55 = provable only premise-free (canonical rescue); 66 = provable only by the exact `by (eval)` tactic (eval rescue). The consistency probe (goal "False") proves iff the count of accumulated s<k> premise labels reaches the scenario's `inconsistent_from` (0-based step index), and returns worker_error from `unknown_from`. Never introduce other constants containing 77/88/55/66 as substrings (770, 551, ...); 999 and 44 are safe and used by the guard/transcription scenarios.

Determinism: the pool is check-only (no `submit`), so the engine falls back to the synchronous path, and step_check_parallelism=1 makes probes then cascades run strictly in step order. Deliberately NOT pinned: rec["prof"] (wall clock), exact theorem text beyond kind/order, and cascade attempt counts on failing steps (the rescue ladder legitimately grows with coverage work).

Known fake-shape traps: the props dict must cover keys 1..n_steps after truncation (the engine trusts validated translate output); algebra vocabulary in any prop/premise routes the step to the direct-domain path, and since the pool merge those theorems flow through the SAME scripted pool: a scenario with direct steps must set rules["domain"] (True/False = prove-with-premises, or a callable(code)->bool), and a general-path scenario that unexpectedly routes direct fails loudly in the pool.
"""
import os
import re

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf at module level)", allow_module_level=True)

from verl.utils.isabelle_utils import engine as eng
from verl.utils.isabelle_utils.stages import formalization

PROBLEM = "Numbers 77, 88, 55, 66, 5 and 2 appear here. Compute answer."
FIXES = [["a", "real"], ["b", "real"], ["c", "real"], ["d", "real"], ["answer", "real"]]
GIVENS = ["answer == a"]


def _steps_response(conclusions):
    body = "".join(
        "<step><premise>-</premise><conclusion>%s</conclusion></step>" % c for c in conclusions
    )
    return body + " \\boxed{77}"


class _ScriptedPool:
    """check-only fake prover: no `submit` attribute, so the engine takes the synchronous path. The result is a pure function of the theorem text (marker legend in the module docstring)."""

    def __init__(self, rules=None):
        self.rules = rules or {}
        self.calls = []

    def check(self, code):
        self.calls.append(code)
        ok, extra = self._decide(code)
        result = {"success": ok, "errors": [] if ok else ["mock fail"],
                  "queue_wait": 0.0, "check_time": 0.01}
        result.update(extra)
        return result

    def _decide(self, code):
        has_assumes = "assumes" in code
        goal = code.split("shows", 1)[1] if "shows" in code else code
        # Direct-domain theorems flow through this SAME pool since the 2026-07-17 merge. rules["domain"]: True/False = prove-with-premises (probe stays consistent, premise-free never proves, matching the old _DomainCheck fake), or a callable(code)->bool for content-dependent scripts; absent = the loud general-path trap.
        if "theorem (in " in code:
            rule = self.rules.get("domain")
            if rule is None:
                pytest.fail("unexpected direct-domain theorem in a general-path scenario: " + code[:120])
            if callable(rule):
                return bool(rule(code)), {}
            if re.search(r'shows\s+"False"', code):
                return False, {}
            if has_assumes:
                return bool(rule), {}
            return False, {}
        if re.search(r'shows\s+"False"', code):
            step_count = len(set(re.findall(r"\bs(\d+)\b", code)))
            if (self.rules.get("inconsistent_from") is not None
                    and step_count >= self.rules["inconsistent_from"]):
                return True, {}
            if (self.rules.get("unknown_from") is not None
                    and step_count >= self.rules["unknown_from"]):
                return False, {"worker_error": True}
            return False, {}
        if "\\<noteq>" in goal:
            return True, {}
        if code.rstrip().endswith("by (eval)"):
            return "66" in goal, {}
        if "77" in goal:
            return has_assumes, {}
        if "88" in goal:
            return False, {}
        if "55" in goal:
            return not has_assumes, {}
        if "66" in goal:
            return False, {}
        return False, {}


class _SubmitPool(_ScriptedPool):
    """submit-capable variant: exercises the engine's FIFO/executor path. Records must match the check-only pool; call ORDER may differ and is not asserted."""

    def submit(self, code):
        import concurrent.futures
        future = concurrent.futures.Future()
        try:
            future.set_result(self.check(code))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future



def _fake_translate(sc, captured_prompts=None, call_counter=None):
    def fake_translate(prompt_base, parse_fn, validate_fn, **_kwargs):
        if call_counter is not None:
            call_counter.append(1)
        if getattr(parse_fn, "__name__", "") == "parse_givens_vars_to_pyexpr":
            if sc.get("fail_givens"):
                return None, [{"mock": "givens-failed"}], False
            parsed = eng.state_classes.PyExprGiven(
                pyexpr_variable_types=[tuple(f) for f in sc.get("fixes", FIXES)],
                pyexpr_givens=list(sc.get("givens", GIVENS)))
            return parsed, [{"mock": "givens"}], False
        if captured_prompts is not None:
            captured_prompts.append(prompt_base)
        if sc.get("fail_steps"):
            return None, [{"mock": "steps-failed"}], False
        parsed = {int(k): eng.state_classes.PyExprStep(
                      pyexpr_conclusion=v["prop"],
                      pyexpr_premises=list(v.get("premises", [])))
                  for k, v in sc["props"].items()}
        return parsed, [{"mock": "steps"}], False
    return fake_translate


def _run(sc, monkeypatch, pool=None, max_steps=0,
         captured_prompts=None, call_counter=None, **config_overrides):
    pool = pool or _ScriptedPool(sc.get("rules"))
    monkeypatch.setattr(formalization.translator, "translate",
                        _fake_translate(sc, captured_prompts, call_counter))
    config = eng.IsabelleConfig(step_check_parallelism=1, **config_overrides)
    item = {"problem": sc.get("problem", PROBLEM), "response": sc["response"],
            "ground_truth": sc.get("gt", "77"), "dataset": "mock", "idx": 0, "sample": 0}
    record = eng.process_one_response(0, item, pool, config, max_steps=max_steps)
    return record, pool


def _kinds(calls):
    kinds = []
    for code in calls:
        if re.search(r'shows\s+"False"', code):
            kinds.append("probe")
        elif "\\<noteq>" in code.split("shows", 1)[-1]:
            kinds.append("nz")
        elif "assumes" in code:
            kinds.append("claim+prem")
        else:
            kinds.append("claim-bare")
    return kinds


def test_all_steps_verified_normal_path(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "b == 77", "premises": []}},
          "response": _steps_response(["value 77", "value 77"])}
    rec, pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "oo"
    assert rec["format_ok"] is True
    assert rec["givens_ok"] is True
    assert rec["steps_ok"] is True
    assert rec["n_steps"] == 2
    assert rec["boxed"] == "77"
    assert rec["outcome_correct"] is True
    assert rec["premise_consistency_inconsistent_at"] is None
    for entry in rec["steps"]:
        assert entry["verified"] is True
        assert entry["rewarded"] is True
        assert entry["guard_ok"] is True
        assert entry["transcription_missing"] == []
        assert entry["premise_consistency_inconsistent"] is False
    # schema pin: top-level keys and step-entry keys of the happy path
    assert {"rid", "dataset", "idx", "sample", "format_ok", "boxed", "outcome_correct",
            "givens_ok", "steps_ok", "n_steps", "steps", "pattern",
            "premise_consistency_inconsistent_at"} <= set(rec.keys())
    # sync path order: both probes first, then each with-premise claim success followed by its equal-strength guard (goal False, same premises and tactic)
    assert _kinds(pool.calls) == ["probe", "probe",
                                  "claim+prem", "probe", "claim+prem", "probe"]


def test_mixed_paths_o_x_neutral_bare_eval(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "b == 88", "premises": []},
                    3: {"prop": "t == a + 77", "premises": []},
                    4: {"prop": "c == 55", "premises": []},
                    5: {"prop": "d == 66", "premises": []}},
          "response": _steps_response(["value 77", "value 88", "define t as a plus 77",
                                       "value 55", "value 66"])}
    rec, _pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "oxgoo"
    steps = rec["steps"]
    assert steps[1]["verified"] is False and steps[1]["rewarded"] is False
    assert steps[2]["neutral"] is True
    assert steps[2]["verified"] is True and steps[2]["rewarded"] is False
    assert steps[2]["n_definitions"] == 1
    assert steps[3]["rewarded"] is True   # canonical premise-free rescue
    assert steps[4]["rewarded"] is True   # eval rescue


def test_inconsistent_premises_hard_cut_c_tail(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "b == 77", "premises": []},
                    3: {"prop": "c == 77", "premises": []},
                    4: {"prop": "d == 77", "premises": []}},
          "response": _steps_response(["value 77"] * 4),
          "rules": {"inconsistent_from": 2}}
    rec, pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "oocc"
    assert rec["premise_consistency_inconsistent_at"] == 2
    assert rec["steps"][2]["premise_consistency_inconsistent"] is True
    assert rec["steps"][3]["premise_consistency_inconsistent"] is True
    # cascades are skipped at/after the cut: 4 probes, then the 2 claim proofs each followed by their equal-strength guard
    assert _kinds(pool.calls) == ["probe"] * 4 + ["claim+prem", "probe",
                                                  "claim+prem", "probe"]


def test_unknown_consistency_scores_u_when_unverified(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "b == 88", "premises": []}},
          "response": _steps_response(["value 77", "value 88"]),
          "rules": {"unknown_from": 1}}
    rec, _pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "ou"
    assert rec.get("premise_consistency_unknown_at") == 1
    assert rec["steps"][1]["premise_consistency_unknown"] is True


def test_guard_invented_constant_scores_g(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77 + 999 - 999", "premises": []},
                    2: {"prop": "b == 77", "premises": []}},
          "response": _steps_response(["value 77", "value 77"])}
    rec, _pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "go"
    entry = rec["steps"][0]
    assert entry["guard_invented"] == ["999"]
    assert entry["guard_ok"] is False
    assert entry["verified"] is True
    assert entry["rewarded"] is False


def test_transcription_missing_scores_m(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77", "premises": []}},
          "response": _steps_response(["a equals 77, not 44"])}
    rec, _pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "m"
    entry = rec["steps"][0]
    assert entry["transcription_missing"] == ["44"]
    assert entry["verified"] is True
    assert entry["rewarded"] is False


def test_admitted_premise_and_nonzero_divisor(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77 / (x - 2)", "premises": ["x == 5"]},
                    2: {"prop": "b == 77", "premises": []}},
          "fixes": FIXES + [["x", "real"]],
          "givens": ["x == 5", "answer == a"],
          "response": _steps_response(["value 77 with x", "value 77"])}
    rec, pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "oo"
    assert rec["steps"][0]["n_admitted_premises"] == 1
    assert _kinds(pool.calls) == ["probe", "probe", "nz",
                                  "claim+prem", "probe", "claim+prem", "probe"]


def test_givens_translation_failure_early_return(monkeypatch):
    sc = {"props": {}, "fail_givens": True,
          "response": _steps_response(["value 77", "value 77"])}
    rec, pool = _run(sc, monkeypatch)
    assert "pattern" not in rec
    assert rec["givens_ok"] is False
    assert rec["steps"] == []
    assert rec["n_steps"] == 2
    assert "translation_record_from_problem" in rec
    assert "translation_record_from_steps" not in rec
    assert pool.calls == []


def test_steps_translation_failure_early_return(monkeypatch):
    sc = {"props": {}, "fail_steps": True,
          "response": _steps_response(["value 77", "value 77"])}
    rec, pool = _run(sc, monkeypatch)
    assert "pattern" not in rec
    assert rec["givens_ok"] is True
    assert rec["steps_ok"] is False
    assert rec["steps"] == []
    assert "translation_record_from_steps" in rec
    assert pool.calls == []


def test_format_failure_no_xml_short_circuits_everything(monkeypatch):
    counter = []
    sc = {"props": {}, "response": "just prose \\boxed{77}"}
    rec, pool = _run(sc, monkeypatch, call_counter=counter)
    assert rec["format_ok"] is False
    assert rec["n_steps"] == 0
    assert rec["steps"] == []
    assert "pattern" not in rec
    assert rec["boxed"] == "77"
    assert pool.calls == []
    assert counter == []   # the translator is never contacted


def test_missing_boxed_answer_still_verifies_steps(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77", "premises": []}},
          "response": "<step><premise>-</premise><conclusion>value 77</conclusion></step>"}
    rec, _pool = _run(sc, monkeypatch)
    assert rec["boxed"] is None
    assert rec["outcome_correct"] is False
    assert rec["pattern"] == "o"


def test_max_steps_truncation(monkeypatch):
    prompts = []
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "b == 77", "premises": []}},
          "response": _steps_response(["value 77 alpha", "value 77 beta",
                                       "value 77 gamma", "value 77 delta"])}
    rec, _pool = _run(sc, monkeypatch, max_steps=2, captured_prompts=prompts)
    assert rec["n_steps"] == 2
    assert len(rec["pattern"]) == 2
    # truncation happens BEFORE translation: the steps prompt never mentions step 3
    assert prompts and all("gamma" not in p for p in prompts)


DOMAIN_STEP = {"prop": "x \\<otimes> y \\<in> carrier G",
               "premises": ["x \\<in> carrier G", "y \\<in> carrier G"]}
DOMAIN_TEXT = "elements x and y lie in the carrier of G, so their product does too"


def test_domain_step_success_is_rewarded(monkeypatch):
    sc = {"props": {1: dict(DOMAIN_STEP), 2: {"prop": "b == 77", "premises": []}},
          "response": _steps_response([DOMAIN_TEXT, "value 77"]),
          "rules": {"domain": True}}
    rec, pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "oo"
    entry = rec["steps"][0]
    assert entry["verified"] is True and entry["rewarded"] is True
    assert entry["domain_reason"] == "rewarded"
    # merged pool: the domain theorems flow through the SAME pool, in-locale; general theorems stay free of group vocabulary
    domain_calls = [c for c in pool.calls if "theorem (in group" in c]
    general_calls = [c for c in pool.calls if "theorem (in " not in c]
    assert domain_calls
    assert all("carrier" not in c for c in general_calls)
    assert 'shows "False"' in domain_calls[0]            # consistency probe first
    assert "assumes" not in domain_calls[-1]             # premise-free non-triviality last


def test_domain_step_failure_scores_x_without_crash(monkeypatch):
    # Hazard pin: pattern generation indexes premise_consistency_inconsistent on every
    # step, and the merge loop writes it BEFORE the domain branch. A refactor that
    # hoists the domain branch above that assignment reintroduces a KeyError here.
    sc = {"props": {1: dict(DOMAIN_STEP), 2: {"prop": "b == 77", "premises": []}},
          "response": _steps_response([DOMAIN_TEXT, "value 77"]),
          "rules": {"domain": False}}
    rec, _pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "xo"
    entry = rec["steps"][0]
    assert entry["verified"] is False and entry["rewarded"] is False
    assert entry["domain_reason"] == "not_proved"
    assert "premise_consistency_inconsistent" in entry


def test_domain_step_ignores_inconsistency_cut(monkeypatch):
    # Documented oddity (2026-07-16): the domain branch runs before the hard-cut mode check,
    # so a domain step past the cut is still verified from ITS OWN stated premises.
    # Deliberate per the mixed-domain design note in engine.py;
    # pinned here so any change to that interaction is a conscious decision.
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "b == 77", "premises": []},
                    3: dict(DOMAIN_STEP)},
          "response": _steps_response(["value 77", "value 77", DOMAIN_TEXT]),
          "rules": {"inconsistent_from": 1, "domain": True}}
    rec, _pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "oco"
    assert rec["premise_consistency_inconsistent_at"] == 1
    entry = rec["steps"][2]
    assert entry["rewarded"] is True
    assert entry["premise_consistency_inconsistent"] is True


def test_domain_conclusion_excluded_from_later_premise_chain(monkeypatch):
    # Direct degeneration check for the mixed-domain residual (CLAUDE.md): a NON-WHITELISTED direct claim (here `x \<otimes> y \<in> carrier G`, not an arithmetic-valued `order/card/ord ... = n`) never enters later general steps' premise chains. Steps: general(0) -> domain(1) -> general(2). Step 2's claim theorem must carry s0 (the general conclusion) but neither s1 (the skipped domain slot) nor any group vocabulary, and no placeholder may leak in as a premise.
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: dict(DOMAIN_STEP),
                    3: {"prop": "b == 77", "premises": []}},
          "response": _steps_response(["value 77", DOMAIN_TEXT, "value 77"]),
          "rules": {"domain": True}}
    rec, pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "ooo"
    step2_claims = [c for c in pool.calls
                    if "assumes" in c and 'shows "False"' not in c and "s0:" in c]
    assert step2_claims, "step 2's claim theorem should carry the general conclusion s0"
    for theorem in step2_claims:
        assert "s1:" not in theorem
        assert "carrier" not in theorem


def test_bridged_direct_claim_feeds_both_directions(monkeypatch):
    # Cross-session bridge pin. general->direct: the domain checker's with-assumptions theorems carry the earlier general conclusion (pv_ arithmetic). direct->general: the whitelisted claim `order G = 15` fills its s* slot; the later general step's premise `order_G == 15` is admitted (15 traces to the bridged claim, not the problem text) and its claim theorem carries the bridged s1.
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "order G = 15",
                        "premises": ["H \\<subseteq> carrier G"]},
                    3: {"prop": "b == 77", "premises": ["order_G == 15"]}},
          "response": _steps_response(["value 77",
                                       "the subgroup H gives the order of G as 15",
                                       "so we get value 77"]),
          "rules": {"domain": True}}
    rec, pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "ooo"
    assert rec["steps"][1]["domain_reason"] == "rewarded"
    domain_with_assumes = [c for c in pool.calls
                           if "theorem (in " in c and "assumes" in c]
    assert domain_with_assumes and all("pv_a" in c for c in domain_with_assumes)
    assert rec["steps"][2]["n_admitted_premises"] == 1
    assert any("s1:" in c and "pv_order_G" in c for c in pool.calls)


def test_direct_chain_second_step_uses_first_conclusion(monkeypatch):
    # Response-level plumbing pin for same-domain chaining (translation-time admission): the second group step's claim proves only when the first group step's conclusion is among its assumptions.
    first_claim = DOMAIN_STEP["prop"]
    second = {"prop": "inv x \\<otimes> (x \\<otimes> y) = y",
              "premises": ["x \\<in> carrier G"]}

    def chain_rule(theorem):
        if 'shows "False"' in theorem:
            return False
        if "assumes" not in theorem:
            return False
        if "inv x" in theorem:   # the second step's claim theorem
            return first_claim in theorem
        return True

    sc = {"props": {1: dict(DOMAIN_STEP), 2: second,
                    3: {"prop": "b == 77", "premises": []}},
          "response": _steps_response([DOMAIN_TEXT,
                                       "then inv x times x times y gives y",
                                       "value 77"]),
          "rules": {"domain": chain_rule}}
    rec, pool = _run(sc, monkeypatch)
    assert rec["pattern"] == "ooo"
    assert rec["steps"][1]["domain_reason"] == "rewarded"
    # the chain stays inside the in-locale theorems: general (label-carrying) theorems see no group vocabulary
    assert all("carrier" not in c for c in pool.calls if "theorem (in " not in c)


class _TrigScriptedPool(_ScriptedPool):
    """Markers for the general trig attempts: a rendered trig value theorem carries a session meta lemma name; `trig_positive` / `trig_guard_false` script the positive rendering and its False twin, and everything else falls back to the numeric marker legend."""

    def __init__(self, rules=None, trig_positive=False, trig_guard_false=False):
        super().__init__(rules)
        self.trig_positive = trig_positive
        self.trig_guard_false = trig_guard_false

    def _decide(self, code):
        if "sin_from_tan" in code or "cos_from_tan" in code:
            if re.search(r'shows\s+"False"', code):
                return self.trig_guard_false, {}
            return self.trig_positive, {}
        return super()._decide(code)


_TRIG_SC = {"props": {1: {"prop": "sin(w) == 3/5",
                          "premises": ["tan(w) == 3/4", "0 < w", "w < pi / 2"]}},
            "fixes": FIXES + [["w", "real"]],
            "problem": "Numbers 3, 4, 5 and 2 appear here. Compute answer.",
            "response": _steps_response(["sin of w is 3/5"])}


def test_trig_value_step_rewarded_with_guard(monkeypatch):
    # The general trig system inside the engine pipeline: the tan premise plus strict Q1
    # bounds yield a guarded value attempt; the scripted pool proves the positive rendering
    # and leaves the byte-identical False twin unproved, so the step is rewarded. Call-order
    # pin: the twin follows immediately after its positive.
    rec, pool = _run(dict(_TRIG_SC), monkeypatch,
                     pool=_TrigScriptedPool(trig_positive=True))
    assert rec["pattern"] == "o"
    trig_calls = [c for c in pool.calls if "sin_from_tan" in c]
    assert len(trig_calls) == 2
    assert 'shows "False"' not in trig_calls[0]
    assert 'shows "False"' in trig_calls[1]


def test_trig_guard_refutation_withholds_reward(monkeypatch):
    # Same scenario, but the False twin also proves (an inconsistency the attempt's own
    # machinery can see): the attempt is voided and the step stays unverified.
    rec, _pool = _run(dict(_TRIG_SC), monkeypatch,
                      pool=_TrigScriptedPool(trig_positive=True, trig_guard_false=True))
    assert rec["pattern"] == "x"


def test_trig_provenance_dropped_premise_plans_no_value_attempt(monkeypatch):
    # The tan value 9/4 has no provenance (problem text lacks 9), so admission drops the
    # premise, the trig context carries no tan evidence, and no trig value theorem is
    # ever submitted.
    sc = {"props": {1: {"prop": "sin(w) == 3/5",
                        "premises": ["tan(w) == 9/4", "0 < w", "w < pi / 2"]}},
          "fixes": FIXES + [["w", "real"]],
          "problem": "Numbers 3, 4, 5 and 2 appear here. Compute answer.",
          "response": _steps_response(["sin of w is 3/5"])}
    rec, pool = _run(sc, monkeypatch, pool=_TrigScriptedPool(trig_positive=True))
    assert rec["pattern"] == "x"
    assert not any("sin_from_tan" in c for c in pool.calls)


def test_submit_pool_produces_identical_record(monkeypatch):
    sc = {"props": {1: {"prop": "a == 77", "premises": []},
                    2: {"prop": "b == 88", "premises": []},
                    3: {"prop": "t == a + 77", "premises": []},
                    4: {"prop": "c == 55", "premises": []},
                    5: {"prop": "d == 66", "premises": []}},
          "response": _steps_response(["value 77", "value 88", "define t as a plus 77",
                                       "value 55", "value 66"])}
    rec_sync, _ = _run(sc, monkeypatch, pool=_ScriptedPool())
    rec_submit, _ = _run(sc, monkeypatch, pool=_SubmitPool())
    rec_sync.pop("prof", None)
    rec_submit.pop("prof", None)
    assert rec_sync == rec_submit


def test_chunked_translation_merges_across_chunks(monkeypatch):
    # translate_chunk_steps=2 over 6 steps -> 3 steps-translate calls; later chunks
    # carry the earlier props as context ("Steps already translated"). The engine
    # keeps only keys <= block_end from each chunk's parse, so returning the full
    # props dict every call exercises that filter.
    prompts = []
    sc = {"props": {k: {"prop": "%s == 77" % v, "premises": []}
                    for k, v in enumerate(["a", "b", "c", "d", "u", "v"], start=1)},
          "fixes": FIXES + [["u", "real"], ["v", "real"]],
          "response": _steps_response(["value 77"] * 6)}
    rec, _pool = _run(sc, monkeypatch, captured_prompts=prompts, translate_chunk_steps=2)
    assert rec["pattern"] == "oooooo"
    assert len(prompts) == 3
    assert "Steps already translated" not in prompts[0]
    assert all("Steps already translated" in p for p in prompts[1:])
    assert isinstance(rec["translation_record_from_steps"], list) and len(rec["translation_record_from_steps"]) == 3
