"""CPU-only contract tests for the preparation stage of the verification pipeline.

Pins the prepare() seam: response-wide type inference and declarations, the definition vs claim split, provenance-checked premise admission, the g/s/p/d premise labels with direct-domain steps excluded from later premise chains, the premise-consistency probe (built here; None on dangerous terms), the DirectDomainStep passing through into the ordered steps list unchanged, and the guards/counts riding the IsabelleStep as verification input (preparation creates no result objects). Pure python; no Isabelle, no prover, no translator.
"""
import os
from fractions import Fraction
from types import SimpleNamespace

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("engine imports server_pool (Linux-only os.sysconf)",
                allow_module_level=True)

from verl.utils.isabelle_utils import pyexpr, state_classes, tactics
from verl.utils.isabelle_utils.stages import preparation


def _formalized(conclusions, premises=None, givens=None, declared=None,
                domain_steps=None, problem_nums=(4,), transcription=None):
    n = len(conclusions)
    premises = [list(p) for p in (premises or [[]] * n)]
    domain_steps = dict(domain_steps or {})
    # the typed ordered list: a DirectDomainStep takes its real position, everything else is a PyExprStep built from the paired conclusions/premises
    steps = [domain_steps[k] if k in domain_steps
             else state_classes.PyExprStep(pyexpr_conclusion=conclusions[k],
                                           pyexpr_premises=premises[k])
             for k in range(n)]
    return state_classes.FormalizationOutput(
        nl_steps=[state_classes.NaturalLanguageStep(
                      nl_premises=[], nl_conclusion="step %d" % k,
                      nl_step_text="step %d text" % k) for k in range(n)],
        problem_nums={Fraction(v) for v in problem_nums}
                     | {Fraction(0), Fraction(1), Fraction(2)},
        givens_ok=True, steps_ok=True,
        pyexpr_givens=list(givens or ["answer == n"]),
        pyexpr_variable_types_declared=dict(
            declared or {"n": "int", "answer": "int"}),
        steps=steps,
        transcription_missing=list(transcription or [[]] * n))


def test_type_inference_and_fixes():
    prepared = preparation.prepare(_formalized(["m == n + 4", "answer == m"]))
    s0 = prepared.steps[0]
    # every declared variable is int, so the fresh intermediate defaults to int
    assert s0.pyexpr_variable_types["m"] == "int"
    assert (pyexpr._add_prefix_pv("m"), "int") in prepared.isabelle_fixes
    assert prepared.function_arities == {}
    assert len(prepared.steps) == 2
    assert len(prepared.isabelle_step_conclusions) == 2
    assert len(prepared.isabelle_problem_conditions) == 1


def test_definition_vs_claim_split_and_premise_labels():
    prepared = preparation.prepare(
        _formalized(["m == n + 4 and m > 0", "answer == m"]))
    s0, s1 = prepared.steps
    # m is fresh and defined over known identifiers -> assumption, not obligation
    assert s0.n_definitions == 1
    assert [lbl for lbl, _ in s0.isabelle_definitions] == ["d0_0"]
    assert len(s0.isabelle_claims) == 1        # m > 0 stays an obligation
    assert s1.claim_restates_given == [False]
    # theorem premise order: problem givens, then earlier conclusions
    assert [lbl for lbl, _ in s1.isabelle_premises] == ["g0", "s0"]


def test_premise_admission_requires_known_numbers():
    prepared = preparation.prepare(
        _formalized(["answer == 4"], premises=[["n == 4", "q == 77"]]))
    s0 = prepared.steps[0]
    # 77 traces to no problem number, earlier conclusion, or step text -> dropped
    assert s0.n_admitted_premises == 1
    assert [lbl for lbl, _ in s0.isabelle_step_premises] == ["p0_0"]


def _domain_step(claim, name="group", text="step text"):
    family = "algebra" if name in ("ring", "field") else name
    return state_classes.DirectDomainStep(
        spec=SimpleNamespace(name=name, chain_family=family),
        premises=["x \\<in> carrier G"], claim=claim, nl_step_text=text)


def test_domain_step_excluded_from_later_premise_chains():
    domain = {1: _domain_step("inv x \\<otimes> x = \\<one>")}
    prepared = preparation.prepare(
        _formalized(["answer == 4", "inv x \\<otimes> x = \\<one>", "n == 4"],
                    domain_steps=domain))
    # the DirectDomainStep keeps its real list position (enriched copy, chain empty here)
    assert isinstance(prepared.steps[1], state_classes.DirectDomainStep)
    assert prepared.steps[1].claim == domain[1].claim
    assert prepared.steps[1].previous_conclusions == ()
    assert prepared.isabelle_step_conclusions[1] is None
    labels = [lbl for lbl, _ in prepared.steps[2].isabelle_previous_conclusions]
    assert labels == ["s0"]     # the domain step's s1 never enters later general chains


def test_direct_chain_accumulates_same_domain_conclusions_only():
    domain = {1: _domain_step("inv x \\<otimes> x = \\<one>"),
              2: _domain_step("x \\<oplus> y = y \\<oplus> x", name="ring"),
              3: _domain_step("x \\<otimes> \\<one> = x")}
    prepared = preparation.prepare(
        _formalized(["answer == 4", "d", "d", "d"], domain_steps=domain))
    assert prepared.steps[1].previous_conclusions == ()
    # a ring step never receives the group chain
    assert prepared.steps[2].previous_conclusions == ()
    # the later group step receives exactly the earlier group conclusion, not its own
    assert prepared.steps[3].previous_conclusions == ("inv x \\<otimes> x = \\<one>",)


def test_ring_and_field_share_one_chain_group_separate():
    domain = {0: _domain_step("x \\<oplus> y = y \\<oplus> x", name="ring"),
              1: _domain_step("a \\<otimes> inv a = \\<one>", name="field"),
              2: _domain_step("inv g \\<otimes> g = \\<one>", name="group")}
    prepared = preparation.prepare(
        _formalized(["d", "d", "d"], domain_steps=domain))
    # ring and field share the algebra family: the field step receives the ring conclusion
    assert prepared.steps[1].previous_conclusions == ("x \\<oplus> y = y \\<oplus> x",)
    # group is a different structure: no algebra-family conclusion enters it
    assert prepared.steps[2].previous_conclusions == ()


def test_whitelisted_direct_claim_bridges_into_general_chain():
    # direct->general bridge: `order G = 15` fills its own s* slot as the nominal pyexpr equation, typed and fixed like any general conclusion
    domain = {0: _domain_step("order G = 15")}
    prepared = preparation.prepare(
        _formalized(["d", "answer == 4"], domain_steps=domain))
    term, _carrier = prepared.isabelle_step_conclusions[0]
    assert "pv_order_G" in term and "15" in term
    labels = [lbl for lbl, _ in prepared.steps[1].isabelle_previous_conclusions]
    assert labels == ["s0"]
    assert ("pv_order_G", "int") in prepared.isabelle_fixes


def test_direct_step_receives_earlier_general_conclusions():
    # general->direct bridge: the direct step carries every earlier general-session conclusion, in transpiled pv_ form
    domain = {1: _domain_step("inv x \\<otimes> x = \\<one>")}
    prepared = preparation.prepare(
        _formalized(["answer == 4", "d"], domain_steps=domain))
    (nc,) = prepared.steps[1].general_conclusions
    assert "pv_answer" in nc and "4" in nc


def test_direct_claim_numerals_join_general_premise_provenance():
    # 15 appears only inside the direct claim; the later general step's premise using it must be admitted (before this flow it would be dropped like the 77 above)
    domain = {0: _domain_step("order G = 15")}
    prepared = preparation.prepare(
        _formalized(["d", "answer == 4"], premises=[[], ["q == 15"]],
                    domain_steps=domain))
    assert prepared.steps[1].n_admitted_premises == 1


def test_claim_repeating_earlier_conclusion_is_flagged():
    """s*-chain echo pin: a step whose every claim conjunct restates an earlier conclusion is flagged claim_repeats_earlier (verification withholds the reward, pattern x); one fresh conjunct keeps the step normal."""
    prepared = preparation.prepare(_formalized([
        "m == n + 4",            # step 0: definition, enters the chain as s0
        "m == n + 4",            # step 1: m is now known -> claim restating s0 -> echo
        "m == n + 4 and m > 0",  # step 2: fresh conjunct m > 0 -> not an echo
        "answer == m",           # step 3: fresh claim -> not an echo
    ]))
    flags = [s.claim_repeats_earlier for s in prepared.steps]
    assert flags == [False, True, False, False]


def test_bridged_direct_claim_counts_as_echo_material():
    """A general step restating the direct->general bridged equation (order_G == 15 fills the s* slot) is the same echo."""
    domain = {0: _domain_step("order G = 15")}
    prepared = preparation.prepare(
        _formalized(["d", "order_G == 15", "answer == 4"], domain_steps=domain))
    assert prepared.steps[1].claim_repeats_earlier is True
    assert prepared.steps[2].claim_repeats_earlier is False


def test_claim_repeating_given_is_flagged():
    """g*-side echo pin: a step whose every claim conjunct restates a GIVEN is claim_repeats_given (verification withholds the reward, pattern x); a novel conjunct keeps the step on normal verification with the restated conjunct auto-satisfied per claim_restates_given."""
    prepared = preparation.prepare(_formalized([
        "n == 3",             # step 0: restates the given verbatim
        "n == 3 and m == 9",  # step 1: novel conjunct m == 9, normal verification
    ], givens=["n == 3"], problem_nums=(3, 9)))
    assert [s.claim_repeats_given for s in prepared.steps] == [True, False]
    assert [s.claim_repeats_earlier for s in prepared.steps] == [False, False]
    assert prepared.steps[1].claim_restates_given == [True, False]


def test_echo_mixing_given_and_chain_content_is_claim_repeats_earlier():
    """An all-echo step whose conjuncts split between the chain (s*) and the givens (g*) proves nothing new either; it lands under claim_repeats_earlier because the chain side may be unverified."""
    prepared = preparation.prepare(_formalized([
        "m == n + 4",             # step 0: definition, enters the chain as s0
        "m == n + 4 and n == 3",  # step 1: one conjunct from the chain, one from the givens
    ], givens=["n == 3"], problem_nums=(3, 4)))
    assert prepared.steps[1].claim_repeats_earlier is True
    assert prepared.steps[1].claim_repeats_given is False


def test_trig_context_extracts_only_admitted_sources():
    """The general trig system consumes ONLY admission-passed sources, each bound to its named assumption: a tan premise dropped by number provenance contributes no evidence, and with provenance the context carries the tan value plus both exact-pi bounds under their p{k}_{i} names."""
    steps = ["sin(a) == 1"]
    premises = [["tan(a) == 9", "0 < a", "a < pi / 2"]]
    declared = {"n": "int", "answer": "int", "a": "real"}
    dropped = preparation.prepare(_formalized(
        steps, premises=premises, declared=declared))          # 9 has no provenance
    assert all(value.function != "tan" for value in dropped.steps[0].trig_context.values)
    admitted = preparation.prepare(_formalized(
        steps, premises=premises, declared=declared, problem_nums=(4, 9)))
    context = admitted.steps[0].trig_context
    assert any(value.function == "tan" and value.source_names == ("p0_0",)
               for value in context.values)
    assert len(context.pi_bounds) == 2
    source_names = {source.name for source in admitted.steps[0].admitted_pyexpr_sources}
    assert {"g0", "p0_0", "p0_1", "p0_2"} <= source_names


def test_consistency_probe_built_here_and_none_when_dangerous():
    prepared = preparation.prepare(
        _formalized(["answer == 4", "n == n ** 2", "n == 4"]))
    thms = [s.premise_consistency_theorem for s in prepared.steps]
    # steps 0 and 1 sit on an all-affine premise chain -> complete decision procedure
    assert 'shows "False"' in thms[0] and tactics.LINEAR_FALSE in thms[0]
    # step 2 accumulates the nonlinear step-1 conclusion -> general tactic
    assert tactics.ALTERNATION in thms[2]
    dangerous = preparation.prepare(
        _formalized(["n == 4"], givens=["answer == 2 ** 1500"]))
    assert dangerous.steps[0].premise_consistency_theorem is None


def test_transcription_copied_from_formalization():
    prepared = preparation.prepare(
        _formalized(["answer == 4"], transcription=[["7"]]))
    assert prepared.steps[0].transcription_missing == ["7"]
    assert prepared.steps[0].guard_ok is True
