"""CPU-only contract tests for direct Isabelle-domain verification.

Direct-domain steps are statements emitted in Isabelle syntax for group, ring,
and field claims that the restricted Python expression language cannot
represent. These tests use a deterministic prover stub and do not start
Isabelle.

The tests separate existing fail-closed rules from two required soundness
rules:

* a translated premise must not be the same statement as the translated claim;
* a translated premise must not introduce a non-vocabulary identifier absent
  from the model's source step.
"""

from verl.utils.isabelle_utils.stages import direct_verify


class RecordingCheck:
    """Small deterministic stand-in for one domain pool's theorem checker."""

    def __init__(self, *, prove_with_premises=True, prove_without_premises=False,
                 inconsistent=False, incomplete=False):
        self.prove_with_premises = prove_with_premises
        self.prove_without_premises = prove_without_premises
        self.inconsistent = inconsistent
        self.incomplete = incomplete
        self.calls = []

    def __call__(self, theorem):
        self.calls.append(theorem)
        if self.incomplete:
            return {"success": False, "incomplete": True}
        if 'shows "False"' in theorem:
            return {"success": self.inconsistent, "incomplete": False}
        has_assumptions = "assumes" in theorem
        return {
            "success": (
                self.prove_with_premises
                if has_assumptions
                else self.prove_without_premises
            ),
            "incomplete": False,
        }


def _valid_claim():
    return r"inv (a \<otimes> b) = inv b \<otimes> inv a"


def test_valid_group_claim_is_rewarded():
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is True
    assert reason == "rewarded"
    # probe = freecheck plus every distinct claim tactic against False, then one successful claim attempt, then the premise-free non-triviality check over the SAME battery (every tactic must fail to prove the bare claim)
    probes = 1 + len([t for t in direct_verify.GROUP_SPEC.claim_tactics
                      if t != direct_verify.GROUP_SPEC.freecheck])
    assert len(check.calls) == probes + 1 + probes


def test_banned_construct_is_rejected_before_proving():
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [],
        r"a = a by sorry",
        "The step mentions a.",
        check,
    )

    assert rewarded is False
    assert reason.startswith("sanitize:banned construct")
    assert check.calls == []


def test_invented_claim_identifier_is_rejected():
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        r"inv z = z",
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "invented:z"
    assert check.calls == []


def test_inconsistent_premises_are_rejected():
    check = RecordingCheck(inconsistent=True)

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "inconsistent"
    assert len(check.calls) == 1


def test_incomplete_premise_consistency_is_rejected():
    check = RecordingCheck(incomplete=True)

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "premise_consistency_unknown"
    assert len(check.calls) == 1


def test_unproved_claim_is_rejected():
    check = RecordingCheck(prove_with_premises=False)

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "not_proved"


def test_premise_free_claim_is_rejected_as_trivial():
    check = RecordingCheck(prove_without_premises=True)

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "trivial"


def test_premise_free_check_incomplete_is_rejected():
    class CheckClaimThenIncompleteFree:
        def __init__(self):
            self.calls = []

        def __call__(self, theorem):
            self.calls.append(theorem)
            if 'shows "False"' in theorem:
                return {"success": False, "incomplete": False}
            if "assumes" in theorem:
                return {"success": True, "incomplete": False}
            return {"success": False, "incomplete": True}

    check = CheckClaimThenIncompleteFree()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "freecheck_unknown"


def test_claim_cannot_be_repeated_as_a_premise():
    """A translator must not turn `assume P; show P` into a reward."""
    claim = _valid_claim()
    repeated_with_spacing_difference = claim.replace("inv (", "inv   (")
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [repeated_with_spacing_difference],
        claim,
        "The step states the inverse law for a, b, and the group identity.",
        check,
    )

    assert rewarded is False
    assert reason == "claim_repeated_as_premise"
    assert check.calls == []


def test_premise_identifier_absent_from_source_is_rejected():
    """A direct premise may not introduce a new entity unseen in the step."""
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"z \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "invented_premise:z"
    assert check.calls == []


def test_previous_conclusion_enables_the_proof():
    """An earlier same-domain conclusion joins the probe and the with-premises proof (general s* analog)."""
    injected = r"a \<otimes> b \<in> carrier G"

    class ProvesOnlyWithInjected:
        def __init__(self):
            self.calls = []

        def __call__(self, theorem):
            self.calls.append(theorem)
            if 'shows "False"' in theorem:
                return {"success": False, "incomplete": False}
            if "assumes" in theorem:
                return {"success": injected in theorem, "incomplete": False}
            return {"success": False, "incomplete": False}

    check = ProvesOnlyWithInjected()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
        previous_conclusions=(injected,),
    )

    assert rewarded is True
    assert reason == "rewarded"
    # every with-assumptions theorem (probe and claim) carries the injected conclusion
    assert all(injected in c for c in check.calls if "assumes" in c)


def test_poisoned_previous_conclusion_fails_closed():
    """A contradictory injected chain rejects the step at the probe, even with no own premises."""
    check = RecordingCheck(inconsistent=True)

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [],
        _valid_claim(),
        "The step states the inverse law for a and b in group G.",
        check,
        previous_conclusions=(r"\<one> \<noteq> \<one>",),
    )

    assert rewarded is False
    assert reason == "inconsistent"
    assert len(check.calls) == 1


def test_previous_conclusion_exempt_from_entity_anchor():
    """Injected conclusions were anchored at their origin step: a foreign identifier there must not reject THIS step."""
    claim = _valid_claim()
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        claim,
        "The step states that a and b are elements of group G.",
        check,
        previous_conclusions=(r"z \<otimes> z = \<one>",),
    )

    assert rewarded is True
    assert reason == "rewarded"


def test_claim_repeating_injected_conclusion_is_rejected():
    """The direct-path s*-chain echo: a claim equal to an injected chain conclusion proves only via its (possibly unverified) twin, so it earns nothing. Replaces the earlier chain-reuse exemption (2026-07-17 decision)."""
    claim = _valid_claim()

    for kwargs in ({"previous_conclusions": (claim,)},
                   {"general_conclusions": (claim,)}):
        check = RecordingCheck()
        rewarded, reason = direct_verify.verify(
            direct_verify.GROUP_SPEC,
            [r"a \<in> carrier G", r"b \<in> carrier G"],
            claim,
            "The step states that a and b are elements of group G.",
            check,
            **kwargs,
        )
        assert rewarded is False
        assert reason == "claim_repeats_chain"
        assert check.calls == []


def test_probe_runs_every_claim_tactic_against_false():
    """A contradiction visible only to a stronger tactic (not the freecheck) must still reject the step: with a freecheck-only probe the claim tactic could prove the claim ex falso."""

    class MetisOnlyContradiction:
        def __init__(self):
            self.calls = []

        def __call__(self, theorem):
            self.calls.append(theorem)
            if 'shows "False"' in theorem:
                return {"success": "metis" in theorem, "incomplete": False}
            return {"success": "assumes" in theorem, "incomplete": False}

    check = MetisOnlyContradiction()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "inconsistent"


def test_general_conclusions_join_probe_and_proof():
    """Bridged general-chain facts (transpiled pv_ arithmetic) behave exactly like same-domain injected conclusions: probed, usable in the with-premises proof, exempt from this step's entity anchor."""
    injected = r"(pv_n::int) = (15::int)"

    class ProvesOnlyWithInjected:
        def __init__(self):
            self.calls = []

        def __call__(self, theorem):
            self.calls.append(theorem)
            if 'shows "False"' in theorem:
                return {"success": False, "incomplete": False}
            if "assumes" in theorem:
                return {"success": injected in theorem, "incomplete": False}
            return {"success": False, "incomplete": False}

    check = ProvesOnlyWithInjected()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
        general_conclusions=(injected,),
    )

    assert rewarded is True
    assert reason == "rewarded"
    assert all(injected in c for c in check.calls if "assumes" in c)


def test_invented_claim_numeral_is_rejected():
    """Numeral provenance: a quantity the step's own text never states cannot enter the claim (the entity anchor exempts every digit token, so without this check the translator could pin any value)."""
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G"],
        r"order G = (15::nat)",
        "The step states that a generates the whole group.",
        check,
    )

    assert rewarded is False
    assert reason == "invented_numeral:15"
    assert check.calls == []


def test_invented_premise_numeral_is_rejected():
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"card (rcosets (\<one>)) = (8::nat)", r"a \<in> carrier G"],
        r"inv a = a",
        "The step states that a is its own inverse in group G.",
        check,
    )

    assert rewarded is False
    assert reason == "invented_premise_numeral:8"
    assert check.calls == []


def test_numeral_stated_in_the_step_text_is_allowed():
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G"],
        r"order G = (15::nat)",
        "The step states that a generates G and that the order of G is 15.",
        check,
    )

    assert rewarded is True
    assert reason == "rewarded"


def test_premise_free_battery_uses_every_claim_tactic():
    """A claim only metis can prove premise-free is exactly as vacuous as one simp can: the negative check must run the full battery, not the freecheck alone."""

    class MetisFreeProof:
        def __init__(self):
            self.calls = []

        def __call__(self, theorem):
            self.calls.append(theorem)
            if 'shows "False"' in theorem:
                return {"success": False, "incomplete": False}
            if "assumes" in theorem:
                return {"success": True, "incomplete": False}
            return {"success": "metis" in theorem, "incomplete": False}

    check = MetisFreeProof()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G", r"b \<in> carrier G"],
        _valid_claim(),
        "The step states that a and b are elements of group G.",
        check,
    )

    assert rewarded is False
    assert reason == "trivial"


def test_claim_covered_by_a_premise_conjunct_is_rejected():
    """Coverage, not exact equality: `assume C \<and> D, show C` proves nothing (C is fully assumed), and so does a claim reassembled from conjuncts spread over the injected chain."""
    check = RecordingCheck()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"inv a = \<one> \<and> a \<in> carrier G"],
        r"inv a = \<one>",
        "The step states that the inverse of a is the identity in the group.",
        check,
    )
    assert (rewarded, reason) == (False, "claim_repeated_as_premise")
    assert check.calls == []

    check = RecordingCheck()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"b \<in> carrier G"],
        r"inv a = \<one> \<and> a \<in> carrier G",
        "The step restates the two earlier conclusions about a.",
        check,
        previous_conclusions=(r"inv a = \<one>", r"a \<in> carrier G"),
    )
    assert (rewarded, reason) == (False, "claim_repeats_chain")
    assert check.calls == []


def test_restatement_guards_catch_textual_disguises():
    """A swapped equality restating a premise and a claim restating an injected conclusion conjoined with True are the same restatements in disguise; collapsed-whitespace equality alone missed both."""
    check = RecordingCheck()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"\<one> = inv a"],
        r"inv a = \<one>",
        "The step states that the inverse of a is the identity.",
        check,
    )
    assert (rewarded, reason) == (False, "claim_repeated_as_premise")
    assert check.calls == []

    check = RecordingCheck()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"a \<in> carrier G"],
        r"inv a = \<one>",
        "The step restates that the inverse of a is the identity.",
        check,
        previous_conclusions=(r"inv a = \<one> \<and> True",),
    )
    assert (rewarded, reason) == (False, "claim_repeats_chain")
    assert check.calls == []


def test_subgroup_names_are_not_vocabulary():
    """H, K, N are problem entities (a subgroup, a kernel), not locale carriers: they must appear in the step's own text like any other entity."""
    check = RecordingCheck()

    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"subgroup H G"],
        r"card (rcosets H) = order G",
        "The step states that the cosets partition the group.",   # no H mentioned
        check,
    )

    assert rewarded is False
    assert reason == "invented:H"

    check = RecordingCheck()
    rewarded, reason = direct_verify.verify(
        direct_verify.GROUP_SPEC,
        [r"subgroup H G"],
        r"card (rcosets H) = order G",
        "The step states that the cosets of the subgroup H partition G.",
        check,
    )
    assert rewarded is True


def test_domain_matching_is_specific_and_arithmetic_is_not_direct():
    assert direct_verify.match_domain(r"a \<otimes> b = b \<otimes> a") is direct_verify.GROUP_SPEC
    assert direct_verify.match_domain(r"a \<oplus> b = b \<oplus> a") is direct_verify.RING_SPEC
    assert direct_verify.match_domain(
        r"a \<otimes> inv a = \<one>",
        r"a \<noteq> \<zero>",
    ) is direct_verify.FIELD_SPEC
    assert direct_verify.match_domain("x + 1 == 2") is None


def test_domain_matching_generalizations():
    """Routing gaps from the review: a standalone arithmetic-valued group claim (order G, ord a) must route direct, and `carrier R` identifies the algebra family even in a purely multiplicative statement; the general path's bridged pyexpr forms must keep routing general."""
    assert direct_verify.match_domain("order G = 15") is direct_verify.GROUP_SPEC
    assert direct_verify.match_domain("ord a = 6") is direct_verify.GROUP_SPEC
    assert direct_verify.match_domain(
        r"a \<otimes> b = b \<otimes> a", r"a \<in> carrier R") is direct_verify.RING_SPEC
    # the bridged general-side equation and a variable merely named ord stay general
    assert direct_verify.match_domain("order_G == 15") is None
    assert direct_verify.match_domain("ord == 6") is None
