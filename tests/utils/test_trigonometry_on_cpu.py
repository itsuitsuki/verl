"""CPU-only structure tests for the general trigonometry system (trigonometry.py).

Pins the extraction normalizations (equality and inequality orientation, comparison chains, conjunction splitting, compound canonical angles, exact rational-pi bounds with radian rejection),
the conflicting-value ambiguity rule,
the quadrant sign provider's strict and fail-closed behavior,
the kernel-checked square-root candidates,
the bounded deterministic planner,
and the single-renderer twin identity (positive and False theorems differ ONLY in the shows line).
Pure python: no prover, no pool.
"""
from fractions import Fraction

from verl.utils.isabelle_utils import state_classes, theorem_builders, trigonometry
from verl.utils.isabelle_utils.pyexpr import parse_expr

VT = {"a": "real", "b": "real", "x": "real"}


def _src(name, source, kind=state_classes.PremiseSource.STEP):
    return state_classes.AdmittedPyExprSource(
        source_kind=kind, name=name, pyexpr_source=source, isabelle_term=source)


def _context(*sources):
    return trigonometry.extract_trig_context(sources, VT)


def test_value_evidence_normalizes_equality_orientation():
    context = _context(_src("p0", "tan(a) == 3/4"), _src("p1", "3/4 == tan(a)"))
    assert len(context.values) == 2
    assert len({(v.function, v.angle_key, v.value_term) for v in context.values}) == 1
    assert context.ambiguous_value_keys == ()   # same value both ways is not a conflict


def test_conflicting_values_become_ambiguous_never_a_choice():
    context = _context(_src("p0", "tan(a) == 3/4"), _src("p1", "tan(a) == 5/4"))
    assert context.values == ()
    assert len(context.ambiguous_value_keys) == 1
    attempts = trigonometry.plan_attempts(context, "sin pv_a = x", {})
    assert all(not a.premise_dependent for a in attempts)   # no value-based attempt


def test_bound_evidence_normalizes_inequality_orientation_chains_and_conjunctions():
    for sources in ((_src("p0", "pi/2 < a"), _src("p1", "a < pi")),
                    (_src("p0", "a > pi/2"), _src("p1", "pi > a")),
                    (_src("p0", "pi/2 < a < pi"),),
                    (_src("p0", "(pi/2 < a) and (a < pi)"),)):
        context = _context(*sources)
        sides = {(b.side, b.coefficient) for b in context.pi_bounds}
        assert sides == {("lower", Fraction(1, 2)), ("upper", Fraction(1))}, sources


def test_exact_pi_coefficient_and_radian_rejection():
    assert trigonometry.exact_pi_coefficient(parse_expr("pi / 2")) == Fraction(1, 2)
    assert trigonometry.exact_pi_coefficient(parse_expr("3 * pi / 2")) == Fraction(3, 2)
    assert trigonometry.exact_pi_coefficient(parse_expr("2 * pi")) == Fraction(2)
    assert trigonometry.exact_pi_coefficient(parse_expr("-3 * pi / 2")) == Fraction(-3, 2)
    assert trigonometry.exact_pi_coefficient(parse_expr("0")) == Fraction(0)
    assert trigonometry.exact_pi_coefficient(parse_expr("1")) is None
    assert trigonometry.exact_pi_coefficient(parse_expr("pi + 1")) is None


def test_compound_angles_are_distinct_canonical_keys():
    context = _context(_src("p0", "sin(pi - a) == 1/2"), _src("p1", "tan(a) == 3/4"),
                       _src("p2", "0 < a"), _src("p3", "a < pi / 2"))
    keys = {v.angle_key for v in context.values}
    assert len(keys) == 2                       # `pi - a` never collapses onto `a`
    signs = trigonometry._quadrant_signs(context)
    assert len(signs) == 1                      # bounds attach to the bare `a` key only


def test_algebraically_equal_angles_share_one_canonical_key():
    """The angle key is the exact linear form,
    so the spellings 2*a, a*2, and a+a join their evidence (bounds admitted under one spelling serve a value admitted under another)
    and conflicting values across spellings are detected as ONE ambiguous key.
    Distinct linear forms are distinct real functions, so nothing else merges."""
    keys = {trigonometry._angle_key(parse_expr(s)) for s in ("2*a", "a*2", "a+a")}
    assert len(keys) == 1
    assert (trigonometry._angle_key(parse_expr("2*a"))
            != trigonometry._angle_key(parse_expr("2*b")))
    context = _context(_src("p0_0", "tan(a*2) == 3/4"),
                       _src("p0_1", "0 < 2*a"), _src("p0_2", "2*a < pi/2"))
    signs = trigonometry._quadrant_signs(context)
    assert set(signs) == {v.angle_key for v in context.values}
    conflict = _context(_src("p0", "tan(2*a) == 3/4"), _src("p1", "tan(a+a) == 5/4"))
    assert conflict.values == ()
    assert len(conflict.ambiguous_value_keys) == 1


def test_sign_evidence_orientations():
    context = _context(_src("p0", "cos(a) < 0"), _src("p1", "0 < sin(a)"))
    parsed = {(s.function, s.positive) for s in context.signs}
    assert parsed == {("cos", False), ("sin", True)}


def test_quadrant_sign_provider_is_strict_and_fail_closed():
    strict = _context(_src("p0", "pi < a"), _src("p1", "a < 3 * pi / 2"))
    signs = trigonometry._quadrant_signs(strict)
    entry = signs[next(iter(signs))]
    assert entry.cos_positive is False and entry.sin_positive is False
    # a non-strict endpoint contributes nothing (the endpoint leaves the sign open)
    endpoint = _context(_src("p0", "0 <= a"), _src("p1", "a < pi / 2"))
    assert trigonometry._quadrant_signs(endpoint) == {}
    # an empty interval yields nothing rather than a quadrant match
    empty = _context(_src("p0", "pi < a"), _src("p1", "a < pi / 2"))
    assert trigonometry._quadrant_signs(empty) == {}
    # an interval crossing quadrants yields nothing
    wide = _context(_src("p0", "0 < a"), _src("p1", "a < pi"))
    assert trigonometry._quadrant_signs(wide) == {}


def test_square_root_candidates_are_exact_or_absent():
    assert trigonometry._hypotenuse_root(parse_expr("3/4")) == Fraction(5, 4)
    assert trigonometry._hypotenuse_root(parse_expr("sqrt(3)")) == Fraction(2)
    assert trigonometry._hypotenuse_root(parse_expr("2/7")) is None
    assert trigonometry._unit_complement_root(parse_expr("3/5")) == Fraction(4, 5)
    assert trigonometry._unit_complement_root(parse_expr("1/3")) is None
    assert trigonometry._unit_complement_root(parse_expr("2")) is None


def _full_context():
    return _context(_src("p0_0", "tan(a) == 3/4"),
                    _src("p0_1", "0 < a"), _src("p0_2", "a < pi / 2"))


def _name_to_term():
    return {"p0_0": "((tan pv_a) = ((3::real) / (4::real)))",
            "p0_1": "((0::real) < pv_a)", "p0_2": "(pv_a < pi / (2::real))"}


def test_planner_is_deterministic_ordered_and_bounded():
    context = _full_context()
    first = trigonometry.plan_attempts(context, "sin pv_a = x", _name_to_term())
    second = trigonometry.plan_attempts(context, "sin pv_a = x", _name_to_term())
    assert first == second
    assert 0 < len(first) <= trigonometry.MAX_ATTEMPTS
    premise_free = [a for a in first if not a.premise_dependent]
    assert first[:len(premise_free)] == tuple(premise_free)   # premise-free groups first
    assert any(a.name == "TAN_VALUE_FROM_QUADRANT" for a in first)
    # a goal without any trig application plans nothing
    assert trigonometry.plan_attempts(context, "pv_x = (3::real)", _name_to_term()) == ()


def test_relevance_orders_claim_evidence_ahead_of_the_cap():
    """Evidence attempts whose relevance_terms miss the goal drop behind those that hit it BEFORE the MAX_ATTEMPTS cap,
    so many unrelated tangent premises can no longer crowd the claim's own family out of the window
    (measured: 14 unrelated tangent premises made provability depend on premise order).
    The variable names here are prefixes of one another (x1, x11, ...), pinning the identifier-SET matching:
    a substring test counted every shorter name as a hit on the longest name's goal and cut the target family anyway (measured)."""
    names = ["x" + "1" * (i + 1) for i in range(14)]
    vt = {name: "real" for name in names}
    sources = []
    for i, name in enumerate(names):
        sources.append(_src(f"p0_{3 * i}", f"tan({name}) == 3/4"))
        sources.append(_src(f"p0_{3 * i + 1}", f"0 < {name}"))
        sources.append(_src(f"p0_{3 * i + 2}", f"{name} < pi/2"))
    name_to_term = {source.name: source.isabelle_term for source in sources}
    context = trigonometry.extract_trig_context(sources, vt)
    target = f"pv_{names[-1]}"
    goal = f"sin {target} = ((3::real) / (5::real))"
    attempts = trigonometry.plan_attempts(context, goal, name_to_term)
    assert 0 < len(attempts) <= trigonometry.MAX_ATTEMPTS
    assert any(target in a.relevance_terms for a in attempts)
    # the ordering is a clean partition:
    # every goal-hitting (or always-relevant) attempt sits ahead of every goal-missing one
    goal_identifiers = theorem_builders.identifiers(goal)
    hits = [trigonometry._goal_relevant(a, goal_identifiers) for a in attempts]
    assert hits == sorted(hits, reverse=True)
    # and ONLY the target's own family hits; the prefix-colliding names all miss
    relevant_evidence = [a for a in attempts if a.relevance_terms
                         and trigonometry._goal_relevant(a, goal_identifiers)]
    assert relevant_evidence
    assert all(a.relevance_terms == (target,) for a in relevant_evidence)


def test_guarded_attempt_carries_sign_chain_and_kernel_checked_root():
    context = _full_context()
    attempts = trigonometry.plan_attempts(context, "sin pv_a = x", _name_to_term())
    guarded = next(a for a in attempts if a.name == "TAN_VALUE_FROM_QUADRANT")
    fact_names = [fact.name for fact in guarded.auxiliary_facts]
    assert fact_names[:4] == ["cs", "sn", "nz", "ts"]     # the quadrant sign chain
    assert "ms" in fact_names and "mc" in fact_names       # sin AND cos value facts
    root_fact = next(fact for fact in guarded.auxiliary_facts if fact.name == "dr")
    assert "real_sqrt_unique" in root_fact.proof           # Python candidate, kernel checked
    premise_names = [name for name, _ in guarded.premises]
    assert premise_names == ["p0_0", "p0_1", "p0_2"]


def test_renderer_twin_is_byte_identical_except_the_goal():
    context = _full_context()
    attempts = trigonometry.plan_attempts(context, "sin pv_a = x", _name_to_term())
    guarded = next(a for a in attempts if a.premise_dependent)
    fixes = [("pv_a", "real"), ("pv_x", "real")]
    goal = "sin pv_a = pv_x"
    positive = trigonometry.render_attempt(fixes, goal, guarded)
    twin = trigonometry.render_attempt(fixes, "False", guarded)
    marker_positive = f'  shows "{goal}"\n'
    marker_twin = '  shows "False"\n'
    head_p, _, tail_p = positive.partition(marker_positive)
    head_t, _, tail_t = twin.partition(marker_twin)
    assert head_p == head_t and tail_p == tail_t
    assert tail_p                                          # the split actually happened


def test_attempt_fixes_are_minimal_and_sorted():
    context = _full_context()
    attempts = trigonometry.plan_attempts(context, "sin pv_a = x", _name_to_term())
    guarded = next(a for a in attempts if a.premise_dependent)
    step_fixes = [("pv_a", "real"), ("pv_unused", "real"), ("pv_x", "real")]
    fixes = trigonometry.attempt_fixes(step_fixes, "sin pv_a = pv_x", guarded)
    assert ("pv_unused", "real") not in fixes
    assert fixes == sorted(fixes)


def test_explicit_sign_premise_variant_normalizes_the_sign_fact():
    context = _context(_src("p0_0", "tan(a) == 3/4"), _src("p0_1", "cos(a) > 0"))
    attempts = trigonometry.plan_attempts(
        context, "sin pv_a = x",
        {"p0_0": "((tan pv_a) = ((3::real) / (4::real)))",
         "p0_1": "((cos pv_a) > (0::real))"})
    guarded = next(a for a in attempts if a.name == "TAN_VALUE_FROM_SIGN")
    cs = guarded.auxiliary_facts[0]
    assert cs.name == "cs" and cs.proposition == "0 < cos pv_a"
    assert "p0_1" in cs.proof


def test_pythagorean_value_attempt_uses_the_session_lemmas():
    context = _context(_src("p0_0", "sin(a) == 3/5"),
                       _src("p0_1", "pi / 2 < a"), _src("p0_2", "a < pi"))
    attempts = trigonometry.plan_attempts(
        context, "cos pv_a = x",
        {"p0_0": "((sin pv_a) = ((3::real) / (5::real)))",
         "p0_1": "(pi / (2::real) < pv_a)", "p0_2": "(pv_a < pi)"})
    guarded = next(a for a in attempts if a.name == "COS_VALUE_FROM_QUADRANT")
    meta = next(fact for fact in guarded.auxiliary_facts if fact.name == "mo")
    assert "cos_from_sin_cneg" in meta.proof               # Q2: cosine negative
    root = next(fact for fact in guarded.auxiliary_facts if fact.name == "dr")
    assert "real_sqrt_unique" in root.proof


def test_tangent_definedness_conditions_cover_every_tan_application():
    """HOL totalizes tan (tan(pi/2) evaluates to 0 in the kernel),
    so every value-sensitive tangent application in a claim yields its transpiled angle term,
    from which the verifier builds and proves cos != 0 before any claim attempt;
    duplicates collapse, non-tangent claims yield nothing."""
    claims = [parse_expr("tan(pi / 2) == 0"),
              parse_expr("tan(a) == tan(a)")]
    angles = trigonometry.tangent_definedness_conditions(claims, VT)
    # the constant-angle value pin requires its cosine-nonzero condition;
    # the second claim is the trivial periodicity rewrite (u - v = 0 * pi) and stays exempt
    assert len(angles) == 1
    assert "pi" in angles[0][0]
    assert angles[0][1] is None          # base pi/2 carries no decomposition
    # a non-base literal (tan(5*pi/4)) carries its kernel-checkable decomposition,
    # so the cosine-nonzero check can prove cos(5*pi/4) != 0 premise-free
    ((_angle, decomp),) = trigonometry.tangent_definedness_conditions(
        [parse_expr("tan(5 * pi / 4) == 1")], VT)
    assert decomp == "(pi + (pi / (4::real)))"
    assert trigonometry.tangent_definedness_conditions(
        [parse_expr("sin(a) == 1")], VT) == ()
    # value pinning always requires its condition:
    # trig-free other side, constant angles (including a disguised constant value), and comparison chains
    for value_pin in ("tan(a) == 0", "tan(pi / 2) == sin(pi / 2) * 0",
                      "0 < tan(a) < 1"):
        assert trigonometry.tangent_definedness_conditions(
            [parse_expr(value_pin)], VT) != ()


def test_tangent_identity_exemption_is_structural_not_any_other_side_trig():
    """The identity exemption matches the whole equation structurally:
    periodicity (u - v an integer pi multiple), odd/reflection (u + v an integer pi multiple, negated side), and the same-angle definition unfold.
    The earlier per-side test ("does the other side carry a variable-angle trig application") also exempted value pins like tan(a) == sin(b),
    where premises a == pi/2 and b == 0 make HOL prove the totalized 0 == 0 (the audited FP door)."""
    for identity in ("tan(-a) == -tan(a)", "tan(a + pi) == tan(a)",
                     "tan(a - pi) == tan(a)", "tan(a + 2 * pi) == tan(a)",
                     "tan(pi - a) == -tan(a)", "-tan(a) == tan(pi - a)",
                     "tan(a) == sin(a) / cos(a)"):
        assert trigonometry.tangent_definedness_conditions(
            [parse_expr(identity)], VT) == (), identity
    # the audited FP shapes now REQUIRE their conditions
    require = {
        "tan(a) == sin(b)": 1, "tan(a) == cos(b)": 1, "tan(a) == tan(b)": 2,
        "tan(a) == cos(a)": 1,
        # the tan_double formula:
        # at a premise-pinned singular angle both sides totalize to 0 (x/0 = 0 in HOL), so it is NOT exempt
        "tan(2*a) == 2 * tan(a) / (1 - tan(a)**2)": 2,
        # near-identities that fail the structural match stay conditioned (fail closed)
        "tan(-a) == tan(a)": 2, "tan(a + pi/2) == tan(a)": 2,
        "tan(a) == tan(a) + 0": 1,
    }
    for source, count in require.items():
        angles = trigonometry.tangent_definedness_conditions([parse_expr(source)], VT)
        assert len(angles) == count, (source, angles)
    # constant angles never qualify as identities:
    # tan(pi/2 + pi) == tan(pi/2) is HOL-provable as 0 == 0 but mathematically undefined
    assert len(trigonometry.tangent_definedness_conditions(
        [parse_expr("tan(pi/2 + pi) == tan(pi/2)")], VT)) == 2


def test_tangent_conditions_scan_model_authored_assumptions():
    """Definitions, admitted step premises, and earlier chain conclusions enter the theorem as assumptions,
    so a tangent pinned at a singular angle must yield its condition from there too
    (the audited bypasses: `x == tan(a)` splitting into a definition with a tan-free residual claim, and a REJECTED `x == tan(pi/2)` step still entering the chain as s*)."""
    claim = [parse_expr("x == 0")]
    definition_bypass = trigonometry.tangent_definedness_conditions(
        claim, VT, assumption_nodes=[parse_expr("x == tan(a)")])
    assert definition_bypass == (("pv_a", None),)
    chain_bypass = trigonometry.tangent_definedness_conditions(
        claim, VT, assumption_nodes=[parse_expr("x == tan(pi / 2)")])
    assert chain_bypass == (("(pi / (2::real))", None),)
    # an inherited identity rewrite stays exempt on the assumption side too
    assert trigonometry.tangent_definedness_conditions(
        claim, VT, assumption_nodes=[parse_expr("tan(-a) == -tan(a)")]) == ()
    # a NONZERO tangent value assumption self-certifies (in HOL, cos u = 0 forces tan u = 0, contradicting the nonzero value),
    # so chains over tan(x) == 3/4 do not re-demand a quadrant proof at every step;
    # zero and claim-side values still require
    assert trigonometry.tangent_definedness_conditions(
        claim, VT, assumption_nodes=[parse_expr("tan(x) == 3/4")]) == ()
    assert trigonometry.tangent_definedness_conditions(
        claim, VT, assumption_nodes=[parse_expr("tan(x) == sqrt(3)")]) == ()
    assert trigonometry.tangent_definedness_conditions(
        claim, VT, assumption_nodes=[parse_expr("tan(x) == 0")]) == (("pv_x", None),)
    assert trigonometry.tangent_definedness_conditions(
        [parse_expr("tan(x) == 3/4")], VT) == (("pv_x", None),)


def test_cosine_nonzero_theorem_twin_shares_everything_but_the_goal():
    """The cosine-nonzero check's quadrant sign theorem and its False guard twin come from ONE renderer with only the goal swapped (the equal-strength doctrine's byte identity)."""
    fixes = [("pv_a", "real")]
    premises = [("p0_1", "((0::real) < pv_a)"), ("p0_2", "(pv_a < pi / (4::real))")]
    angle = "((2::real) * pv_a)"
    positive = trigonometry.cosine_nonzero_theorem(
        fixes, premises, angle, "cos_pos_q1", True, f"cos {angle} \\<noteq> 0")
    twin = trigonometry.cosine_nonzero_theorem(
        fixes, premises, angle, "cos_pos_q1", True, "False")
    assert 'have s: "0 < cos ((2::real) * pv_a)"' in positive
    assert "by (intro cos_pos_q1; linarith)" in positive
    marker = "  shows "
    p_head, p_tail = positive.split(marker, 1)
    t_head, t_tail = twin.split(marker, 1)
    assert p_head == t_head
    assert p_tail.split("\n", 1)[1] == t_tail.split("\n", 1)[1]   # identical bodies


def test_new_premise_free_groups_are_present_and_orientation_separated():
    names = [name for name, _ in trigonometry.PREMISE_FREE_GROUPS]
    for required in ("PRODUCT_TO_SUM", "SUM_TO_PRODUCT",
                     "DOUBLE_ANGLE_COS_FORM", "DOUBLE_ANGLE_SIN_FORM"):
        assert required in names
    tactics_by_name = dict(trigonometry.PREMISE_FREE_GROUPS)
    # mutual inverse families never share one simp set (rewrite cycles)
    assert "sin_plus_sin" not in tactics_by_name["PRODUCT_TO_SUM"]
    assert "sin_times_sin" not in tactics_by_name["SUM_TO_PRODUCT"]
    assert "cos_double_sin" not in tactics_by_name["DOUBLE_ANGLE_COS_FORM"]
    # algebra_simps commutes `2 * x` into `x * 2` before the double-angle rules can match
    assert "algebra_simps" not in tactics_by_name["DOUBLE_ANGLE_COS_FORM"]
    assert "algebra_simps" not in tactics_by_name["DOUBLE_ANGLE_SIN_FORM"]


def test_isqrt_exact_is_pure_integer_arithmetic():
    """A float detour returns wrong roots around 100 digits and overflows around 400; math.isqrt must keep giant literals exact and crash-free (trig planning runs before the dangerous-term check)."""
    big_root = 10 ** 200 + 12345
    assert trigonometry._isqrt_exact(big_root * big_root) == big_root
    assert trigonometry._isqrt_exact(big_root * big_root + 1) is None
    assert trigonometry._isqrt_exact(-4) is None
    assert trigonometry._isqrt_exact(0) == 0


def test_period_shifted_bounds_reach_the_quadrant_through_a_transfer_chain():
    """Bounds one full period above or below [0, 2*pi) still yield sign facts: the chain proves csr/snr on the shifted term (a -/+ 2*pi, whose bounds linarith derives from the assumptions) and transfers each sign onto the original angle through the angle-addition expansion."""
    context = _context(_src("p0_0", "tan(a) == -3/4"),
                       _src("p0_1", "5 * pi / 2 < a"), _src("p0_2", "a < 3 * pi"))
    attempts = trigonometry.plan_attempts(
        context, "sin pv_a = x",
        {"p0_0": "((tan pv_a) = (- ((3::real) / (4::real))))",
         "p0_1": "((5::real) * pi / (2::real) < pv_a)",
         "p0_2": "(pv_a < (3::real) * pi)"})
    guarded = next(a for a in attempts if a.name == "TAN_VALUE_FROM_QUADRANT")
    fact_names = [fact.name for fact in guarded.auxiliary_facts]
    assert fact_names[:6] == ["csr", "snr", "cs", "sn", "nz", "ts"]
    csr = guarded.auxiliary_facts[0]
    assert "- (2::real) * pi" in csr.proposition       # sign proved on the shifted term
    assert "cos_neg_q2" in csr.proof                    # Q2 after the +1 period shift
    assert guarded.auxiliary_facts[2].proposition == "cos pv_a < 0"
    # one period below: -3*pi/2 < a < -pi is Q2 shifted by -1 (transfer adds 2*pi)
    below = _context(_src("p0_0", "sin(a) == 3/5"),
                     _src("p0_1", "-3 * pi / 2 < a"), _src("p0_2", "a < -pi"))
    attempts = trigonometry.plan_attempts(
        below, "cos pv_a = x",
        {"p0_0": "((sin pv_a) = ((3::real) / (5::real)))",
         "p0_1": "(- ((3::real) * pi / (2::real)) < pv_a)",
         "p0_2": "(pv_a < - pi)"})
    guarded = next(a for a in attempts if a.name == "COS_VALUE_FROM_QUADRANT")
    assert "+ (2::real) * pi" in guarded.auxiliary_facts[0].proposition
    # a non-strict endpoint never gains signs, shifted or not
    endpoint = _context(_src("p0_0", "tan(a) == 3/4"),
                        _src("p0_1", "5 * pi / 2 <= a"), _src("p0_2", "a < 3 * pi"))
    assert trigonometry._quadrant_signs(endpoint) == {}


def test_literal_angle_decomposition_shapes_and_planner_emission():
    """5*pi/6 reflects off pi, 7*pi/4 off 2*pi, 13*pi/6 shifts one period down; base angles the plain groups already evaluate emit nothing. The attempt assumes nothing, so it stays premise-free-classified (no guard twin needed)."""
    frac = Fraction
    assert trigonometry._decomposed_pi_term(frac(5, 6)) == "(pi - (pi / (6::real)))"
    assert trigonometry._decomposed_pi_term(frac(7, 4)) == \
        "((2::real) * pi - (pi / (4::real)))"
    assert trigonometry._decomposed_pi_term(frac(13, 6)) == \
        "((2::real) * pi + (pi / (6::real)))"
    assert trigonometry._decomposed_pi_term(frac(-1, 3)) == \
        "(((2::real) * pi - (pi / (3::real))) - (2::real) * pi)"
    # multiple whole periods nest one layer per period
    # (a flat 4*pi would strand the expansion: the default simpset evaluates sin/cos only at 2*pi itself)
    assert trigonometry._decomposed_pi_term(frac(9, 2)) == \
        "((2::real) * pi + ((2::real) * pi + (pi / (2::real))))"
    assert trigonometry._decomposed_pi_term(frac(100)) is None   # beyond the period cap
    claim = parse_expr("sin(5 * pi / 6) == 1/2")
    attempts = trigonometry.plan_attempts(
        _context(), "sin ((5::real) * pi / (6::real)) = x", {}, claim_node=claim)
    literal = next(a for a in attempts if a.name == "LITERAL_ANGLE")
    assert not literal.premise_dependent
    assert literal.premises == ()
    assert literal.auxiliary_facts[0].name == "e0"
    assert "= (pi - (pi / (6::real)))" in literal.auxiliary_facts[0].proposition
    base = parse_expr("sin(pi / 6) == 1/2")
    attempts = trigonometry.plan_attempts(
        _context(), "sin (pi / (6::real)) = x", {}, claim_node=base)
    assert not any(a.name == "LITERAL_ANGLE" for a in attempts)


def test_multi_period_interval_nests_the_shifted_term():
    """9*pi/2 < a < 5*pi is Q2 shifted by two whole periods: the sign is proved on the doubly nested term ((a - 2*pi) - 2*pi) and transferred one layer per simp pass."""
    context = _context(_src("p0_0", "tan(a) == -3/4"),
                       _src("p0_1", "9 * pi / 2 < a"), _src("p0_2", "a < 5 * pi"))
    attempts = trigonometry.plan_attempts(
        context, "sin pv_a = x",
        {"p0_0": "((tan pv_a) = (- ((3::real) / (4::real))))",
         "p0_1": "((9::real) * pi / (2::real) < pv_a)",
         "p0_2": "(pv_a < (5::real) * pi)"})
    guarded = next(a for a in attempts if a.name == "TAN_VALUE_FROM_QUADRANT")
    csr = guarded.auxiliary_facts[0]
    assert csr.name == "csr"
    assert csr.proposition.count("- (2::real) * pi") == 2
    assert "cos_neg_q2" in csr.proof


def test_tan_double_supports_sqrt_valued_tangents():
    """t = sqrt(3) has rational t^2, so the doubled value is the exact rational coefficient 2/(1-3) = -1 times the premise's own rendered term; t^2 = 1 stays excluded (singular denominator)."""
    sqrt_value = _context(_src("p0_0", "tan(a) == sqrt(3)"),
                          _src("p0_1", "pi / 4 < a"), _src("p0_2", "a < pi / 2"))
    name_to_term = {"p0_0": "((tan pv_a) = (sqrt ((3::real))))",
                    "p0_1": "(pi / (4::real) < pv_a)",
                    "p0_2": "(pv_a < pi / (2::real))"}
    attempts = trigonometry.plan_attempts(sqrt_value, "tan ((2::real) * pv_a) = x",
                                          name_to_term)
    double = next(a for a in attempts if a.name == "TAN_DOUBLE_FROM_QUADRANT")
    mv = double.auxiliary_facts[-1]
    assert mv.name == "mv"
    assert "(-1::real) * (" in mv.proposition and "sqrt" in mv.proposition
    singular = _context(_src("p0_0", "tan(a) == 1"),
                        _src("p0_1", "0 < a"), _src("p0_2", "a < pi / 4"))
    attempts = trigonometry.plan_attempts(
        singular, "tan ((2::real) * pv_a) = x",
        {"p0_0": "((tan pv_a) = (1::real))",
         "p0_1": "((0::real) < pv_a)", "p0_2": "(pv_a < pi / (4::real))"})
    assert not any(a.name == "TAN_DOUBLE_FROM_QUADRANT" for a in attempts)


def test_inverse_literal_attempts_resolve_special_values():
    """arcsin(1/2) = pi/6 goes through a kernel-checked value equality (1/2 = sin(pi/6)) unfolded into composition shape; odd functions map negative arguments to negated principal angles; non-special arguments emit nothing."""
    claim = parse_expr("arcsin(1/2) == pi / 6")
    attempts = trigonometry.plan_attempts(
        _context(), "arcsin ((1::real) / (2::real)) = (pi / (6::real))", {},
        claim_node=claim)
    literal = next(a for a in attempts if a.name == "INVERSE_LITERAL")
    assert not literal.premise_dependent
    assert literal.unfold_fact_names == ("e0",)
    assert [f.name for f in literal.auxiliary_facts] == ["e0", "ba0", "bb0", "m0"]
    assert "= sin (pi / (6::real))" in literal.auxiliary_facts[0].proposition
    assert "sin_30" in literal.auxiliary_facts[0].proof
    # the range bounds are explicit linarith facts consumed by DIRECT rule instantiation:
    # simp's conditional-rewrite solver discharges them neither from the simpset nor from chained facts (measured on the real pool)
    assert literal.auxiliary_facts[1].proof == "using pi_gt_zero by linarith"
    assert literal.auxiliary_facts[3].proof == "by (rule arcsin_sin[OF ba0 bb0])"
    negative = parse_expr("arcsin(-1/2) == -pi / 6")
    attempts = trigonometry.plan_attempts(
        _context(), "arcsin (- ((1::real) / (2::real))) = x", {}, claim_node=negative)
    literal = next(a for a in attempts if a.name == "INVERSE_LITERAL")
    assert "= sin (- (pi / (6::real)))" in literal.auxiliary_facts[0].proposition
    plain = parse_expr("arcsin(3/5) == a")
    attempts = trigonometry.plan_attempts(
        _context(), "arcsin ((3::real) / (5::real)) = pv_a", {}, claim_node=plain)
    assert not any(a.name == "INVERSE_LITERAL" for a in attempts)


def test_arctan_premise_and_quadratic_tan_double():
    """An admitted arctan(x) == pi/6 premise yields the guarded ARCTAN_PREMISE_VALUE attempt (tan_arctan is unconditional, so x = tan(pi/6) needs no range assumption; arcsin/arccos premises pin nothing in HOL and emit no attempt); a quadratic-irrational tangent 1 + sqrt(2) gets its doubled value -1 through exact Q(sqrt 2) conjugate division, with the sq fact for the kernel."""
    context = _context(_src("p0_0", "arctan(x) == pi / 6"))
    attempts = trigonometry.plan_attempts(context, "pv_x = y",
                                          {"p0_0": "((arctan pv_x) = (pi / (6::real)))"})
    premise_value = next(a for a in attempts if a.name == "ARCTAN_PREMISE_VALUE")
    assert [f.name for f in premise_value.auxiliary_facts] == ["mt", "mx", "mv"]
    # rationalized value with the measured working lemma set
    # (1/sqrt 3 cannot be bridged to the claim's sqrt(3)/3 spelling by the final tactic)
    assert "sqrt 3 / 3" in premise_value.auxiliary_facts[0].proposition
    assert premise_value.auxiliary_facts[0].proof.startswith("by (simp add: tan_def sin_30")
    assert "metis tan_arctan" in premise_value.auxiliary_facts[1].proof
    arcsin_premise = _context(_src("p0_0", "arcsin(x) == pi / 6"))
    assert not any(a.name == "ARCTAN_PREMISE_VALUE"
                   for a in trigonometry.plan_attempts(
                       arcsin_premise, "pv_x = y",
                       {"p0_0": "((arcsin pv_x) = (pi / (6::real)))"}))
    assert trigonometry._quadratic_parts(parse_expr("1 + sqrt(2)")) == \
        (Fraction(1), Fraction(1), Fraction(2))
    rendered, sq_info, den_info = trigonometry._tan_double_value_term(
        parse_expr("1 + sqrt(2)"), "((1::real) + (sqrt ((2::real))))")
    assert rendered == "(-1::real)"
    assert sq_info == ("(sqrt ((2::real)))", 2)
    assert den_info == ("((-2::real) + (-2::real) * (sqrt ((2::real))))", True)


def test_tan_double_planner_requires_the_doubled_interval_to_fit_one_quadrant():
    """tan_double's hypotheses need cos != 0 for the angle AND its double, so the attempt exists only when the doubled interval also sits inside one principal quadrant; the mt fact consumes the kernel-proved nz and nzd."""
    fitting = _context(_src("p0_0", "tan(a) == 3/4"),
                       _src("p0_1", "0 < a"), _src("p0_2", "a < pi / 4"))
    name_to_term = {"p0_0": "((tan pv_a) = ((3::real) / (4::real)))",
                    "p0_1": "((0::real) < pv_a)",
                    "p0_2": "(pv_a < pi / (4::real))"}
    attempts = trigonometry.plan_attempts(fitting, "tan ((2::real) * pv_a) = x",
                                          name_to_term)
    double = next(a for a in attempts if a.name == "TAN_DOUBLE_FROM_QUADRANT")
    fact_names = [fact.name for fact in double.auxiliary_facts]
    assert fact_names[-4:] == ["csd", "nzd", "mt", "mv"]
    assert double.auxiliary_facts[-2].proof == "using nz nzd by (simp add: tan_double)"
    # mv restates the PREMISE-derived doubled value 2*(3/4)/(1-9/16) = 24/7 for the kernel
    assert "((24::real) / (7::real))" in double.auxiliary_facts[-1].proposition
    # 0 < a < pi/2 doubles to (0, pi), crossing quadrants:
    # no attempt, and the tangent definedness check keeps the claim failing closed
    crossing = _context(_src("p0_0", "tan(a) == 3/4"),
                        _src("p0_1", "0 < a"), _src("p0_2", "a < pi / 2"))
    attempts = trigonometry.plan_attempts(
        crossing, "tan ((2::real) * pv_a) = x",
        {"p0_0": name_to_term["p0_0"], "p0_1": name_to_term["p0_1"],
         "p0_2": "(pv_a < pi / (2::real))"})
    assert not any(a.name == "TAN_DOUBLE_FROM_QUADRANT" for a in attempts)


def test_inverse_trig_attempts_gate_on_arc_mentions_and_premises():
    """The premise-free conditional-rewrite group always plans; the guarded variant exists only when the goal mentions an inverse function AND there are premises to discharge the range side conditions from."""
    plain = trigonometry.plan_attempts(_context(), "sin pv_a = x",
                                       {"p0_0": "(pv_a < pi)"})
    assert any(a.name == "INVERSE_TRIG" for a in plain)
    assert not any(a.name == "INVERSE_TRIG_WITH_PREMISES" for a in plain)
    arc = trigonometry.plan_attempts(_context(), "arcsin (sin pv_a) = pv_a",
                                     {"p0_0": "(pv_a < pi / (2::real))"})
    guarded = next(a for a in arc if a.name == "INVERSE_TRIG_WITH_PREMISES")
    assert guarded.premise_dependent
    assert guarded.premises == (("p0_0", "(pv_a < pi / (2::real))"),)
    assert not any(a.name == "INVERSE_TRIG_WITH_PREMISES"
                   for a in trigonometry.plan_attempts(
                       _context(), "arcsin (sin pv_a) = pv_a", {}))


def test_arc_functions_parse_and_transpile():
    assert "arctan" in trigonometry._transpile(parse_expr("arctan(1) == pi / 4"), VT)
    assert "arcsin" in trigonometry._transpile(parse_expr("arcsin(x)"), VT)


def test_tan_from_values_requires_a_nonzero_closed_cosine():
    good = _context(_src("p0_0", "sin(a) == 3/5"), _src("p0_1", "cos(a) == 4/5"))
    attempts = trigonometry.plan_attempts(
        good, "tan pv_a = x",
        {"p0_0": "((sin pv_a) = ((3::real) / (5::real)))",
         "p0_1": "((cos pv_a) = ((4::real) / (5::real)))"})
    guarded = next(a for a in attempts if a.name == "TAN_FROM_VALUES")
    assert guarded.auxiliary_facts[0].name == "nz"         # cos != 0 precedes tan_def
    zero_cos = _context(_src("p0_0", "sin(a) == 1"), _src("p0_1", "cos(a) == 0"))
    attempts = trigonometry.plan_attempts(
        zero_cos, "tan pv_a = x",
        {"p0_0": "((sin pv_a) = (1::real))", "p0_1": "((cos pv_a) = (0::real))"})
    assert not any(a.name == "TAN_FROM_VALUES" for a in attempts)
