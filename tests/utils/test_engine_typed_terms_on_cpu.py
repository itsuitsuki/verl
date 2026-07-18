"""CPU-only semantic tests for typed Isabelle term construction and the theorem builders."""

from fractions import Fraction

from verl.utils.isabelle_utils import tactics, theorem_builders
from verl.utils.isabelle_utils.pyexpr import parse_expr, transpile
from verl.utils.isabelle_utils.stages.preparation import _transpile_conjunctive
from verl.utils.isabelle_utils.stages.verification import tolerance_goal


def test_dispatch_predicates_read_structure_not_substrings():
    """has_division fires only on NONCONSTANT denominators (a numeric fraction must not pay the field tactic), and has_powr is a word-boundary match (a variable merely named pv_powr must not fire the exponent tactic)."""
    assert tactics.has_division("(pv_x) / (pv_y - (2::real))") is True
    assert tactics.has_division("((6::real)) / ((4::real))") is False
    assert tactics.has_division("(pv_x) / ((2::real) * pv_y)") is True
    assert tactics.has_division("((3::real)) / ((10::real)) = pv_p") is False
    assert tactics.has_powr("(pv_a) powr (pv_b)") is True
    assert tactics.has_powr("(pv_powr) = (2::real)") is False


def test_alternation_branches_close_or_fail():
    """Isabelle's `|` keeps the first succeeding branch, and simp/auto succeed on partial rewriting; every branch that can leave the goal open must be wrapped `; fail` so the next branch still runs."""
    for open_capable in ("(simp;", "(auto;", "(simp add: field_simps;",
                         "(simp add: algebra_simps;"):
        assert open_capable in tactics.ALTERNATION
    assert "(simp) |" not in tactics.ALTERNATION
    assert "(auto) |" not in tactics.ALTERNATION


def test_unfoldable_rejects_conjunctions():
    """The transpiler joins conjuncts with `&`; a conjunction is not a plain `var = expr` rewrite rule (the old check only knew \\<and> and mislabeled these)."""
    assert theorem_builders._unfoldable("(pv_x = (1::real))") is True
    assert theorem_builders._unfoldable(
        "((pv_x = (1::real)) & (pv_y = (2::real)))") is False


def test_mixed_conjunction_keeps_each_sort():
    term, _, _, carrier = _transpile_conjunctive(
        "x / 2 == 1 and n % 2 == 0", {"x": "real", "n": "int"})
    assert carrier == "mixed"
    assert "x / (2::real)" in term
    assert "n mod (2::int)" in term
    assert "n mod (2::real)" not in term


def test_mixed_conjunction_term_is_reusable_as_premise():
    term, _, _, _ = _transpile_conjunctive(
        "x / 2 == 1 and n % 2 == 0", {"x": "real", "n": "int"})
    premise = f'assumes s0: "{term}"'
    assert "(2::real)" in premise and "(2::int)" in premise


def test_tolerance_preserves_trailing_zero_precision():
    node = parse_expr("x == 0.250")
    goal = tolerance_goal([node], {"x": "real"}, "real")
    assert "(1::real) / (2000::real)" in goal
    assert "(1::real) / (200::real)" not in goal


def test_tolerance_preserves_long_decimal_value():
    node = parse_expr("x == 0.12345678901234567890123")
    goal = tolerance_goal([node], {"x": "real"}, "real")
    assert "12345678901234567890123" in goal
    assert "200000000000000000000000" in goal


def test_tolerance_scientific_notation_precision():
    node = parse_expr("x == 2.50e-3")
    goal = tolerance_goal([node], {"x": "real"}, "real")
    assert "(1::real) / (200000::real)" in goal


def test_log_numeral_haves_only_for_actual_applications():
    """Facts come only from literal `log (B::real) (V::real)` applications: scanning every numeral against every base manufactured facts about pairs no term applies log to, widening the proof surface for nothing."""
    haves, names = theorem_builders.log_numeral_haves(
        ["(log (2::real) (8::real)) = (pv_x)", "(pv_y) = (32::real)"])
    assert names == ["lg2_8"]        # 32 is a numeral, but nothing applies log to it
    assert "log 2 8 = (3::real)" in haves


def test_log_numeral_exponent_has_no_artificial_cap():
    value = 2 ** 50
    haves, names = theorem_builders.log_numeral_haves(
        [f"(log (2::real) ({value}::real))"])
    assert names == [f"lg2_{value}"]
    assert "= (50::real)" in haves


def test_reversed_numeric_integral_bounds_transpile_oriented():
    """The textbook integral from a down to b (a > b) is the negated ordered integral; Isabelle's {a..b} with a > b is the empty set (integral 0), which could validate a claimed 0 for a nonzero oriented value."""
    reversed_term = transpile(parse_expr("integral(x, x, 5, 2) == 0"), {}, "real")
    assert "- (integral {(2::real)..(5::real)}" in reversed_term
    forward_term = transpile(parse_expr("integral(x, x, 2, 5) == 0"), {}, "real")
    assert "- (integral" not in forward_term
    assert "integral {(2::real)..(5::real)}" in forward_term


def test_symbolic_integral_bounds_transpile_to_the_conditional_oriented_form():
    """Symbolic bounds cannot be ordered at translation, and the plain set form would let a premise `a > b` prove the empty set's 0 for a nonzero oriented integral; the conditional keeps both orders honest (constants like 0 <= pi resolve under simp, an order only the premises decide resolves there or fails closed)."""
    term = transpile(parse_expr("integral(1, x, a, b) == 0"),
                     {"a": "real", "b": "real"}, "real")
    assert "if pv_a \\<le> pv_b" in term
    assert "then integral {pv_a..pv_b}" in term
    assert "else - (integral {pv_b..pv_a}" in term


def test_integral_recipe_orders_reversed_numeric_bounds():
    node = parse_expr("integral(x, x, 5, 2) == 0")
    goal = transpile(node, {}, "real")
    theorem = theorem_builders.integral_recipe(node, goal)
    assert theorem is not None
    assert "{(2::real)..(5::real)}" in theorem
    assert "{(5::real)..(2::real)}" not in theorem
