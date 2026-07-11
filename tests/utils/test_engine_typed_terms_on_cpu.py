"""CPU-only semantic tests for typed Isabelle term construction."""

from verl.utils.isabelle_utils.engine import (
    _transpile_conjunctive,
    tolerance_goal,
)
from verl.utils.isabelle_utils.pyexpr import parse_expr


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
