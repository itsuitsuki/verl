"""CPU-only tests for the 2026-07-11 pyexpr soundness fixes:
(a) sign-guarded symbolic integer exponents (bare `nat e` silently mapped
    negative exponents to 0, changing the math);
(b) float literals keep their exact SOURCE text (ast rounds to IEEE float
    first, so ultra-long decimals silently lost precision)."""
from fractions import Fraction

import pytest

from verl.utils.isabelle_utils.pyexpr import (
    PyExprError, parse_expr, py_to_isabelle, transpile,
)


def test_symbolic_exponent_sign_guarded():
    term, _, _, carrier = py_to_isabelle("2 ** (T - 5) == 8", {"T": "int"})
    assert carrier == "real"          # symbolic exponent forces real
    assert "if" in term and ">= (0::int)" in term and "nat" in term
    assert "T - (5::int)" in term
    assert "T - (5::real)" not in term
    # both branches present: positive uses ^, negative uses the reciprocal
    assert "1 /" in term


def test_symbolic_exponent_int_carrier_fails_closed():
    node = parse_expr("2 ** (T - 5)")
    with pytest.raises(PyExprError):
        transpile(node, {"T": "int"}, "int")


def test_literal_exponents_unchanged():
    term, _, _, _ = py_to_isabelle("2 ** 10 == 1024", {})
    assert "^ 10" in term and "if" not in term
    term2, _, _, _ = py_to_isabelle("2 ** -3 == x", {"x": "real"})
    assert "1 / " in term2 and "^ 3" in term2 and "if" not in term2


def test_long_decimal_exact_from_source():
    lit = "0.12345678901234567890123"
    exact = Fraction(lit)
    assert exact != Fraction(str(float(lit)))   # float rounding differs
    term, _, consts, _ = py_to_isabelle(f"x == {lit}", {"x": "real"})
    assert exact in consts                       # analyze saw the exact value
    assert str(exact.numerator) in term          # transpile used it


def test_normal_float_still_decimal_exact():
    term, _, consts, _ = py_to_isabelle("x == 0.025", {"x": "real"})
    assert Fraction(1, 40) in consts
    assert "/ (40::real)" in term or "(1::real)" in term


def test_symbolic_positive_form_also_guarded():
    # T + 1 is symbolic even though contextually positive: still guarded
    term, _, _, carrier = py_to_isabelle("2 ** (T + 1) == 8", {"T": "int"})
    assert carrier == "real" and "if" in term and "nat" in term


def test_zero_and_literal_exponents_stay_plain():
    term, _, _, _ = py_to_isabelle("2 ** 0 == 1", {})
    assert "^ 0" in term and "if" not in term


def test_negative_decimal_exact():
    _, _, consts, _ = py_to_isabelle("x == -0.5", {"x": "real"})
    assert Fraction(1, 2) in consts or Fraction(-1, 2) in consts


def test_trailing_zero_decimal_exact():
    _, _, consts, _ = py_to_isabelle("x == 0.250", {"x": "real"})
    assert Fraction(1, 4) in consts


def test_int_mod_stays_int():
    # checklist #8: integer modulus must not be polluted into real by an
    # unrelated real elsewhere -- at pyexpr level an int-only prop must
    # keep the int carrier and int annotations.
    term, _, _, carrier = py_to_isabelle("n % 2 == 0", {"n": "int"})
    assert carrier == "int"
    assert "::int" in term and "::real" not in term
