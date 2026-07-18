"""CPU-only tests for the giant-number guard (tactics.py, 2026-07-11). A goal
or premise carrying a huge literal, a >=1000 literal exponent, a factorial of
>=100, or a power tower makes the leading simp/presburger of ALTERNATION grind
60-75s past the 15s watchdog (measured). is_dangerous_isabelle routes claims
to eval alone and marks premise consistency as unknown without running the
expensive check."""
from verl.utils.isabelle_utils.tactics import (
    SAFE_DANGEROUS, is_dangerous_isabelle,
)


def test_flags_giant_exponent():
    assert is_dangerous_isabelle("(2::int)^100000 = 0")
    assert is_dangerous_isabelle("(2::int)^5000 = x")
    assert is_dangerous_isabelle("y = 3 ^ 1000")


def test_flags_factorial_and_tower_and_long_literal():
    assert is_dangerous_isabelle("(fact 50000::int) = 0")
    assert is_dangerous_isabelle("(2::int)^(3^11) = 0")          # power tower
    assert is_dangerous_isabelle("x = 12345678901234567890123456789012345678901")


def test_flags_transpiled_shapes():
    # the guard runs on TRANSPILED Isabelle terms at the call sites, where a
    # factorial argument is nat-wrapped (2026-07-11 review: the old regex was
    # dead against this form) and a folded literal tower becomes a bare
    # >=1000 exponent.
    assert is_dangerous_isabelle("(answer = (int (fact (nat (100::int)))))")
    assert is_dangerous_isabelle("(x = (real (fact (nat (200::real)))))")
    assert is_dangerous_isabelle("(answer = ((2::real) ^ 177147))")   # folded tower
    assert is_dangerous_isabelle("(answer = (n ^ 177147))")           # sym base, lit exp


def test_symbolic_nested_power_not_flagged():
    # 2026-07-11 review false-positive fix: a SYMBOLIC nested power never
    # materializes, so it must NOT be routed to eval-only (reward loss).
    assert not is_dangerous_isabelle(
        "(answer = (if (n ^ 2) >= (0::int) then ((2::real) ^ (nat (n ^ 2))) else 1))")
    assert not is_dangerous_isabelle("(answer = ((2::real) ^ (nat (2 ^ (nat n)))))")
    assert not is_dangerous_isabelle("(answer = ((2::real) ^ (nat (n + (1::int)))))")
    assert not is_dangerous_isabelle("(answer = (int (fact (nat (5::int)))))")  # small


def test_safe_for_ordinary_math():
    # ordinary competition-scale arithmetic must NOT be flagged
    assert not is_dangerous_isabelle("(2::int)^10 = 1024")
    assert not is_dangerous_isabelle("x^2 + y^2 = z^2")
    assert not is_dangerous_isabelle("a^3 * b^4 = c")            # two small powers
    assert not is_dangerous_isabelle("answer = 189 / 32")
    assert not is_dangerous_isabelle("n mod 2 = 0 and n = 44")
    assert not is_dangerous_isabelle("2 ** (T + 1) = 8")         # symbolic exp
    assert not is_dangerous_isabelle("fact 5 = 120")            # small factorial


def test_multiple_terms_any_dangerous():
    assert is_dangerous_isabelle("x = 1", "y = 2", "(2::int)^9999 = z")
    assert not is_dangerous_isabelle("x = 1", "y = 2", "z = x + y")


def test_safe_tactic_is_eval_only():
    # eval honors the 15s watchdog and proves legit moderate computations;
    # simp/presburger/linarith all grind on giants, so the guard uses eval.
    assert SAFE_DANGEROUS == "(eval)"
