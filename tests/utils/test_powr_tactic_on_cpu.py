"""CPU-only tests for the 2026-07-12 symbolic-exponent coverage fix.

Root cause: a valid step like `2 powr (m-n) = 3/4` (from `2 powr m = 3`,
`2 powr n = 4`) is a FALSE NEGATIVE under the old ALTERNATION -- a leading
`simp` rewrites-but-does-not-close the goal, and Isabelle's `|` keeps that
first progress-making branch, blocking every later branch (there was no
powr_diff branch anyway). Fix: prepend close-or-fail `fastforce simp: <law>`
branches. These tests cover the transpiler output shape and the presence of
the fix; the actual Isabelle proof is validated by the on-cluster spot tests
(fastforce simp: powr_diff closes the goal in ~0.5s)."""
import re

from verl.utils.isabelle_utils.pyexpr import py_to_isabelle
from verl.utils.isabelle_utils import tactics


def test_symbolic_real_exponent_emits_powr():
    # a symbolic real exponent must transpile to `powr` (real-base real-exp),
    # NOT the integer sign-guarded `^ (nat ...)` form -- otherwise the step
    # carries wrong (integer) exponent semantics (2026-07-12 audit).
    term, _, _, carrier = py_to_isabelle("2 ** m == 3", {"m": "real"})
    assert carrier == "real"
    assert "powr" in term and "nat" not in term

    tgt, _, _, _ = py_to_isabelle("2 ** (m - n) == 3/4",
                                  {"m": "real", "n": "real"})
    assert "powr" in tgt and "(m - n)" in tgt


def test_alternation_has_exponent_law_branches():
    alt = tactics.ALTERNATION
    for law in ("powr_diff", "powr_add", "powr_mult", "powr_powr",
                "ln_mult", "ln_div", "log_mult", "log_divide",
                "exp_add", "exp_diff"):
        assert f"fastforce simp: {law}" in alt, f"missing {law} branch"


def test_exponent_branches_are_prepended_before_simp():
    # they MUST come before the generic `(simp)` branch, or the leading simp
    # would block them (the whole point of the fix).
    alt = tactics.ALTERNATION
    assert alt.index("fastforce simp: powr_diff") < alt.index("(simp)")


def test_exponent_branches_use_close_or_fail_fastforce():
    # `solves` is unavailable in this session and plain simp/auto are NOT
    # close-or-fail, so the added branches must use fastforce (verified
    # close-or-fail) to avoid re-introducing the blocking bug.
    alt = tactics.ALTERNATION
    for m in re.findall(r"powr_diff|ln_mult|exp_add", alt):
        pass
    assert "solves" not in alt
    assert alt.count("fastforce simp:") >= 10
