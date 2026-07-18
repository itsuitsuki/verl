"""Tactic selection for the Isabelle general verification path."""

import re

# Isabelle's `|` keeps the FIRST branch that succeeds, and simp/auto "succeed" on partial rewriting without closing the goal, which then fails the whole `by` and masks every later branch. Each branch that can leave a goal open is therefore wrapped close-or-fail (`tac; fail`): the branch either closes the goal or fails cleanly so the next branch runs. The specialized fastforce branches close-or-fail by nature and still precede generic simp so a symbolic exponent or logarithm law is applied before simp rewrites the goal away from it.
ALTERNATION = (
    "((fastforce simp: powr_diff) | (fastforce simp: powr_add) "
    "| (fastforce simp: powr_mult) | (fastforce simp: powr_powr) "
    "| (fastforce simp: ln_mult) | (fastforce simp: ln_div) "
    "| (fastforce simp: log_mult) | (fastforce simp: log_divide) "
    "| (fastforce simp: exp_add) | (fastforce simp: exp_diff) "
    "| (simp; fail) | (simp add: field_simps; fail) "
    "| (simp add: algebra_simps; fail) "
    "| (eval) | (linarith) | (presburger) | (auto; fail) "
    "| (auto simp: field_simps; fail) "
    "| (simp add: floor_eq_iff ceiling_eq_iff; fail))"
)
EVAL_TACTIC = "(eval)"
LINEAR_FALSE = "((linarith) | (presburger))"
LINEAR_CLAIM = "((argo) | (smt (verit)))"

# Evaluation avoids the tactics that may materialize giant integers indefinitely.
SAFE_DANGEROUS = "(eval)"

# Specialized tactics are attempted only when their corresponding syntax is present.
SOS_TACTIC = "(sos)"
SMT_TACTIC = "(smt (verit))"
EXPONENT_TACTIC = "(smt (verit) powr_inj powr_one one_less_numeral_iff)"
FIELD_TACTIC = "(auto simp: field_simps)"
ALGEBRA_TACTIC = "(simp add: eval_nat_numeral algebra_simps)"
TRIG_TACTIC = (
    "(simp add: sin_diff cos_diff sin_add cos_add tan_def "
    "sin_30 cos_30 sin_45 cos_45 sin_60 cos_60 field_simps)"
)


def has_powr(*terms) -> bool:
    """Return whether a term applies the real exponentiation operator `powr`. Word-boundary match: an identifier merely containing the letters (pv_powr) must not fire the exponent tactic."""
    return any(term and re.search(r"\bpowr\b", term) for term in terms)


def has_poly(*terms: str) -> bool:
    """Return whether a term contains multiplication or a power."""
    return any(term and ("^" in term or "*" in term) for term in terms)


_CAST_RE = re.compile(r"::\s*(?:real|int|nat|rat)\b")


def _denominator_at(term: str, slash_idx: int) -> str:
    """The denominator substring following the '/' at slash_idx: a balanced parenthesized group, or the next identifier/numeral token."""
    i = slash_idx + 1
    while i < len(term) and term[i] == " ":
        i += 1
    if i >= len(term):
        return ""
    if term[i] == "(":
        depth = 0
        for j in range(i, len(term)):
            if term[j] == "(":
                depth += 1
            elif term[j] == ")":
                depth -= 1
                if depth == 0:
                    return term[i:j + 1]
        return term[i:]
    match = re.match(r"[A-Za-z0-9_']+", term[i:])
    return match.group(0) if match else ""


def has_division(*terms) -> bool:
    """Return whether a term divides by a NONCONSTANT denominator: the denominator text, sort casts removed, still contains an identifier (a pv_ variable or a function application). A purely numeric denominator such as (4::real) stays False, so plain numeric fractions do not fire the field tactic."""
    for term in terms:
        text = term or ""
        for match in re.finditer("/", text):
            denominator = _CAST_RE.sub("", _denominator_at(text, match.start()))
            if re.search(r"[A-Za-z_]", denominator):
                return True
    return False


def has_trig(*terms) -> bool:
    """Return whether a term applies sin, cos, or tan."""
    return any(re.search(r"\b(?:sin|cos|tan)\b", term or "") for term in terms)


# The patterns match the transpiled forms seen by the verifier: a literal exponent of at least four digits, a literal power tower, a factorial of at least three digits, or an integer literal of at least forty digits.
_DANGER_RE = re.compile(
    r"\^\s*\(?\s*-?\d{4,}"
    r"|\^\s*\(\s*(?:nat\s*\(\s*)?\d+\s*\^\s*\(?\s*(?:nat\s*\(\s*)?\d"
    r"|\bfact\b[\s(]*(?:nat[\s(]*)?\d{3,}"
    r"|\d{40,}"
)


def is_dangerous_isabelle(*terms) -> bool:
    """Return whether a term may make Isabelle materialize a giant integer."""
    return any(term and _DANGER_RE.search(term) for term in terms)
