"""Isabelle tactic constants and theorem-building utilities.

Lifted from scripts/isabelle_poc_math500/pipeline_v3.py.
"""
import re
from fractions import Fraction

FREE_NUMS = {Fraction(0), Fraction(1), Fraction(2)}

ALTERNATION = ("((simp) | (simp add: field_simps) | (simp add: algebra_simps) "
               "| (eval) | (linarith) | (presburger) | (auto) "
               "| (auto simp: field_simps) "
               "| (simp add: floor_eq_iff ceiling_eq_iff))")
EVAL_RESCUE = "(eval)"
FALSE_TACTIC = "((auto) | (linarith) | (presburger))"
NZ_TACTIC = ("((simp) | (auto simp: add_eq_0_iff) | (auto) | (linarith) "
             "| (presburger))")

# Giant-number guard (2026-07-11). Goals/premises carrying a huge literal,
# a >=1000 literal exponent, a power tower, or a large factorial make the
# leading simp/presburger of ALTERNATION grind 60-75s while EVADING the 15s
# cooperative headless watchdog (the work is a single uninterruptible native
# GMP op) -- the Isabelle analog of the math-verify bignum spin. Measured:
# simp on 2^100000 = 75s, presburger on 2^5000 = 52s; eval alone honors the
# 15s watchdog on factorials (aborts) AND still proves legit moderate
# computations (2^100 = <value> in 0.7s), whereas linarith on a factorial
# also grinds 75s. So dangerous CLAIMS get eval alone; dangerous consistency
# PROBES are treated as 'undetermined' (never 'consistent') without grinding.
SAFE_DANGEROUS = "(eval)"

# These match BOTH the raw source shape (e.g. "2^5000", "fact 50000") and the
# transpiled Isabelle shape the guard actually sees at the call sites, where a
# computed exponent is wrapped as `(nat (...))` and a factorial argument as
# `(nat (N::int))` (2026-07-11 review: the old `\bfact\s*\(?\s*\d{3,}` and
# `\^\s*\([^)]*\^` were DEAD / FALSE-POSITIVE against transpiler output).
_DANGER_RE = re.compile(
    # power with a >=1000 LITERAL exponent (literal exponents are emitted
    # bare, e.g. "2 ^ 5000"; computed ones get the nat wrapper handled below)
    r"\^\s*\(?\s*-?\d{4,}"
    # LITERAL power tower: an inner `<lit> ^ <lit>` inside a (possibly
    # nat-wrapped) exponent -- 2^(3^11) -> "^ (nat (3 ^ 11))". A SYMBOLIC
    # nested power (2^(n^2) -> "nat (n ^ 2)"; 2^(2^n) -> "nat (2 ^ (nat n))")
    # has a non-literal base or exponent and must NOT match (it never
    # materializes, so eval-only would only lose its reward).
    r"|\^\s*\(\s*(?:nat\s*\(\s*)?\d+\s*\^\s*\(?\s*(?:nat\s*\(\s*)?\d"
    # factorial of a >=100 literal, raw "fact 100" or transpiled
    # "fact (nat (100::int))"
    r"|\bfact\b[\s(]*(?:nat[\s(]*)?\d{3,}"
    # a >=40-digit integer literal
    r"|\d{40,}"
)


def is_dangerous_isabelle(*terms) -> bool:
    """True if any Isabelle term string could drive a tactic to materialize a
    giant integer (long literal, >=1000 literal exponent, power tower, or
    factorial of >=100). Such a term makes the leading simp/presburger of
    ALTERNATION grind 60-75s past the 15s watchdog. Callers route dangerous
    claims to SAFE_DANGEROUS (eval) and dangerous consistency probes to an
    'undetermined' verdict."""
    for t in terms:
        if t and _DANGER_RE.search(t):
            return True
    return False

RESERVED = {
    "True", "False", "if", "then", "else", "case", "of", "let", "in",
    "lambda", "SOME", "THE", "ALL", "EX", "Not", "conj", "disj",
    "implies", "undefined", "div", "mod", "dvd", "int", "nat", "real",
    "rat", "bool", "set", "list", "Nil", "Cons", "Suc", "hd", "tl",
    "fst", "snd", "INF", "SUP", "Min", "Max", "Abs", "Pair",
    "card", "finite", "infinite", "UNIV", "insert", "Union", "Inter",
    "image", "vimage", "range", "dom", "ran", "comp", "Id",
    "Pow", "Sum", "Prod", "length", "nth", "map", "filter", "concat",
    "rev", "sort", "distinct", "remdups", "zip", "enumerate",
    "foldl", "foldr", "takeWhile", "dropWhile", "take", "drop",
    "butlast", "last", "hd", "tl", "replicate", "rotate",
    "shows", "fixes", "assumes", "using", "by", "have", "obtain",
    "where", "proof", "qed", "sorry", "oops",
    "theorem", "lemma", "corollary", "proposition",
    "definition", "fun", "primrec", "function", "abbreviation",
    "begin", "end", "theory", "imports", "section", "subsection",
    "of_nat", "of_int", "of_real", "floor", "ceiling", "round",
    "sqrt", "abs", "sgn", "min", "max", "gcd", "lcm", "fact",
    "choose", "sin", "cos", "tan", "exp", "ln", "log", "pi",
}

WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
    "half": "1/2", "third": "1/3", "quarter": "1/4",
    "first": 1, "second": 2,
}


def fixes_clause(fixes: list) -> str:
    return "".join(f"  fixes {n} :: {t}\n" for n, t in fixes)


def identifiers(expr: str) -> set:
    return {t for t in re.findall(r"[A-Za-z_][\w']*", expr)
            if t not in RESERVED and not t.isdigit()}


def make_theorem(fixes, premises, shows, tactic) -> str:
    """Build an Isabelle theorem string.

    premises: list of (name, prop) pairs.
    """
    if premises:
        ass = "  assumes " + "\n      and ".join(
            f'{n}: "{p}"' for n, p in premises) + "\n"
    else:
        ass = ""
    return (f"theorem chk:\n{fixes_clause(fixes)}{ass}"
            f'  shows "{shows}"\n  '
            + (f"using assms by {tactic}" if premises else f"by {tactic}"))


def num_values(text: str, words: bool = False) -> set:
    vals = set()
    merged = re.sub(r"(?<=\d),(?:\\!)?(?=\d{3})", "", text)
    split = re.sub(r"(?<=\d),(?=\d)", " ", text)
    for t in ((merged, split) if merged != text else (text,)):
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\.?\d)", t):
            try:
                vals.add(Fraction(m.group(1)))
            except ValueError:
                pass
        for m in re.finditer(r"(?<![\w.])(\.\d+)", t):
            vals.add(Fraction("0" + m.group(1)))
    if words:
        for w, v in WORD_NUMS.items():
            if re.search(rf"\b{w}\b", text, re.IGNORECASE):
                vals.add(Fraction(v))
        for m in re.finditer(r"10\s*\^\s*\{?\s*(-?\d+)\s*\}?", text):
            k = int(m.group(1))
            if -30 <= k <= 30:
                vals.add(Fraction(10) ** k)
        for m in re.finditer(
                r"(\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|\*)\s*10\s*"
                r"\^\s*\{?\s*(-?\d+)\s*\}?", text):
            k = int(m.group(2))
            if -30 <= k <= 30:
                vals.add(Fraction(m.group(1)) * Fraction(10) ** k)
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*"
                             r"(thousand|million|billion|trillion)", text,
                             re.IGNORECASE):
            mult = {"thousand": 3, "million": 6, "billion": 9,
                    "trillion": 12}[m.group(2).lower()]
            vals.add(Fraction(m.group(1)) * Fraction(10) ** mult)
    return vals


def anchor_ground_numerals(prop: str, ty: str, declared: set) -> str:
    if identifiers(prop) & declared:
        return prop
    out, last = [], 0
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\.?\d)(?!\w)", prop):
        j = m.start() - 1
        while j >= 0 and prop[j] == " ":
            j -= 1
        if j >= 0 and prop[j] == "^":
            continue
        if re.match(r"\s*::", prop[m.end():]):
            continue
        t = "real" if "." in m.group(1) else ty
        out.append(prop[last:m.start()] + f"({m.group(1)}::{t})")
        last = m.end()
    out.append(prop[last:])
    return "".join(out)
