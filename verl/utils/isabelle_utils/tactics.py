"""Isabelle tactic constants and theorem-building utilities.

Lifted from scripts/isabelle_poc_math500/pipeline_v3.py.
"""
import re
from fractions import Fraction

FREE_NUMS = {Fraction(0), Fraction(1), Fraction(2)}

# Exponent/log/exp-law branches are PREPENDED (2026-07-12). Isabelle's `|`
# combinator keeps the FIRST branch that makes progress, so a leading `simp`
# that rewrites-but-does-not-close a symbolic-exponent goal (e.g.
# `2 powr (m-n) = 3/4` from `2 powr m = 3`, `2 powr n = 4`) blocks every later
# branch and the whole `by` fails -- a FALSE NEGATIVE on a valid step (the
# generic tactics never invoke powr_diff). `fastforce` is close-or-fail (it
# fails cleanly instead of leaving subgoals), so these branches never block
# the generic ones below and, being first, are never blocked themselves. Each
# was verified to close its representative lemma-case and to fail fast (<0.6s)
# otherwise. This is sound (fastforce proves only true goals) and strictly
# widens coverage -- it recovers reward for valid exponent/log steps.
ALTERNATION = ("((fastforce simp: powr_diff) | (fastforce simp: powr_add) "
               "| (fastforce simp: powr_mult) | (fastforce simp: powr_powr) "
               "| (fastforce simp: ln_mult) | (fastforce simp: ln_div) "
               "| (fastforce simp: log_mult) | (fastforce simp: log_divide) "
               "| (fastforce simp: exp_add) | (fastforce simp: exp_diff) "
               "| (simp) | (simp add: field_simps) | (simp add: algebra_simps) "
               "| (eval) | (linarith) | (presburger) | (auto) "
               "| (auto simp: field_simps) "
               "| (simp add: floor_eq_iff ceiling_eq_iff))")
EVAL_RESCUE = "(eval)"
FALSE_TACTIC = "((auto) | (linarith) | (presburger))"
# Consistency probe for a PURELY LINEAR premise chain (2026-07-14). When every
# accumulated premise is affine with constant coefficients (is_linear_arith),
# linarith (reals) and presburger (integers) are COMPLETE decision procedures:
# they prove |- False iff the premises are genuinely contradictory and FAIL
# FAST otherwise. ALTERNATION, by contrast, tries its ten leading
# `fastforce simp: powr_*` / `ln_*` / `exp_*` branches FIRST, and those GRIND to
# the 15s watchdog searching a 10+-equation linear goal they can make no use of
# -- measured on the arithmetic-sequence givens (a_1+a_4+a_7=48, ...): 15.3s
# incomplete -> undetermined -> the whole solution wrongly forced into safe mode
# (regression oo -> uu). `(linarith|presburger)` closes the same consistent set
# in 0.4s and still proves False on a genuinely inconsistent one in ~2s. Sound
# because it is selected ONLY when the premises are linear, the fragment where
# these two tactics miss no contradiction a heavier tactic could exploit.
LINEAR_FALSE = "((linarith) | (presburger))"
# Claim proof for a PURELY LINEAR step (2026-07-14). Proving a specific linear
# EQUALITY from MANY linear equalities (a_3+a_6+a_9 = 32 from an 11-variable
# arithmetic-sequence system, ~19 premises with the givens attached to every
# step) defeats the usual tactics on the ACTUAL engine premise set:
#   linarith  -- Fourier-Motzkin variable elimination is worst-case doubly-
#                exponential in the number of variables -> 19s watchdog / incomplete;
#   force/fastforce/auto/simp -- search / rewrite blow up on 13-19 premises -> 18-20s;
#   presburger/algebra/metis -- fail fast but do not prove it.
# `argo` (Isabelle's BUILT-IN linear-arithmetic + congruence decision procedure,
# simplex-based -> polynomial in the variable count, no external solver) closes
# BOTH the simple difference step and the 11-variable value step in ~0.4s and
# fails fast (<0.4s) on an unentailed goal, so it never hits the watchdog. Its
# one gap is integer division/rounding (3*x = 12 => x = 4), which `smt (verit)`
# (bundled, certificate replayed -> sound) covers; smt in turn drops real
# division (x/2 = 3), which argo handles -- so the two are complementary and the
# pair proves the whole linear battery (sequence, gsm8k, division, integer,
# percent) while both fail fast on unprovable goals (no grind).
# NOTE: this is for PROVING the claim; the consistency probe keeps LINEAR_FALSE
# (disproving `False` from a consistent set is fast, and linarith/presburger stay
# COMPLETE for detecting a linear contradiction, which argo/smt do not guarantee).
LINEAR_CLAIM = "((argo) | (smt (verit)))"
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

# Nonlinear real rescue (2026-07-13). `sos` (sum-of-squares / Positivstellensatz,
# checked certificate -> sound) closes polynomial goals linarith/presburger
# cannot: geometric-ratio steps (r^2 = 9, r > 0 ==> r = 3) and coordinate
# squared-distance goals. Measured: proves the geometric case in 0.7s, but
# GRINDS to the 15s watchdog on any polynomial goal it cannot close, and
# proves anything ex-falso from inconsistent premises in 0.5s. So the engine
# gates it to nonlinear goals (is_nonlinear), non-safe steps, and non-danger.
SOS_RESCUE = "(sos)"

# SMT rescue (2026-07-13). smt (verit) discharges uninterpreted-function and
# quantified goals that auto/linarith/sos cannot -- functional equations like
# `forall x. f(x+1)=f(x)+2, f 0=5 ==> f 2=9` (0.7s) and existentials with a
# functional witness. It reconstructs a checked proof (sound) but calls an
# external solver: it can GRIND ~15s on nonlinear numeric goals it cannot
# close, so the engine gates it to steps carrying an uninterpreted function
# (where nothing else applies), non-safe (ex-falso), and non-danger.
SMT_RESCUE = "(smt (verit))"

# Exponential-equation rescue (2026-07-15). Solving `b powr f(x) = c` for the
# exponent -- e.g. `4 powr (2*y) = 4 => 2*y = 1` (x^(2y)=4, x=4) -- needs the
# INJECTIVITY of a real power with base > 1: 4 = 4 powr 1, so 4 powr (2y) =
# 4 powr 1, and powr_inj gives 2y = 1. No plain tactic does it (auto/simp/sos
# fail; metis and plain smt GRIND 15s), but smt WITH powr_inj + powr_one +
# one_less_numeral_iff reconstructs it in 0.6s. Sound (checked certificate);
# gated to a step whose premises/goal carry `powr` (a variable-exponent power),
# non-safe (ex-falso), non-danger.
EXP_RESCUE = "(smt (verit) powr_inj powr_one one_less_numeral_iff)"


def has_powr(*terms) -> bool:
    """True if any Isabelle term uses `powr` (real power with a possibly
    non-integer / variable exponent) -- the trigger for EXP_RESCUE."""
    return any("powr" in t for t in terms)

# Rational-equation rescue (2026-07-14). An unknown in a DENOMINATOR -- the
# work-rate/mixture family: "1/comb = 1/pipe - 1/leak" with comb, leak known,
# solve for pipe -- is not closed by ALTERNATION even though it contains
# `(auto simp: field_simps)`: the `(eval)` earlier in the alternation aborts on
# the free-variable goal before field_simps is reached. Applied on its own,
# WITH the denominators proved nonzero, field_simps clears them and solves.
FIELD_RESCUE = "(auto simp: field_simps)"

# Polynomial-identity rescue (2026-07-14). A factoring / expansion identity
# (58*x^5 - 203*x^11 = 29*x^5*(2 - 7*x^6)) is closed by expanding both sides
# with algebra_simps, but ALTERNATION -- although it CONTAINS (simp add:
# algebra_simps) -- fails on it: an earlier `fastforce simp: powr_*` branch
# grinds/aborts on the polynomial before the algebra_simps branch is reached.
# Run it standalone (gated on a power/product in the goal).
ALGEBRA_RESCUE = "(simp add: algebra_simps)"


def has_poly(*terms: str) -> bool:
    """A term multiplies or raises to a power (a polynomial identity that
    algebra_simps can expand)."""
    return any(t and ("^" in t or "*" in t) for t in terms)

# Trig rescue (2026-07-14). Two families:
#  (1) identities: `tan(pi - a) = -tan(a)`, `sin(pi/2 - a) = cos a`, the
#      angle-sum/-difference laws -- expand tan into sin/cos and apply the
#      addition laws; field_simps clears the sin/cos denominators tan makes.
#  (2) standard-angle VALUES: sin/cos/tan at pi/6, pi/4, pi/3 (30/45/60 deg)
#      via the sin_30/cos_30/sin_45/cos_45/sin_60/cos_60 library lemmas, so
#      `sin(pi/6) = 1/2`, `tan(pi/4) = 1` (the last via tan_def + sin_45/cos_45).
# (Solving sin from tan + a quadrant constraint is NOT covered -- that needs the
# Pythagorean identity plus sign reasoning, beyond automation.)
TRIG_RESCUE = ("(simp add: sin_diff cos_diff sin_add cos_add tan_def "
               "sin_30 cos_30 sin_45 cos_45 sin_60 cos_60 field_simps)")


def has_division(*terms) -> bool:
    """A term divides by a NON-constant denominator (`/` followed by a name or
    `(`), so it needs field_simps and the denominator proved nonzero. A bare
    numeric fraction like 3/4 (`/` then a digit) does NOT match."""
    return any(re.search(r"/\s*[A-Za-z(]", t or "") for t in terms)


def has_trig(*terms) -> bool:
    """A term applies sin/cos/tan (word-bounded, so a variable like pv_sin does
    not match)."""
    return any(re.search(r"\b(?:sin|cos|tan)\b", t or "") for t in terms)

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
    "choose", "sin", "cos", "tan", "exp", "ln", "log", "pi", "prime",
    "sum", "prod", "real", "of_nat", "forall", "exists", "powr",
    "integral", "has_real_derivative", "has_integral", "at", "arcsin",
    "has_vector_derivative", "has_field_derivative",
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
    # Local binders are NOT free variables and must never be declared as
    # fixes: `%k. ...` (sum/prod index) and `\<forall>x. ...` / `\<exists>x.`
    # (quantified variable).
    bnd = set(re.findall(r"%\s*([A-Za-z_][\w']*)\s*\.", expr))
    bnd |= set(re.findall(r"\\<(?:forall|exists)>\s*([A-Za-z_][\w']*)", expr))
    # Drop Isabelle symbol tokens (\<forall>, \<longrightarrow>, \<noteq>, ...)
    # so their inner letters are never read as variables.
    expr = re.sub(r"\\<[A-Za-z]+>", " ", expr)
    return {t for t in re.findall(r"[A-Za-z_][\w']*", expr)
            if t not in RESERVED and t not in bnd and not t.isdigit()}


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


def _unfoldable(prop: str) -> bool:
    """A transpiled premise that is a plain equation `var = expr` (simple
    variable on the left), usable as an `unfolding` rewrite rule -- not an
    inequality or a compound proposition."""
    if any(b in prop for b in ("\\<le>", "\\<ge>", "\\<noteq>", "\\<and>",
                               "\\<or>", "\\<longrightarrow>", "\\<forall>",
                               "\\<exists>", "<", ">")):
        return False
    return bool(re.match(r"\s*\(*\s*[A-Za-z_][\w']*\s*=\s", prop))


def make_theorem_unfold(fixes, premises, shows, tactic="eval"):
    """Unfold the definitional equation premises (var = expr) INTO the goal,
    then run `tactic` (default eval). Proves `answer = 16` from the given
    `answer = gcd 32 48`: `using assms by eval` cannot, because eval needs a
    ground goal and the free `answer` blocks evaluation until the definition is
    substituted. None if no premise is an unfoldable equation."""
    eqs = [n for n, p in premises if _unfoldable(p)]
    if not eqs:
        return None
    ass = "  assumes " + "\n      and ".join(
        f'{n}: "{p}"' for n, p in premises) + "\n"
    return (f"theorem chk:\n{fixes_clause(fixes)}{ass}"
            f'  shows "{shows}"\n  unfolding {" ".join(eqs)} by {tactic}')


# Log-numeral toolbox (2026-07-13). `log B V = N` (B, V, N numerals with
# V = B^N) is NOT closable by any single tactic: it needs V rewritten as
# `B powr N` first (the numeral<->powr gap), which no automatic tactic bridges,
# and smt treats log as uninterpreted. But the multi-step Isar proof is
# mechanical. Rather than special-case one goal shape, we PROVE every such
# `log B V = N` fact derivable from the bases and numerals in a step and offer
# them as `have` lemmas; the downstream tactic (simp for a bare log goal, smt
# for a functional one like f(log b x)=x) then composes them with any
# reasoning. This is general over all log-numeral computation, not one pattern.
_LOG_BASE_RE = re.compile(r"log \((-?\d+)::real\)")
_NUMERAL_RE = re.compile(r"\((-?\d+)::real\)")


def has_log(*terms: str) -> bool:
    return any(t and _LOG_BASE_RE.search(t) for t in terms)


def log_numeral_haves(terms: list) -> tuple:
    """(`have` block, lemma names) proving every `log B V = N` (V = B^N)
    derivable from the log bases and the numerals occurring in `terms`."""
    blob = " ".join(t for t in terms if t)
    bases = {int(b) for b in _LOG_BASE_RE.findall(blob)}
    nums = {int(v) for v in _NUMERAL_RE.findall(blob)}
    haves, names = [], []
    for B in sorted(bases):
        if B < 2:
            continue
        for V in sorted(n for n in nums if n >= 1):
            n_exp, p = 0, 1
            while p < V and n_exp < 40:
                n_exp += 1
                p *= B
            if p != V:                       # V is not an exact power of B
                continue
            nm = f"lg{B}_{V}"
            haves.append(
                f'  have {nm}: "log {B} {V} = ({n_exp}::real)"\n'
                f'  proof -\n    have "({V}::real) = {B} powr {n_exp}" '
                f'by (simp add: powr_numeral)\n'
                f'    thus ?thesis using log_powr[of {B} {B} {n_exp}] '
                f'by simp\n  qed')
            names.append(nm)
    return "\n".join(haves), names


def make_theorem_with_logs(fixes, premises, shows, tactic) -> str:
    """Like make_theorem, but first proves the log-numeral facts of the goal
    and premises as local `have` lemmas and hands them to the tactic. Falls
    back to a plain make_theorem when there is no log-numeral fact to add."""
    haves, names = log_numeral_haves([shows] + [p for _, p in premises])
    if not names:
        return make_theorem(fixes, premises, shows, tactic)
    if premises:
        ass = "  assumes " + "\n      and ".join(
            f'{n}: "{p}"' for n, p in premises) + "\n"
        using = "using assms " + " ".join(names)
    else:
        ass = ""
        using = "using " + " ".join(names)
    return (f"theorem chk:\n{fixes_clause(fixes)}{ass}"
            f'  shows "{shows}"\n'
            f"proof -\n{haves}\n  show ?thesis {using} by {tactic}\nqed")


# --- Calculus (2026-07-13). derivative_eq_intros proves has_real_derivative
# goals; real_asymp decides limits/asymptotics. Definite integrals have no
# automatic tactic -- FTC needs the antiderivative SUPPLIED. For a polynomial
# integrand it is mechanical (power rule), computed here and handed to FTC;
# this is one integral strategy among several (symmetry, approximation, known
# theorems), not the only one. The recipe expands the integrand, so odd
# polynomials over a symmetric interval fall out as 0 automatically. ---
import ast as _ast  # noqa: E402

DERIV_TACTIC = "(auto intro!: derivative_eq_intros)"
LIMIT_TACTIC = "(real_asymp)"


def has_deriv(*terms: str) -> bool:
    return any(t and "has_real_derivative" in t for t in terms)


def has_limit(*terms: str) -> bool:
    return any(t and "\\<longlongrightarrow>" in t for t in terms)


def has_integral_goal(*terms: str) -> bool:
    return any(t and "integral {" in t for t in terms)


def _poly_terms(node, var):
    """{power: Fraction coeff} if `node` is a polynomial in `var`, else None.
    Handles constants, the variable, sums/differences, products (convolution)
    and non-negative integer powers (repeated convolution)."""
    from verl.utils.isabelle_utils.pyexpr import _const_eval
    if isinstance(node, _ast.Constant):
        c = _const_eval(node)
        return {0: c} if c is not None else None
    if isinstance(node, _ast.Name):
        return {1: Fraction(1)} if node.id == var else None
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
        t = _poly_terms(node.operand, var)
        return {p: -c for p, c in t.items()} if t else None
    if isinstance(node, _ast.BinOp):
        if isinstance(node.op, (_ast.Add, _ast.Sub)):
            lo, ro = _poly_terms(node.left, var), _poly_terms(node.right, var)
            if lo is None or ro is None:
                return None
            out = dict(lo)
            for p, c in ro.items():
                out[p] = out.get(p, Fraction(0)) + (
                    c if isinstance(node.op, _ast.Add) else -c)
            return out
        if isinstance(node.op, _ast.Mult):
            lo, ro = _poly_terms(node.left, var), _poly_terms(node.right, var)
            if lo is None or ro is None:
                return None
            out = {}
            for p1, c1 in lo.items():
                for p2, c2 in ro.items():
                    out[p1 + p2] = out.get(p1 + p2, Fraction(0)) + c1 * c2
            return out
        if isinstance(node.op, _ast.Pow):
            from verl.utils.isabelle_utils.pyexpr import _const_eval
            base, e = _poly_terms(node.left, var), _const_eval(node.right)
            if base is None or e is None or e.denominator != 1 or e < 0:
                return None
            res = {0: Fraction(1)}
            for _ in range(e.numerator):
                nxt = {}
                for p1, c1 in res.items():
                    for p2, c2 in base.items():
                        nxt[p1 + p2] = nxt.get(p1 + p2, Fraction(0)) + c1 * c2
                res = nxt
            return res
    return None


def _rc(f) -> str:
    f = Fraction(f)
    if f.denominator == 1:
        return f"({f.numerator}::real)"
    return f"(({f.numerator}::real)/{f.denominator})"


def _poly_emit(terms, var, shift) -> str:
    # shift=0 -> the polynomial; shift=1 -> its antiderivative.
    parts = [f"{_rc(terms[p] / (p + shift) if shift else terms[p])} "
             f"* ({var} ^ {p + shift})" for p in sorted(terms)]
    return " + ".join(parts) if parts else "(0::real)"


def integral_recipe(node, goal_term: str):
    """Full Isabelle theorem (FTC + computed antiderivative) for a polynomial
    definite-integral claim `integral(body, x, a, b) == V` (a, b, V numerals),
    else None. Expands the integrand so odd-over-symmetric cases give 0."""
    from verl.utils.isabelle_utils.pyexpr import py_to_isabelle, _const_eval
    if not (isinstance(node, _ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], _ast.Eq)):
        return None
    intg = val = None
    for x, y in ((node.left, node.comparators[0]),
                 (node.comparators[0], node.left)):
        if (isinstance(x, _ast.Call) and isinstance(x.func, _ast.Name)
                and x.func.id == "integral"):
            intg, val = x, y
            break
    if (intg is None or len(intg.args) != 4
            or not isinstance(intg.args[1], _ast.Name)):
        return None
    var = intg.args[1].id
    a, b, v = (_const_eval(intg.args[2]), _const_eval(intg.args[3]),
               _const_eval(val))
    if a is None or b is None or v is None:
        return None
    terms = _poly_terms(intg.args[0], var)
    if terms is None:
        return None
    try:
        from verl.utils.isabelle_utils.pyexpr import transpile as _tp
        # transpile the integrand with the integration variable BOUND, so it is
        # emitted as `var` (matching the lambda) and NOT pv_-prefixed as a free
        # variable.
        orig = _tp(intg.args[0], {var: "real"}, "real", {var: "real"})
    except Exception:  # noqa: BLE001
        return None
    expd, F = _poly_emit(terms, var, 0), _poly_emit(terms, var, 1)
    A_, B_ = _rc(a), _rc(b)
    L = f"(\\<lambda>{var}."
    return (
        f'theorem chk:\n  shows "{goal_term}"\nproof -\n'
        f'  have eq: "{L} {orig}) = {L} {expd})" '
        f'by (simp add: fun_eq_iff algebra_simps)\n'
        f'  have "({L} {expd}) has_integral '
        f'({L} {F}) {B_} - {L} {F}) {A_})) {{{A_}..{B_}}}"\n'
        f'    apply (rule fundamental_theorem_of_calculus)\n     apply simp\n'
        f'    apply (unfold has_real_derivative_iff_has_vector_derivative'
        f'[symmetric])\n    apply (auto intro!: derivative_eq_intros)\n    done\n'
        f'  hence "integral {{{A_}..{B_}}} {L} {expd}) = '
        f'({L} {F}) {B_} - {L} {F}) {A_})" by (rule integral_unique)\n'
        f'  thus ?thesis using eq by (simp add: field_simps)\nqed')


# Unicode superscripts (10⁶, x², 10⁻⁶) -> ASCII, so a
# scientific-notation answer written with real superscript glyphs is seen: the
# exponent digit is recognized as a number the model wrote, not an invention.
_SUPERSCRIPT = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺",
    "0123456789-+")


def num_values(text: str, words: bool = False) -> set:
    vals = set()
    text = text.translate(_SUPERSCRIPT)
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


def range_ints(text: str, cap: int = 200) -> set:
    """Integers spanned by an inclusive-range phrase ('1 through 9' -> 1..9).

    num_values only sees a range's two endpoints, so a range-based problem
    ("the smallest number divisible by the integers 1 through 9") makes the
    faithfulness window reject the interior integers (4, 6, 8) as invented and
    the translation is rejected. Add every integer of a stated inclusive range;
    a range wider than `cap` is skipped so a stray "1 to 1000000" cannot flood
    the window (which would defeat the invented-number guard)."""
    out = set()
    pats = (
        r"(\d+)\s+(?:through|thru)\s+(\d+)",
        r"from\s+(\d+)\s+to\s+(\d+)",
        r"(?:integers?|numbers?|digits?|values?|terms?)\s+(\d+)\s+to\s+(\d+)",
        r"between\s+(\d+)\s+and\s+(\d+)",
        # ellipsis: "1, 2, ..., 9" / "1, 2, \ldots, 9" / "1, 2, \cdots, 9"
        r"(\d+)\s*,\s*\d+\s*,\s*(?:\.\.\.|…|\\l?dots|\\cdots)\s*,?\s*(\d+)",
    )
    for pat in pats:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                lo, hi = int(m.group(1)), int(m.group(2))
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            if 0 <= hi - lo <= cap:
                out |= {Fraction(i) for i in range(lo, hi + 1)}
    return out


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
