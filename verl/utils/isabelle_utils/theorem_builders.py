"""Build Isabelle theorem text for the general verification path.

This module contains deterministic theorem construction only. Tactic selection and proof-attempt order belong to ``tactics.py`` and ``stages/verification.py``.
"""


import ast
import re
from fractions import Fraction

import verl.utils.isabelle_utils.pyexpr as pyexpr


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
    "arccos", "arctan",
    "has_vector_derivative", "has_field_derivative",
}


def fixes_clause(fixes: list) -> str:
    return "".join(f"  fixes {n} :: {t}\n" for n, t in fixes)


def _assumes_clause(premises) -> str:
    if not premises:
        return ""
    return "  assumes " + "\n      and ".join(
        f'{name}: "{proposition}"' for name, proposition in premises) + "\n"


def identifiers(expr: str) -> set:
    # Local binders are not free variables and must not be declared as fixes: `%k. ...` for sum/prod indices, and `\<forall>x. ...` or `\<exists>x. ...` for quantified variables.
    bnd = set(re.findall(r"%\s*([A-Za-z_][\w']*)\s*\.", expr))
    bnd |= set(re.findall(r"\\<(?:forall|exists)>\s*([A-Za-z_][\w']*)", expr))
    # Remove Isabelle symbol tokens before scanning identifiers, so the inner letters of `\<forall>`, `\<longrightarrow>`, and similar symbols are not treated as variables.
    expr = re.sub(r"\\<[A-Za-z]+>", " ", expr)
    return {t for t in re.findall(r"[A-Za-z_][\w']*", expr)
            if t not in RESERVED and t not in bnd and not t.isdigit()}


def make_theorem(fixes, premises, shows, tactic) -> str:
    """Build an Isabelle theorem string.

    premises: list of (name, prop) pairs.
    """
    ass = _assumes_clause(premises)
    return (f"theorem chk:\n{fixes_clause(fixes)}{ass}"
            f'  shows "{shows}"\n  '
            + (f"using assms by {tactic}" if premises else f"by {tactic}"))


def _unfoldable(prop: str) -> bool:
    """A transpiled premise that is a plain equation `var = expr` (simple variable on the left), usable as an `unfolding` rewrite rule; not an inequality, a conjunction (the transpiler joins conjuncts with `&`), or another compound proposition."""
    if any(b in prop for b in ("\\<le>", "\\<ge>", "\\<noteq>", "&", "\\<and>",
                               "\\<or>", "\\<longrightarrow>", "\\<forall>",
                               "\\<exists>", "<", ">")):
        return False
    return bool(re.match(r"\s*\(*\s*[A-Za-z_][\w']*\s*=\s", prop))


def make_theorem_unfold(fixes, premises, shows, tactic="eval"):
    """Substitute simple equation premises into the goal before running a tactic.

    For example, ``using assms by eval`` cannot prove ``answer = 16`` from ``answer = gcd 32 48`` while ``answer`` remains free. ``unfolding`` performs that substitution first. Returns ``None`` when no premise is usable as a rewrite rule.
    """
    eqs = [n for n, p in premises if _unfoldable(p)]
    if not eqs:
        return None
    ass = _assumes_clause(premises)
    return (f"theorem chk:\n{fixes_clause(fixes)}{ass}"
            f'  shows "{shows}"\n  unfolding {" ".join(eqs)} by {tactic}')


_LOG_BASE_RE = re.compile(r"log \((-?\d+)::real\)")


# A literal log application as the transpiler emits it: `(log (B::real) (V::real))`.
_LOG_APP_RE = re.compile(r"log \((-?\d+)::real\) \((-?\d+)::real\)")


def has_log(*terms: str) -> bool:
    return any(t and _LOG_BASE_RE.search(t) for t in terms)


def log_numeral_haves(terms: list) -> tuple:
    """(`have` block, lemma names) proving `log B V = N` for every LITERAL log application `log (B::real) (V::real)` in `terms` whose V is an exact power of B. Only actual applications produce facts (scanning every numeral against every base manufactured facts about pairs no term applies log to); the exponent search is bounded by V itself (p grows geometrically), so no arbitrary cap is needed, and giant literals never reach here (the dangerous-term check keeps them off the log tactic path)."""
    blob = " ".join(t for t in terms if t)
    apps = {(int(b), int(v)) for b, v in _LOG_APP_RE.findall(blob)}
    haves, names = [], []
    for B, V in sorted(apps):
        if B < 2 or V < 1:
            continue
        n_exp, p = 0, 1
        while p < V:
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
    ass = _assumes_clause(premises)
    using = ("using assms " if premises else "using ") + " ".join(names)
    return (f"theorem chk:\n{fixes_clause(fixes)}{ass}"
            f'  shows "{shows}"\n'
            f"proof -\n{haves}\n  show ?thesis {using} by {tactic}\nqed")


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
    if isinstance(node, ast.Constant):
        c = _const_eval(node)
        return {0: c} if c is not None else None
    if isinstance(node, ast.Name):
        return {1: Fraction(1)} if node.id == var else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        t = _poly_terms(node.operand, var)
        return {p: -c for p, c in t.items()} if t else None
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            lo, ro = _poly_terms(node.left, var), _poly_terms(node.right, var)
            if lo is None or ro is None:
                return None
            out = dict(lo)
            for p, c in ro.items():
                out[p] = out.get(p, Fraction(0)) + (
                    c if isinstance(node.op, ast.Add) else -c)
            return out
        if isinstance(node.op, ast.Mult):
            lo, ro = _poly_terms(node.left, var), _poly_terms(node.right, var)
            if lo is None or ro is None:
                return None
            out = {}
            for p1, c1 in lo.items():
                for p2, c2 in ro.items():
                    out[p1 + p2] = out.get(p1 + p2, Fraction(0)) + c1 * c2
            return out
        if isinstance(node.op, ast.Pow):
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


def _lin_arg(arg, var):
    """(k, iso) if `arg` is k*var (k a nonzero constant) or var (k=1), for the
    chain rule antiderivative .../k. Else (None, None)."""
    from verl.utils.isabelle_utils.pyexpr import _const_eval
    if isinstance(arg, ast.Name) and arg.id == var:
        return Fraction(1), var
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
        for c_side, v_side in ((arg.left, arg.right), (arg.right, arg.left)):
            c = _const_eval(c_side)
            if (c is not None and c != 0 and isinstance(v_side, ast.Name)
                    and v_side.id == var):
                return c, f"{_rc(c)} * {var}"
    return None, None


def _one_plus_sq(node, var):
    """True if `node` is 1 + var^2 (either order); 1/(1+x^2) -> arctan x."""
    from verl.utils.isabelle_utils.pyexpr import _const_eval
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        return False
    for a, b in ((node.left, node.right), (node.right, node.left)):
        if (_const_eval(a) == Fraction(1) and isinstance(b, ast.BinOp)
                and isinstance(b.op, ast.Pow)
                and isinstance(b.left, ast.Name) and b.left.id == var
                and _const_eval(b.right) == Fraction(2)):
            return True
    return False


def _antideriv(node, var):
    """Return an antiderivative expression for the supported integrand basis.

    The basis consists of constants, nonnegative powers, linear combinations, ``sin(k*x)``, ``cos(k*x)``, ``exp(k*x)``, ``1/(1+x^2)``, ``1/x``, and ``ln x``. The boolean in the result records whether the generated derivative proof needs ``inverse_eq_divide``.

    Isabelle checks the derivative in the generated FTC proof. A wrong candidate therefore fails to prove rather than validating an incorrect claim.
    """
    from verl.utils.isabelle_utils.pyexpr import _const_eval
    flags = {"inv": False}

    def rec(n):
        c = _const_eval(n)
        if c is not None:                                    # c -> c*x
            return f"({_rc(c)} * {var})"
        if isinstance(n, ast.Name) and n.id == var:         # x -> x^2/2
            return f"({var}^2 / 2)"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            r = rec(n.operand)
            return f"(- {r})" if r else None
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            lo, ro = rec(n.left), rec(n.right)
            if lo is None or ro is None:
                return None
            return f"({lo} {'+' if isinstance(n.op, ast.Add) else '-'} {ro})"
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            if isinstance(n.left, ast.Name) and n.left.id == var:
                e = _const_eval(n.right)
                if e is not None and e.denominator == 1 and e >= 0:
                    return f"({var}^{e.numerator + 1} / {e.numerator + 1})"
            return None
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            for c_side, o_side in ((n.left, n.right), (n.right, n.left)):
                c = _const_eval(c_side)
                if c is not None:
                    r = rec(o_side)
                    return f"({_rc(c)} * {r})" if r else None
            return None
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            num = _const_eval(n.left)
            if num is not None and _one_plus_sq(n.right, var):   # c/(1+x^2)
                flags["inv"] = True
                return f"({_rc(num)} * arctan {var})"
            if (num is not None and isinstance(n.right, ast.Name)
                    and n.right.id == var):                       # c/x -> c*ln x
                flags["inv"] = True
                return f"({_rc(num)} * ln {var})"
            return None
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and len(n.args) == 1):
            fn = n.func.id
            k, iso = _lin_arg(n.args[0], var)
            if iso is not None and fn in ("sin", "cos", "exp"):
                base = {"sin": f"(- cos ({iso}))", "cos": f"(sin ({iso}))",
                        "exp": f"(exp ({iso}))"}[fn]
                return base if k == 1 else f"({base} / {_rc(k)})"
            if (fn == "ln" and isinstance(n.args[0], ast.Name)
                    and n.args[0].id == var):
                return f"({var} * ln {var} - {var})"
            return None
        return None

    F = rec(node)
    if F is None:
        return None
    return F, flags["inv"]


def integral_recipe(node, goal_term: str):
    """Build an FTC theorem for a supported definite-integral equality.

    Polynomial integrands require numeral bounds and values and use a mechanically computed power-rule antiderivative. The other supported integrands use ``_antideriv`` and may have symbolic bounds or values.

    The explicit arguments to ``fundamental_theorem_of_calculus`` avoid slow higher-order unification after lambda eta-contraction. Isabelle checks that the candidate antiderivative has the required derivative, so an incorrect candidate cannot prove the claim.
    """
    from verl.utils.isabelle_utils.pyexpr import _const_eval
    from verl.utils.isabelle_utils.pyexpr import transpile as _tp
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)):
        return None
    intg = val = None
    for x, y in ((node.left, node.comparators[0]),
                 (node.comparators[0], node.left)):
        if (isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                and x.func.id == "integral"):
            intg, val = x, y
            break
    if (intg is None or len(intg.args) != 4
            or not isinstance(intg.args[1], ast.Name)):
        return None
    var = intg.args[1].id
    Lv = f"(\\<lambda>{var}."
    try:
        # integration variable BOUND -> emitted as `var` (matches the lambda),
        # not pv_-prefixed.
        orig = _tp(intg.args[0], {var: "real"}, "real", {var: "real"})
    except Exception:  # noqa: BLE001
        return None
    terms = _poly_terms(intg.args[0], var)
    a, b, v = (_const_eval(intg.args[2]), _const_eval(intg.args[3]),
               _const_eval(val))
    if terms is not None and a is not None and b is not None and v is not None:
        # The polynomial branch requires numeral bounds and value. Reversed numeric bounds (a > b): the transpiled goal already carries the oriented form `- (integral {b..a} ...)`, so the FTC lines run over the ORDERED set and the final simp closes the explicit sign.
        body_ftc, F = _poly_emit(terms, var, 0), _poly_emit(terms, var, 1)
        lo_, hi_ = (a, b) if a <= b else (b, a)
        A_, B_ = _rc(lo_), _rc(hi_)
        eq_line = (f'  have eq: "{Lv} {orig}) = {Lv} {body_ftc})" '
                   f'by (simp add: fun_eq_iff algebra_simps)\n')
        simp_extra = ""
        # `unfolding eq` rewrites the goal's integrand FIRST so the u fact applies directly; passing eq as a simp fact instead (`using u eq by simp`) sends the lambda equation through the merged session's full simp set, which measured as a reliable 15s watchdog timeout (2026-07-18 bisect), while this form closes in ~1.5s.
        final = "  show ?thesis unfolding eq using u by simp\nqed"
    else:
        # The other supported integrands may use symbolic bounds, which are transpiled rather than evaluated here; numeric bounds are ordered like the polynomial branch (the oriented goal carries the sign).
        ad = _antideriv(intg.args[0], var)
        if ad is None:
            return None
        F, needs_inv = ad
        try:
            if a is not None and b is not None and a > b:
                A_, B_ = _rc(b), _rc(a)
            else:
                A_ = _tp(intg.args[2], {}, "real", {})
                B_ = _tp(intg.args[3], {}, "real", {})
        except Exception:  # noqa: BLE001
            return None
        body_ftc, eq_line = orig, ""
        simp_extra = " inverse_eq_divide" if needs_inv else ""
        final = "  show ?thesis using u by (simp add: field_simps)\nqed"
    return (
        f'theorem chk:\n  shows "{goal_term}"\nproof -\n'
        f'{eq_line}'
        f'  have "({Lv} {body_ftc}) has_integral '
        f'({Lv} {F}) {B_} - {Lv} {F}) {A_})) {{{A_}..{B_}}}"\n'
        f'    apply (rule fundamental_theorem_of_calculus'
        f'[of "{A_}" "{B_}" "{Lv} {F})" "{Lv} {body_ftc})"])\n     apply simp\n'
        f'    apply (auto intro!: derivative_eq_intros simp: '
        f'has_real_derivative_iff_has_vector_derivative[symmetric]{simp_extra})'
        f'\n    done\n'
        f'  hence u: "integral {{{A_}..{B_}}} {Lv} {body_ftc}) = '
        f'({Lv} {F}) {B_} - {Lv} {F}) {A_})" by (rule integral_unique)\n'
        + final)
