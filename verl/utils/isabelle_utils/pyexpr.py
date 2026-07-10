#!/usr/bin/env python3
"""Constrained Python expression -> Isabelle/HOL term transpiler (v5).

The judge writes plain Python boolean expressions (a language it has seen
orders of magnitude more often than Isabelle); we validate them against a
strict AST whitelist and transpile mechanically. All mechanical checks
(number windows, transcription, vacuity) read the AST, not regexes.

Design points:
- `/` is real division; `//` is integer division (div); `%` is mod (int).
- `**` becomes ^ for literal non-negative integer exponents, powr otherwise
  (real base).
- fact/comb are nat-valued in Isabelle: the transpiler wraps them in the
  carrier type conversion automatically, which removes the int/nat clash
  failure class wholesale.
- Floats are transpiled to exact fractions (0.025 -> 25 / 1000), so decimal
  constants stay faithful and computable.
"""
import ast
from fractions import Fraction


class PyExprError(ValueError):
    pass


# func name -> (isabelle name, arity); None handled specially
FUNCS = {
    "sqrt": ("sqrt", 1), "abs": ("abs", 1), "floor": ("floor", 1),
    "ceil": ("ceiling", 1), "ceiling": ("ceiling", 1),
    "gcd": ("gcd", 2), "lcm": ("lcm", 2),
    "factorial": ("fact", 1), "fact": ("fact", 1),
    "comb": (None, 2), "choose": (None, 2),
    "log": ("log", 2), "ln": ("ln", 1), "exp": ("exp", 1),
    "sin": ("sin", 1), "cos": ("cos", 1), "tan": ("tan", 1),
    "min": ("min", 2), "max": ("max", 2),
}
REAL_ONLY_FUNCS = {"sqrt", "log", "ln", "exp", "sin", "cos", "tan"}
NAT_VALUED_FUNCS = {"factorial", "fact", "comb", "choose"}

CMP = {ast.Eq: "=", ast.NotEq: "~=", ast.Lt: "<", ast.LtE: "<=",
       ast.Gt: ">", ast.GtE: ">="}


def parse_expr(src: str) -> ast.expr:
    s = src.strip()
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise PyExprError(f"not a valid Python expression ({e.msg})")
    # Preserve the ORIGINAL decimal text of float literals (2026-07-11):
    # ast.parse converts to IEEE float first, so a literal longer than float
    # precision (0.12345678901234567890) silently rounds before we ever see
    # it. Fraction(<source text>) is exact; annotate the node so analyze()
    # and transpile() use the faithful value.
    for n in ast.walk(tree.body):
        if isinstance(n, ast.Constant) and isinstance(n.value, float):
            try:
                seg = ast.get_source_segment(s, n)
            except Exception:  # noqa: BLE001
                seg = None
            if seg:
                try:
                    n._frac_val = Fraction(seg.strip())
                except (ValueError, ZeroDivisionError):
                    pass
    return tree.body


def _frac(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PyExprError(f"unsupported constant {value!r}")
    return Fraction(str(value))


def _const_frac(n: "ast.Constant") -> Fraction:
    """Faithful Fraction of a Constant node: prefers the exact source-text
    annotation from parse_expr over the (possibly rounded) float value."""
    v = getattr(n, "_frac_val", None)
    if v is not None:
        return v
    return _frac(n.value)


def analyze(node):
    """Collect (identifiers, constant values, needs_real) over the tree;
    raise PyExprError on any non-whitelisted construct."""
    idents, consts = set(), set()
    needs_real = False

    def walk(n):
        nonlocal needs_real
        if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            for v in n.values:
                walk(v)
        elif isinstance(n, ast.UnaryOp) and isinstance(
                n.op, (ast.USub, ast.UAdd, ast.Not)):
            walk(n.operand)
        elif isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Div):
                needs_real = True
            elif isinstance(n.op, ast.Pow):
                e = n.right
                lit_nat = (isinstance(e, ast.Constant)
                           and isinstance(e.value, int) and e.value >= 0)
                if not lit_nat:
                    needs_real = True
            elif not isinstance(n.op, (ast.Add, ast.Sub, ast.Mult,
                                       ast.Mod, ast.FloorDiv)):
                raise PyExprError(f"operator {type(n.op).__name__} "
                                  "not allowed")
            walk(n.left)
            walk(n.right)
        elif isinstance(n, ast.Compare):
            for op in n.ops:
                if type(op) not in CMP:
                    raise PyExprError(f"comparison {type(op).__name__} "
                                      "not allowed")
            walk(n.left)
            for c in n.comparators:
                walk(c)
        elif isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in FUNCS:
                name = getattr(getattr(n, "func", None), "id", "?")
                raise PyExprError(
                    f"call to '{name}' not allowed; only "
                    f"{sorted(FUNCS)} - a quantity like a_n is a plain "
                    "variable, never a(n)")
            if len(n.args) != FUNCS[n.func.id][1] or n.keywords:
                raise PyExprError(f"{n.func.id} takes exactly "
                                  f"{FUNCS[n.func.id][1]} argument(s)")
            if n.func.id in REAL_ONLY_FUNCS:
                needs_real = True
            for a in n.args:
                walk(a)
        elif isinstance(n, ast.Name):
            idents.add(n.id)
        elif isinstance(n, ast.Constant):
            v = _const_frac(n)
            consts.add(v)
            if isinstance(n.value, float):
                needs_real = True
        else:
            raise PyExprError(
                f"{type(n).__name__} not allowed (no tuples, lists, sets, "
                "subscripts, lambdas or comprehensions; introduce plain "
                "variables instead)")

    walk(node)
    return idents, consts, needs_real


def _const_eval(n):
    """Fraction value of a pure-constant subtree, else None."""
    if isinstance(n, ast.Constant):
        try:
            return _const_frac(n)
        except PyExprError:
            return None
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _const_eval(n.operand)
        return None if v is None else (-v if isinstance(n.op, ast.USub)
                                       else v)
    if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        a, b = _const_eval(n.left), _const_eval(n.right)
        if a is None or b is None:
            return None
        try:
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if b.denominator == 1 and abs(b.numerator) <= 64:
                return a ** b.numerator
        except (ZeroDivisionError, OverflowError):
            return None
    return None


def transpile(node, var_types: dict, carrier: str) -> str:
    """Emit a fully parenthesized Isabelle term. var_types: name -> int/real;
    carrier: the numeral annotation type for this proposition."""

    def num(value):
        f = _frac(value)
        ty = "real" if (f.denominator != 1 or carrier == "real") else carrier
        if f.denominator == 1:
            return f"({f.numerator}::{ty})"
        return f"(({f.numerator}::real) / ({f.denominator}::real))"

    def t(n, exponent=False):
        if isinstance(n, ast.BoolOp):
            op = " & " if isinstance(n.op, ast.And) else " | "
            return "(" + op.join(t(v) for v in n.values) + ")"
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return f"(~ {t(n.operand)})"
            sign = "-" if isinstance(n.op, ast.USub) else ""
            return f"({sign}{t(n.operand)})"
        if isinstance(n, ast.BinOp):
            a, b = t(n.left), t(n.right)
            if isinstance(n.op, ast.Pow):
                # constant-foldable integer exponents stay computable: ^ for
                # k >= 0, exact reciprocal for k < 0; powr only as last resort
                ev = _const_eval(n.right)
                if ev is not None and ev.denominator == 1:
                    k = ev.numerator
                    if k >= 0:
                        return f"({a} ^ {k})"
                    return f"(1 / ({a} ^ {abs(k)}))"
                # int-valued variable exponents (2 ** (T + 1) with T int)
                # stay in computable ^; powr would freeze simp/eval (the
                # OlympiadBench power-identity failure class).
                # 2026-07-11 soundness fix: a bare `nat e` silently maps a
                # NEGATIVE exponent to 0 (2^(T-5) with T=3 became 2^0 = 1,
                # changing the math). Emit a sign-guarded form instead --
                # correct for both signs and still simp/eval-computable.
                try:
                    eids, _, ereal = analyze(n.right)
                except PyExprError:
                    eids, ereal = set(), True
                if not ereal and eids and all(
                        var_types.get(i) == "int" for i in eids):
                    e_t = t(n.right)
                    if carrier == "real":
                        return (f"(if {e_t} >= 0 then ({a} ^ (nat {e_t})) "
                                f"else (1 / ({a} ^ (nat (- {e_t})))))")
                    # A possibly-negative exponent under a non-real carrier
                    # has no faithful integer encoding (a^-k is fractional);
                    # fail closed rather than silently change the meaning.
                    # (Unreachable via py_to_isabelle: a symbolic exponent
                    # forces needs_real -> real carrier.)
                    raise PyExprError(
                        "symbolic integer exponent requires a real carrier "
                        "(may be negative)")
                return f"({a} powr {t(n.right)})"
            if isinstance(n.op, ast.Mod):
                return f"({a} mod {b})"
            if isinstance(n.op, ast.FloorDiv):
                return f"({a} div {b})"
            sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
                   ast.Div: "/"}[type(n.op)]
            return f"({a} {sym} {b})"
        if isinstance(n, ast.Compare):
            if len(n.ops) == 1:
                return (f"({t(n.left)} {CMP[type(n.ops[0])]} "
                        f"{t(n.comparators[0])})")
            parts, left = [], n.left
            for op, right in zip(n.ops, n.comparators):
                parts.append(f"({t(left)} {CMP[type(op)]} {t(right)})")
                left = right
            return "(" + " & ".join(parts) + ")"
        if isinstance(n, ast.Call):
            fname = n.func.id
            if fname in ("comb", "choose"):
                a, b = (t(x) for x in n.args)
                inner = f"((nat {a}) choose (nat {b}))" \
                    if carrier != "nat" else f"({a} choose {b})"
                return f"({carrier} {inner})" if carrier != "nat" else inner
            iname, _ = FUNCS[fname]
            args = " ".join(t(a) for a in n.args)
            if fname in ("factorial", "fact") and carrier != "nat":
                return f"({carrier} (fact (nat {t(n.args[0])})))"
            if fname == "floor" or fname in ("ceil", "ceiling"):
                # floor/ceiling :: real => int; lift back to the carrier
                if carrier == "real":
                    return f"(real_of_int ({iname} {args}))"
                return f"({iname} {args})"
            return f"({iname} {args})"
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Constant):
            return num(_const_frac(n))   # exact source-text value if annotated
        raise PyExprError(f"untranspilable node {type(n).__name__}")

    return t(node)


def py_to_isabelle(src: str, var_types: dict):
    """Full pipeline: parse, analyze, choose carrier, transpile.

    Returns (isabelle_term, idents, consts, carrier).
    """
    node = parse_expr(src)
    idents, consts, needs_real = analyze(node)
    if needs_real or any(var_types.get(i) == "real" for i in idents):
        carrier = "real"
    else:
        carrier = "int"
    term = transpile(node, var_types, carrier)
    return term, idents, consts, carrier


if __name__ == "__main__":
    VT = {"red": "int", "blue": "int", "x": "real", "answer": "int"}
    CASES = [
        ("red == 17", None),
        ("answer == red + blue", None),
        ("5 * x - 7 == 2 * x + 11", None),
        ("abs(5 * x - 1) == x + 3", None),
        ("answer == comb(9, 4)", None),
        ("3 ** (2 * 2) + 19 == 81 + 19", None),
        ("x ** 0.5 == 2", None),
        ("21 * 4 == 84 and 84 // 12 == 7", None),
        ("6.6 * 10 ** -27 > 0", None),
        ("100 % 7 == 2", None),
        ("a[1] == 2", PyExprError),
        ("f(x) == 3", PyExprError),
        ("(lambda v: v)(3) == 3", PyExprError),
        ("{1, 2} == {2, 1}", PyExprError),
    ]
    for src, expect in CASES:
        try:
            term, ids, cs, carrier = py_to_isabelle(src, VT)
            status = "ERR-MISSED" if expect else "ok"
            print(f"[{status}] {src}\n      -> ({carrier}) {term}")
        except PyExprError as e:
            status = "ok-rejected" if expect else "UNEXPECTED-REJECT"
            print(f"[{status}] {src}\n      !! {e}")
