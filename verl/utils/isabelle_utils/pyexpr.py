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
from decimal import Decimal, InvalidOperation
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
    "prime": ("prime", 1),
    # Pigeonhole meta-functions (map to `min_rep` in Math_Verify_Base):
    #   guarantee_pair(n)    = fewest draws over n categories forcing TWO in one
    #                          category = n + 1        (= min_rep n 2)
    #   guarantee_same(n, k) = fewest draws forcing k in one category
    #                          = n*(k-1) + 1           (= min_rep n k)
    # min_rep's closed form is CERTIFIED by the proven `min_rep_correct`
    # (sufficiency AND minimality), so the judge only supplies the category
    # count (and k); the pigeonhole content is fixed and sound -- it cannot be
    # invented or gotten wrong.
    "guarantee_pair": ("min_rep", 1),
    "guarantee_same": ("min_rep", 2),
}
REAL_ONLY_FUNCS = {"sqrt", "log", "ln", "exp", "sin", "cos", "tan"}
NAT_VALUED_FUNCS = {"factorial", "fact", "comb", "choose",
                    "guarantee_pair", "guarantee_same"}
# Uninterpreted-function names. `f(...)`, `g(...)`, `h(...)` (never in FUNCS)
# are UNDEFINED functions constrained only by equations -- represented as
# `fixes f :: "real => ... => real"` and applied as `f arg`. Used for
# functional equations (f(x+y)=f(x)+f(y), f(log b x)=x). A single defined
# function is still inlined by the judge; these are the undefined ones.
FUNC_VARS = {"f", "g", "h"}


def _pv(name: str) -> str:
    """Prefix a free-variable / uninterpreted-function name so it can never
    collide with an Isabelle constant. The Math_Verify session (HOL-Analysis
    + Probability) defines constants like `AE` (almost-everywhere); a geometry
    segment named `AE` would otherwise parse as that constant and the theorem
    would fail. The AST-level logic (answer-exclusion, def/claim split, number
    windows) keeps using the ORIGINAL names -- only the emitted Isabelle term
    and its fixes carry the prefix, consistently."""
    return "pv_" + name


# Mathematical constants that map to the Isabelle constant, NOT a pv_-prefixed
# free variable: `pi` is Isabelle's real pi, so tan(pi - a) = -tan(a) and the
# quadrant bound pi/2 < a < pi mean what they should (as a free `pv_pi` they
# were unconstrained and every trig identity/quadrant fact silently failed).
# `e` is deliberately excluded -- it is far too often an ordinary variable.
CONSTANTS = {"pi": "pi"}
# Boolean-valued predicates: emit a bool atom, not a numeric value. `prime`
# is defined on any factorial_semiring, so `prime (n::int)` is well typed and
# `eval` decides it for a concrete n. A real argument is meaningless, so we
# fail closed when the proposition carrier is real (see transpile()).
BOOL_FUNCS = {"prime"}

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
                text = seg.strip()
                try:
                    n._frac_val = Fraction(text)
                    # Preserve written precision for tolerance semantics.
                    # Decimal handles trailing zeros and scientific notation:
                    # 0.250 -> 3 places; 2.50e-3 -> 5 decimal places.
                    dec = Decimal(text)
                    n._decimal_text = text
                    n._decimal_places = max(0, -dec.as_tuple().exponent)
                except (ValueError, ZeroDivisionError, InvalidOperation):
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


def _check_aggregate(n: "ast.Call"):
    """Validate a sum()/prod() aggregate node's shape; raise otherwise.
    Accepted: sum(<expr> for <var> in range(lo[, hi]))  -- one generator, a
    plain index variable, no `if` filter, a literal/expression range with 1-2
    args (no step). Returns (comp, generator, [range args])."""
    fn = n.func.id
    if len(n.args) != 1 or n.keywords or not isinstance(
            n.args[0], (ast.GeneratorExp, ast.ListComp)):
        raise PyExprError(
            f"{fn}() must wrap one generator: {fn}(<expr> for k in "
            "range(lo, hi))")
    comp = n.args[0]
    if len(comp.generators) != 1:
        raise PyExprError(f"{fn}() takes exactly one 'for' clause")
    g = comp.generators[0]
    if (not isinstance(g.target, ast.Name) or g.ifs
            or getattr(g, "is_async", 0)
            or not isinstance(g.iter, ast.Call)
            or not isinstance(g.iter.func, ast.Name)
            or g.iter.func.id != "range"):
        raise PyExprError(
            f"{fn}() index must be 'for <var> in range(lo, hi)' with no "
            "condition")
    ra = g.iter.args
    if not (1 <= len(ra) <= 2) or g.iter.keywords:
        raise PyExprError("range() takes 1 or 2 arguments (step unsupported)")
    return comp, g, ra


def analyze(node):
    """Collect (identifiers, constant values, needs_real) over the tree;
    raise PyExprError on any non-whitelisted construct. `bound` carries the
    lambda-bound aggregate indices (sum/prod), which are NOT free variables."""
    idents, consts = set(), set()
    needs_real = False

    def walk(n, bound=frozenset()):
        nonlocal needs_real
        if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            for v in n.values:
                walk(v, bound)
        elif isinstance(n, ast.UnaryOp) and isinstance(
                n.op, (ast.USub, ast.UAdd, ast.Not)):
            walk(n.operand, bound)
        elif isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Div):
                needs_real = True
            elif isinstance(n.op, ast.Pow):
                e = n.right
                lit_nat = (isinstance(e, ast.Constant)
                           and isinstance(e.value, int) and e.value >= 0)
                # A bound (nat) index used directly as an exponent stays a
                # computable nat power, so it must NOT force a real carrier.
                bound_exp = isinstance(e, ast.Name) and e.id in bound
                if not lit_nat and not bound_exp:
                    needs_real = True
            elif not isinstance(n.op, (ast.Add, ast.Sub, ast.Mult,
                                       ast.Mod, ast.FloorDiv)):
                raise PyExprError(f"operator {type(n.op).__name__} "
                                  "not allowed")
            walk(n.left, bound)
            walk(n.right, bound)
        elif isinstance(n, ast.Compare):
            for op in n.ops:
                if type(op) not in CMP:
                    raise PyExprError(f"comparison {type(op).__name__} "
                                      "not allowed")
            walk(n.left, bound)
            for c in n.comparators:
                walk(c, bound)
        elif isinstance(n, ast.Call):
            fid = n.func.id if isinstance(n.func, ast.Name) else None
            if fid in ("sum", "prod"):
                _comp, g, ra = _check_aggregate(n)
                for b in ra:
                    walk(b, bound)               # bounds: outer scope
                walk(_comp.elt, bound | {g.target.id})  # body: index bound
                return
            if fid in ("forall", "exists"):
                if (len(n.args) != 2 or n.keywords
                        or not isinstance(n.args[0], ast.Name)):
                    raise PyExprError(
                        f"{fid}(var, body): first arg is the bound variable")
                walk(n.args[1], bound | {n.args[0].id})   # body: var bound
                return
            if fid == "implies":
                if len(n.args) != 2 or n.keywords:
                    raise PyExprError("implies(a, b) takes exactly 2 arguments")
                for a in n.args:
                    walk(a, bound)
                return
            if fid in ("integral", "deriv", "limit"):
                # integral(body, x, a, b) | deriv(body, x, point, value)
                # | limit(body, x, point, value): arg 2 is the bound variable.
                if (len(n.args) != 4 or n.keywords
                        or not isinstance(n.args[1], ast.Name)):
                    raise PyExprError(
                        f"{fid}(body, var, _, _): 4 args, arg 2 is the variable")
                needs_real = True
                v = n.args[1].id
                walk(n.args[0], bound | {v})         # body: var bound
                walk(n.args[2], bound)
                walk(n.args[3], bound)
                return
            if fid in FUNC_VARS and fid not in FUNCS:
                # uninterpreted function application f(...): real-valued, so
                # the enclosing proposition needs a real carrier; the name is
                # NOT a scalar identifier (handled by func_arities()).
                needs_real = True
                for a in n.args:
                    walk(a, bound)
                return
            if fid is None or fid not in FUNCS:
                raise PyExprError(
                    f"call to '{fid}' not allowed; only {sorted(FUNCS)} "
                    f"(plus sum/prod/forall and functions {sorted(FUNC_VARS)}) "
                    "- a quantity like a_n is a plain variable, never a(n)")
            if len(n.args) != FUNCS[fid][1] or n.keywords:
                raise PyExprError(f"{fid} takes exactly "
                                  f"{FUNCS[fid][1]} argument(s)")
            if fid in REAL_ONLY_FUNCS:
                needs_real = True
            for a in n.args:
                walk(a, bound)
        elif isinstance(n, ast.Name):
            if n.id in CONSTANTS:
                needs_real = True          # pi is real-valued
            elif n.id not in bound:
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


def func_arities(node) -> dict:
    """Map each applied uninterpreted-function name (f/g/h) to its arity, so
    the engine can declare `fixes f :: "real => ... => real"`. A function name
    is identified purely by being APPLIED (f(...)); a bare `f` is a scalar."""
    arities = {}
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in FUNC_VARS and n.func.id not in FUNCS):
            arities[n.func.id] = len(n.args)
    return arities


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


def is_nonlinear(node) -> bool:
    """True if the expression is nonlinear in its variables: a variable base
    raised to a literal power >= 2, a product of two variable-bearing factors,
    or a variable in a denominator. These are exactly the goals `sos` can
    discharge but `linarith`/`presburger` cannot, so the engine routes them to
    an sos rescue. (`const ** var` is powr, handled separately, not flagged;
    `2 * x` has a literal factor, so it stays linear.)"""
    def has_var(n):
        # A function name (the `func` of a Call, e.g. comb in comb(8,2)) is a
        # Name node but NOT a variable -- exclude it so comb(5,2)/comb(8,2)
        # reads as the constant it is, not a variable in a denominator.
        func_ids = {id(x.func) for x in ast.walk(n)
                    if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)}
        return any(isinstance(x, ast.Name) and id(x) not in func_ids
                   for x in ast.walk(n))
    for n in ast.walk(node):
        if not isinstance(n, ast.BinOp):
            continue
        if isinstance(n.op, ast.Pow):
            ev = _const_eval(n.right)
            if (has_var(n.left) and ev is not None
                    and ev.denominator == 1 and ev.numerator >= 2):
                return True
        elif isinstance(n.op, ast.Mult):
            if has_var(n.left) and has_var(n.right):
                return True
        elif isinstance(n.op, ast.Div):
            if has_var(n.right):
                return True
    return False


def is_linear_arith(node) -> bool:
    """True iff `node` (a boolean premise expression) is affine in its
    variables with constant coefficients -- the fragment where `linarith`
    (reals) / `presburger` (integers) are COMPLETE decision procedures, so a
    fast linarith/presburger consistency probe over such premises is SOUND: it
    can never miss a contradiction a heavier tactic would find. STRICTER than
    `not is_nonlinear`: it also rejects any opaque/transcendental function
    applied to a variable (sin(x), sqrt(x), log(b, x)), a variable modulus or
    floor-division, and a variable exponent -- linarith treats those as free
    atoms, so a hidden contradiction there (sin(x) == 2) would be missed by the
    probe yet exploitable ex-falso by the step-proof's stronger tactics. A
    function of only CONSTANT arguments (sqrt(2), comb(5, 2)) is a constant atom
    and stays linear. Returns False on anything it cannot certify linear (so the
    caller falls back to the heavy ALTERNATION probe)."""
    def has_var(n):
        func_ids = {id(x.func) for x in ast.walk(n)
                    if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)}
        return any(isinstance(x, ast.Name) and id(x) not in func_ids
                   for x in ast.walk(n))

    def lin_term(n):
        if isinstance(n, (ast.Constant, ast.Name)):
            return True
        if isinstance(n, ast.UnaryOp) and isinstance(
                n.op, (ast.UAdd, ast.USub)):
            return lin_term(n.operand)
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.Add, ast.Sub)):
                return lin_term(n.left) and lin_term(n.right)
            if isinstance(n.op, ast.Mult):
                # affine only if a constant scales a linear term
                if not has_var(n.left):
                    return lin_term(n.right)
                if not has_var(n.right):
                    return lin_term(n.left)
                return False
            if isinstance(n.op, ast.Div):
                return (not has_var(n.right)) and lin_term(n.left)
            if isinstance(n.op, ast.Pow):
                return not has_var(n)          # a constant power is a constant
            return False                       # Mod, FloorDiv, ... -> nonlinear
        if isinstance(n, ast.Call):
            return not has_var(n)              # constant-arg call = constant
        return False

    def lin_bool(n):
        if isinstance(n, ast.BoolOp):
            return all(lin_bool(v) for v in n.values)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return lin_bool(n.operand)
        if isinstance(n, ast.Compare):
            if any(isinstance(o, (ast.In, ast.NotIn, ast.Is, ast.IsNot))
                   for o in n.ops):
                return False
            return lin_term(n.left) and all(
                lin_term(c) for c in n.comparators)
        return False   # implies()/forall()/prime()/a bare term -> not linear

    try:
        return lin_bool(node)
    except Exception:
        return False


def transpile(node, var_types: dict, carrier: str, bound=None) -> str:
    """Emit a fully parenthesized Isabelle term. var_types: name -> int/real;
    carrier: the numeral annotation type for this proposition. `bound` maps
    names to 'nat'/'real' for pre-bound variables (a lambda body, e.g. the
    integral recipe transpiling `3*x**2` with x already bound so it stays `x`
    and is not pv_-prefixed as a free variable)."""

    def num(value):
        f = _frac(value)
        ty = "real" if (f.denominator != 1 or carrier == "real") else carrier
        if f.denominator == 1:
            return f"({f.numerator}::{ty})"
        return f"(({f.numerator}::real) / ({f.denominator}::real))"

    def _coerce_bound(name, bound):
        # A nat-bound aggregate index (sum/prod) is lifted into the carrier
        # where used as a value; a real-bound forall variable is already real
        # and is emitted bare. (A nat index used as an exponent is handled in
        # Pow, where it must stay nat.)
        if bound.get(name) == "real":
            return name
        if carrier == "real":
            return f"(real {name})"
        if carrier == "int":
            return f"(int {name})"
        return name

    def _nat_bound(nd, bound):
        # A range endpoint must be nat: {a..<b} over nat is the finite index
        # set range(a, b); over real it is a (useless) continuous interval.
        # A symbolic endpoint (e.g. n+1) is inherently integral, so emit it
        # under an INT carrier -- not the proposition carrier, which may be
        # real and would produce the ill-typed `nat (n + (1::real))`.
        ev = _const_eval(nd)
        if ev is not None and ev.denominator == 1 and ev.numerator >= 0:
            return f"({ev.numerator}::nat)"
        return f"(nat {transpile(nd, var_types, 'int')})"

    def t(n, bound=None):
        if bound is None:
            bound = {}          # name -> 'nat' (sum index) | 'real' (forall var)
        if isinstance(n, ast.BoolOp):
            op = " & " if isinstance(n.op, ast.And) else " | "
            return "(" + op.join(t(v, bound) for v in n.values) + ")"
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return f"(~ {t(n.operand, bound)})"
            sign = "-" if isinstance(n.op, ast.USub) else ""
            return f"({sign}{t(n.operand, bound)})"
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Pow):
                # A nat-bound index directly as the exponent stays a
                # computable nat power (2**k -> base ^ k), never powr.
                if (isinstance(n.right, ast.Name)
                        and bound.get(n.right.id) == "nat"):
                    return f"({t(n.left, bound)} ^ {n.right.id})"
                # The base/result use the proposition carrier. A symbolic
                # integer exponent must be emitted independently as int;
                # reusing the real carrier produces ill-typed terms such as
                # T - (5::real) when T :: int.
                a = t(n.left, bound)
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
                    e_t = transpile(n.right, var_types, "int")
                    if carrier == "real":
                        return (f"(if {e_t} >= (0::int) then "
                                f"({a} ^ (nat {e_t})) else "
                                f"(1 / ({a} ^ (nat (- {e_t})))))")
                    # A possibly-negative exponent under a non-real carrier
                    # has no faithful integer encoding (a^-k is fractional);
                    # fail closed rather than silently change the meaning.
                    # (Unreachable via py_to_isabelle: a symbolic exponent
                    # forces needs_real -> real carrier.)
                    raise PyExprError(
                        "symbolic integer exponent requires a real carrier "
                        "(may be negative)")
                return f"({a} powr {t(n.right, bound)})"
            a, b = t(n.left, bound), t(n.right, bound)
            if isinstance(n.op, ast.Mod):
                return f"({a} mod {b})"
            if isinstance(n.op, ast.FloorDiv):
                return f"({a} div {b})"
            sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
                   ast.Div: "/"}[type(n.op)]
            return f"({a} {sym} {b})"
        if isinstance(n, ast.Compare):
            if len(n.ops) == 1:
                return (f"({t(n.left, bound)} {CMP[type(n.ops[0])]} "
                        f"{t(n.comparators[0], bound)})")
            parts, left = [], n.left
            for op, right in zip(n.ops, n.comparators):
                parts.append(f"({t(left, bound)} {CMP[type(op)]} "
                             f"{t(right, bound)})")
                left = right
            return "(" + " & ".join(parts) + ")"
        if isinstance(n, ast.Call):
            fname = n.func.id
            if fname in ("sum", "prod"):
                # Finite aggregate over range(lo, hi):
                #   sum(<expr> for k in range(lo, hi)) ->
                #   (sum (%k. <expr>) {lo..<hi})   with k :: nat.
                comp, g, ra = _check_aggregate(n)
                k = g.target.id
                if len(ra) == 1:
                    lo_s, hi_s = "(0::nat)", _nat_bound(ra[0], bound)
                else:
                    lo_s = _nat_bound(ra[0], bound)
                    hi_s = _nat_bound(ra[1], bound)
                body = t(comp.elt, {**bound, k: "nat"})
                agg = "sum" if fname == "sum" else "prod"
                return f"({agg} (%{k}. {body}) {{{lo_s}..<{hi_s}}})"
            if fname in ("forall", "exists"):
                # forall/exists(x, body) -> quantifier; the bound variable is
                # real (the domain of an uninterpreted function).
                var = n.args[0].id
                body = t(n.args[1], {**bound, var: "real"})
                q = "\\<forall>" if fname == "forall" else "\\<exists>"
                return f"({q}{var}. {body})"
            if fname == "implies":
                return (f"({t(n.args[0], bound)} \\<longrightarrow> "
                        f"{t(n.args[1], bound)})")
            if fname in ("integral", "deriv", "limit"):
                v = n.args[1].id
                body = t(n.args[0], {**bound, v: "real"})
                p2 = t(n.args[2], bound)     # a (integral) / point (deriv,limit)
                p3 = t(n.args[3], bound)     # b (integral) / value (deriv,limit)
                if fname == "integral":
                    return f"(integral {{{p2}..{p3}}} (%{v}. {body}))"
                if fname == "deriv":
                    return (f"(((%{v}. {body}) has_real_derivative ({p3})) "
                            f"(at ({p2})))")
                return (f"(((%{v}. {body}) \\<longlongrightarrow> ({p3})) "
                        f"(at ({p2})))")
            if fname in FUNC_VARS and fname not in FUNCS:
                # uninterpreted function application f(a, b) -> (pv_f a b)
                fargs = " ".join(t(a, bound) for a in n.args)
                return f"({_pv(fname)} {fargs})"
            if fname == "prime":
                # Boolean predicate. `prime` needs an integer argument; under a
                # real carrier the term is ill typed, so fail closed rather than
                # emit `prime (x::real)` (no false reward, no false negative on
                # a genuinely integer step).
                if carrier == "real":
                    raise PyExprError("prime() requires an integer argument")
                return f"(prime ({t(n.args[0], bound)}))"
            if fname in ("comb", "choose"):
                a, b = (t(x, bound) for x in n.args)
                inner = f"((nat {a}) choose (nat {b}))" \
                    if carrier != "nat" else f"({a} choose {b})"
                return f"({carrier} {inner})" if carrier != "nat" else inner
            if fname in ("guarantee_pair", "guarantee_same"):
                # min_rep :: nat => nat => nat. guarantee_pair(n) fixes k = 2.
                a = t(n.args[0], bound)
                kk = t(n.args[1], bound) if fname == "guarantee_same" else "2"
                inner = f"(min_rep (nat {a}) (nat {kk}))" \
                    if carrier != "nat" else f"(min_rep {a} {kk})"
                return f"({carrier} {inner})" if carrier != "nat" else inner
            iname, _ = FUNCS[fname]
            args = " ".join(t(a, bound) for a in n.args)
            if fname in ("factorial", "fact") and carrier != "nat":
                return f"({carrier} (fact (nat {t(n.args[0], bound)})))"
            if fname == "floor" or fname in ("ceil", "ceiling"):
                # floor/ceiling :: real => int; lift back to the carrier
                if carrier == "real":
                    return f"(real_of_int ({iname} {args}))"
                return f"({iname} {args})"
            return f"({iname} {args})"
        if isinstance(n, ast.Name):
            if n.id in bound:
                return _coerce_bound(n.id, bound)
            if n.id in CONSTANTS:
                return CONSTANTS[n.id]   # pi -> Isabelle's real pi, unprefixed
            return _pv(n.id)             # free var: prefix to avoid const clash
        if isinstance(n, ast.Constant):
            return num(_const_frac(n))   # exact source-text value if annotated
        raise PyExprError(f"untranspilable node {type(n).__name__}")

    return t(node, bound)


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
