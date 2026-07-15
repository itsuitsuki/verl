"""IsabelleEngine: Isabelle/HOL step-level verification for math reasoning.

Lifted from scripts/isabelle_poc_math500/pipeline_v5.py. The judge writes
constrained Python boolean expressions; pyexpr.py validates the AST and
transpiles to Isabelle terms.
"""
import ast
import json
import os
import re
import threading
import time
from fractions import Fraction
from pathlib import Path

from verl.utils.isabelle_utils.server_pool import IsabelleServerPool
from verl.utils.isabelle_utils.tactics import (
    ALTERNATION, EVAL_RESCUE, FALSE_TACTIC, LINEAR_FALSE, LINEAR_CLAIM,
    NZ_TACTIC, SAFE_DANGEROUS,
    SOS_RESCUE, SMT_RESCUE, FIELD_RESCUE, ALGEBRA_RESCUE, TRIG_RESCUE,
    has_division, has_trig, has_poly, has_powr, EXP_RESCUE,
    DERIV_TACTIC, LIMIT_TACTIC, has_log, has_deriv,
    has_limit, has_integral_goal, integral_recipe, is_dangerous_isabelle,
    make_theorem, make_theorem_with_logs, make_theorem_unfold, num_values,
    range_ints, fixes_clause,
    identifiers, anchor_ground_numerals, FREE_NUMS,
)
from verl.utils.isabelle_utils.judge import translate
from verl.utils.isabelle_utils.xml_utils import (
    block_stats, boxed_answer, corrupt_steps, parse_xml_steps,
)
from verl.utils.isabelle_utils.pyexpr import (
    FUNCS, PyExprError, analyze, func_arities, is_nonlinear, is_linear_arith,
    parse_expr, py_to_isabelle, transpile, _pv,
)

_PKG_DIR = Path(__file__).parent.resolve()
FREE_NUMS = {Fraction(0), Fraction(1), Fraction(2)}
UNIT_CLOSURE = (Fraction(60), Fraction(100), Fraction(1000))


def _eval_values(node, vals):
    """Values of every numeric sub-expression of a TRANSLATED proposition under
    the pinned variable values `vals` -- side1 - side2 = 3, price * qty = 10,
    (a + b) / 2 = ... -- for ANY operator, bounded to what the proposition
    actually writes (not a blind allow-list). The faithfulness check on a
    conclusion's numbers uses this so a number the proposition COMPUTES (in
    variable form) counts as transcribed, whatever the operation."""
    out = set()

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            v = Fraction(n.value)
            out.add(v)
            return v
        if isinstance(n, ast.Name):
            return vals.get(n.id)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            op = type(n.op)
            try:
                if op is ast.Add:
                    r = a + b
                elif op is ast.Sub:
                    r = a - b
                elif op is ast.Mult:
                    r = a * b
                elif op is ast.Div:
                    r = a / b if b else None
                elif op is ast.FloorDiv:
                    r = a // b if b else None
                elif op is ast.Mod:
                    r = a % b if b else None
                elif op is ast.Pow:
                    r = (a ** b if b.denominator == 1 and abs(b) <= 20
                         else None)
                else:
                    r = None
            except (ZeroDivisionError, ValueError, OverflowError):
                r = None
            if r is not None:
                out.add(r)
            return r
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = ev(n.operand)
            if v is not None:
                out.add(-v)
                return -v
            return None
        # Compare / BoolOp / Call: recurse so their operand sub-expressions are
        # still evaluated (the comparison/boolean value itself is not numeric).
        for ch in ast.iter_child_nodes(n):
            ev(ch)
        return None

    ev(node)
    return out

PROMPTS_DIR = _PKG_DIR / "prompts"
PROMPT_GIVENS_PY = (PROMPTS_DIR / "translate_givens.txt").read_text(
    encoding="utf-8")
PROMPT_STEPS_PY = (PROMPTS_DIR / "translate_steps.txt").read_text(
    encoding="utf-8")

from dataclasses import dataclass, field


@dataclass
class IsabelleConfig:
    judge_url: str = "http://127.0.0.1:4873/v1"
    judge_model: str = "Qwen3.6-35B-A3B"
    max_model_len: int = 12288
    pool_workers: int = 32
    session: str = "HOL-Number_Theory"
    # Per-call Isabelle verification deadline (seconds). Aligns with the
    # Z3 path's fol_timeout: a stuck use_theories call past this limit is
    # treated as failure rather than blocking the reward computation.
    check_deadline: float = 60.0
    # Per-HTTP-request judge deadline (seconds). Wired from the reward
    # api_config's api_timeout (2026-07-11: previously hardcoded inside
    # judge.call_judge, so the configured value silently did nothing).
    # Operational value stays 240 -- wiring only, per user decision.
    api_timeout: float = 240.0
    # Per-worker poly-tree RSS cap (GB); Hydra config knob (not env), plumbed
    # to IsabelleServerPool. 12GB validated safe on the 300GB cgroup.
    rss_cap_gb: float = 12.0


EXAMPLE_LINES = {
    "red == 17", "blue == 29", "answer == red + blue",
    "p == 6.6 * 10 ** -27", "abs(5 * x - 1) == x + 3",
    "answer == x * sqrt(p)",
}


PLACEHOLDER_RE = re.compile(r"<[A-Za-z][\w |/-]*>")


def _normalize_src(p: str) -> str:
    """Math-habit leniency: ^ means power, a lone = means equality."""
    p = p.replace("^", "**").replace("−", "-")
    p = re.sub(r"(?<![=<>!*+\-/])=(?!=)", "==", p)
    return p


def _junk_line(p: str) -> bool:
    """Prose/placeholder lines the judge's commentary leaks around the real
    output (e.g. quoting `GIVEN: <one Python boolean expression>` or rule
    text). A junk line has a placeholder, or fails to parse AND contains no
    comparison - real-but-wrong expressions stay and get error feedback."""
    if not p or PLACEHOLDER_RE.search(p):
        return True
    if re.search(r"[=<>]", p):
        return False
    try:
        parse_expr(_normalize_src(p))
        return False
    except PyExprError:
        return True


def parse_givens_py(reply: str):
    lines = reply.splitlines()
    # the canonical block starts at the LAST VARS line; everything before it
    # is the model's commentary (which loves to quote GIVEN lines)
    last_vars = None
    for i, line in enumerate(lines):
        if re.match(r"\s*VARS\b", line, re.IGNORECASE):
            last_vars = i
    if last_vars is None:
        return None
    fixes, givens = [], []
    m = re.match(r"\s*VARS\b:?\s*(.+)", lines[last_vars], re.IGNORECASE)
    if m:
        for part in m.group(1).split(","):
            w = part.split()
            if len(w) == 2 and w[1] in ("int", "real", "nat"):
                fixes.append((w[0], "int" if w[1] == "nat" else w[1]))
    for line in lines[last_vars + 1:]:
        m = re.match(r"\s*GIVEN\b:?\s*(.+)", line.strip(), re.IGNORECASE)
        if not m:
            continue
        p = _normalize_src(m.group(1).strip().replace('"', ' ').strip())
        if _junk_line(p):
            continue
        if p and p not in givens and p not in EXAMPLE_LINES:
            givens.append(p)
    seen = set()
    fixes = [(n, t) for n, t in fixes if not (n in seen or seen.add(n))]
    if not fixes or not givens:
        return None
    return fixes, givens


STEP_LINE_RE = re.compile(
    r"\s*STEP\s*(\d+)\s*\|\s*(?:premises?:\s*(.*?)\s*\|\s*)?prop:\s*(.+)",
    re.IGNORECASE)


def parse_props_py(reply: str):
    """{k: {"premises": [src...], "prop": src}}; premises part optional."""
    out = {}
    for line in reply.splitlines():
        m = STEP_LINE_RE.match(line.strip())
        if not m:
            continue
        prop = _normalize_src(m.group(3).strip().replace('"', ' ').strip())
        if _junk_line(prop):
            continue
        prems = []
        for p in (m.group(2) or "").split(";"):
            p = _normalize_src(p.strip().replace('"', ' ').strip())
            if p and p != "-" and not _junk_line(p):
                prems.append(p)
        out[int(m.group(1))] = {"premises": prems, "prop": prop}
    return out or None


# ---------- AST-level checks ----------

def expr_info(src: str, var_types: dict):
    """(isabelle_term, idents, consts, carrier) or raise PyExprError."""
    return py_to_isabelle(src, var_types)


def conjuncts(src: str):
    node = parse_expr(src)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return node.values
    return [node]


def _node_carrier(node, var_types: dict, fallback: str = "real") -> str:
    """Choose the numeral carrier for one typed expression subtree."""
    try:
        ids, _, needs_real = analyze(node)
    except PyExprError:
        return fallback
    if needs_real or any(var_types.get(i) == "real" for i in ids):
        return "real"
    return "int"


def _transpile_conjunctive(src: str, var_types: dict):
    """Transpile top-level conjunctions with a carrier per conjunct.

    A proposition such as ``x / 2 == 1 and n % 2 == 0`` cannot use one global
    carrier: the first conjunct is real while the second must remain int.
    The combined term is used both as the current claim and as a later step's
    premise, so it must preserve both sorts at construction time.
    """
    node = parse_expr(src)
    nodes = (node.values if isinstance(node, ast.BoolOp)
             and isinstance(node.op, ast.And) else [node])
    terms = []
    carriers = []
    for part in nodes:
        carrier = _node_carrier(part, var_types)
        terms.append(transpile(part, var_types, carrier))
        carriers.append(carrier)
    term = "(" + " & ".join(terms) + ")" if len(terms) > 1 else terms[0]
    ids, consts, _ = analyze(node)
    carrier = carriers[0] if len(set(carriers)) == 1 else "mixed"
    return term, ids, consts, carrier


def is_vacuous_node(n) -> bool:
    if isinstance(n, ast.Constant) and n.value is True:
        return True
    return (isinstance(n, ast.Compare) and len(n.ops) == 1
            and isinstance(n.ops[0], ast.Eq)
            and ast.dump(n.left) == ast.dump(n.comparators[0]))


def pin_from(src: str, vals: dict):
    try:
        node = parse_expr(src)
    except PyExprError:
        return
    for c in (node.values if isinstance(node, ast.BoolOp) else [node]):
        if (isinstance(c, ast.Compare) and len(c.ops) == 1
                and isinstance(c.ops[0], ast.Eq)
                and isinstance(c.left, ast.Name)):
            from verl.utils.isabelle_utils.pyexpr import _const_eval
            v = _const_eval(c.comparators[0])
            if v is not None:
                vals[c.left.id] = v


def transcription_missing_py(judge_prop, model_conclusion, var_vals,
                             premise_text, context_srcs):
    """Anti-repair check: every number the MODEL's conclusion states must be
    PRESENT in the JUDGE's translated proposition, so the judge cannot silently
    fix a wrong number (write == 41 for a stated 40) into a passing check. A
    number counts as present if the proposition CONTAINS it literally OR
    COMPUTES it -- side1 - side2 = 3, price * qty = 10 (see _eval_values) -- so
    the variable form is accepted for any operator. Returns the model's numbers
    that the proposition neither writes nor computes."""
    from verl.utils.isabelle_utils.pyexpr import _const_eval, analyze as _an
    try:
        node = parse_expr(judge_prop)
    except PyExprError:
        return []
    ctx_dumps = set()
    for c_src in context_srcs:
        try:
            ctx_dumps.add(ast.dump(parse_expr(c_src)))
        except PyExprError:
            pass
    present_nums, idents = set(), set()
    for c in (node.values if isinstance(node, ast.BoolOp) else [node]):
        if is_vacuous_node(c) or ast.dump(c) in ctx_dumps:
            continue
        try:
            ids, cs, _ = _an(c)
        except PyExprError:
            continue
        idents |= ids
        present_nums |= cs
    present_nums |= {var_vals[i] for i in idents if i in var_vals}
    # A variable's subscript index counts as present: the conclusion says "the
    # third / sixth term" (3, 6) and the proposition names them a3 / a_6 -- the
    # digit lives in the variable name, which num_values does not see.
    for i in idents:
        present_nums |= {Fraction(m) for m in re.findall(r"\d+", i)}
    # Evaluate what the proposition actually COMPUTES (side1 - side2 = 3,
    # price * qty = 10): a conclusion result written in variable form still
    # counts as transcribed, for any operator -- general, and bounded to what
    # the proposition writes.
    present_nums |= _eval_values(node, var_vals)
    prem_nums = num_values(premise_text, words=True)
    con_text = re.sub(r"_\{?\d+\}?", "", model_conclusion)
    out = []
    for v in num_values(con_text, words=True):
        if v in FREE_NUMS or v in prem_nums:
            continue
        forms = {v}
        for c in UNIT_CLOSURE:
            forms.add(v * c)
            forms.add(v / c)
        if forms & present_nums:
            continue
        # An ordinal ("the third / sixth term") is read by num_values as the
        # fraction 1/3, 1/6 -- a position, not a value: if its reciprocal (the
        # position) is present as a subscript (a3, a6), it is transcribed.
        if v.numerator == 1 and Fraction(v.denominator) in present_nums:
            continue
        out.append(str(v))
    return out


def tolerance_goal(claims_nodes, var_types, carrier):
    """Rewrite decimal equalities using half a written decimal unit.

    Decimal value and precision come from source annotations installed by
    parse_expr(), not from repr(float): ``0.250`` therefore keeps three places
    and long/scientific literals retain their exact value.
    """
    parts, changed = [], False

    def decimal_info(node):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, float)
                and hasattr(node, "_frac_val")
                and hasattr(node, "_decimal_places")):
            return node._frac_val, int(node._decimal_places)
        return None

    def written_places(node):
        dec = decimal_info(node)
        if dec is not None:
            return dec[1]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return 0
        return None

    for c in claims_nodes:
        dec = None
        ccarrier = _node_carrier(c, var_types, carrier)
        if (isinstance(c, ast.Compare) and len(c.ops) == 1
                and isinstance(c.ops[0], ast.Eq)):
            sides = [c.left, c.comparators[0]]
            decimal_sides = [(i, decimal_info(s))
                             for i, s in enumerate(sides)
                             if decimal_info(s) is not None]
            if decimal_sides:
                idx, (val, decimal_places) = decimal_sides[0]
                all_places = [written_places(s) for s in sides]
                all_places = [p for p in all_places if p is not None]
                places = min(all_places) if all_places else decimal_places
                dec = (sides[1 - idx], val, places)
        if dec is None:
            parts.append(transpile(c, var_types, ccarrier))
            continue
        other, val, places = dec
        lhs_t = transpile(other, var_types, "real")
        tol_den = 2 * 10 ** places
        parts.append(f"(abs ({lhs_t} - (({val.numerator}::real) / "
                     f"({val.denominator}::real))) < "
                     f"(1::real) / ({tol_den}::real))")
        changed = True
    if not changed or not parts:
        return None
    return " & ".join(parts) if len(parts) > 1 else parts[0]


def nz_divisors(src: str, var_types: dict, carrier: str):
    """Isabelle terms of non-constant denominators (Div/Mod right sides)."""
    from verl.utils.isabelle_utils.pyexpr import _const_eval
    out = []
    try:
        node = parse_expr(src)
    except PyExprError:
        return out
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(
                n.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if _const_eval(n.right) is None:
                try:
                    out.append(transpile(n.right, var_types, carrier))
                except PyExprError:
                    pass
    return out


# ---------- quadrant-trig recipe (2026-07-14) ----------
# "angle a in a quadrant, tan a = T, find sin a / cos a". The value-independent
# meta-theorems (sin_from_tan_c{pos,neg}, cos_from_tan_c{pos,neg}, the sign
# lemmas cos_{pos,neg}_q*, and div_sqrt_eq) are PRECOMPILED into the Math_Verify
# session (Math_Verify_Base.thy). This recipe detects the pattern and emits a
# short self-contained theorem that instantiates them, so the answer -- rational
# (3/5) or irrational (2/sqrt 5) -- closes by simp evaluation. None if no match.
_TRIG_DISCH = ("(rule div_sqrt_eq; simp add: power2_eq_square real_sqrt_pow2 "
               "field_simps)")
# inclusive-quadrant table over multiples of pi: (lo, hi) -> (cos-sign, lemma)
_TRIG_QUAD = [((0.0, 0.5), ("pos", "cos_pos_q1")),
             ((0.5, 1.0), ("neg", "cos_neg_q2")),
             ((1.0, 1.5), ("neg", "cos_neg_q3")),
             ((1.5, 2.0), ("pos", "cos_pos_q4"))]


def _pi_mult(node):
    """A quadrant bound as a multiple of pi (pi->1, so pi/2->0.5); None if the
    expression is not a clean rational multiple of pi (or a bare 0)."""
    def ev(n):
        if isinstance(n, ast.Constant):
            return float(n.value) if isinstance(n.value, (int, float)) else None
        if isinstance(n, ast.Name):
            return 1.0 if n.id == "pi" else None
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b if b else None
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            return None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = ev(n.operand)
            return None if v is None else -v
        return None
    return ev(node)


def _trig_vfloat(node):
    """Numeric value of the claimed trig answer (pi->pi, sqrt/abs supported);
    None if not a closed numeric expression. Used only to pick the sign branch."""
    import math as _m
    def ev(n):
        if isinstance(n, ast.Constant):
            return float(n.value) if isinstance(n.value, (int, float)) else None
        if isinstance(n, ast.Name):
            return _m.pi if n.id == "pi" else None
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            op = type(n.op)
            if op is ast.Mult:
                return a * b
            if op is ast.Div:
                return a / b if b else None
            if op is ast.Add:
                return a + b
            if op is ast.Sub:
                return a - b
            if op is ast.Pow:
                return a ** b
            return None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = ev(n.operand)
            return None if v is None else -v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            args = [ev(a) for a in n.args]
            if any(x is None for x in args):
                return None
            f = {"sqrt": _m.sqrt, "abs": abs}.get(n.func.id)
            return f(*args) if f else None
        return None
    try:
        return ev(node)
    except Exception:  # noqa: BLE001
        return None


def trig_quadrant_theorem(claim_src, prem_srcs, vt):
    """A self-contained Isabelle theorem for `sin(a)==V` / `cos(a)==V` given a
    `tan(a)==T` premise and quadrant bounds on `a`, via the precompiled trig
    meta-theorems; None if the pattern does not apply."""
    try:
        node = parse_expr(claim_src)
    except PyExprError:
        return None
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)):
        return None
    lhs, rhs = node.left, node.comparators[0]
    if not (isinstance(lhs, ast.Call) and isinstance(lhs.func, ast.Name)
            and lhs.func.id in ("sin", "cos") and len(lhs.args) == 1
            and isinstance(lhs.args[0], ast.Name)):
        return None
    fn, ang = lhs.func.id, lhs.args[0].id
    vf = _trig_vfloat(rhs)
    if vf is None:
        return None
    tan_src = None
    lo = hi = None
    assm_srcs = []
    for ps in prem_srcs:
        try:
            pn = parse_expr(ps)
        except PyExprError:
            continue
        used = False
        if (isinstance(pn, ast.Compare) and len(pn.ops) == 1
                and isinstance(pn.ops[0], ast.Eq)):
            l, r = pn.left, pn.comparators[0]
            if (isinstance(l, ast.Call) and isinstance(l.func, ast.Name)
                    and l.func.id == "tan" and len(l.args) == 1
                    and isinstance(l.args[0], ast.Name) and l.args[0].id == ang):
                tan_src = r
                assm_srcs.append(ps)
                used = True
        if (isinstance(pn, ast.Compare)
                and all(isinstance(o, (ast.Lt, ast.LtE)) for o in pn.ops)):
            terms = [pn.left] + list(pn.comparators)
            for i in range(len(pn.ops)):
                left, right = terms[i], terms[i + 1]
                if isinstance(right, ast.Name) and right.id == ang:
                    m = _pi_mult(left)
                    if m is not None:
                        lo = m if lo is None else max(lo, m)
                        used = True
                if isinstance(left, ast.Name) and left.id == ang:
                    m = _pi_mult(right)
                    if m is not None:
                        hi = m if hi is None else min(hi, m)
                        used = True
            if used and ps not in assm_srcs:
                assm_srcs.append(ps)
    if tan_src is None or lo is None or hi is None:
        return None
    quad = None
    for (a, b), info in _TRIG_QUAD:
        if a - 1e-6 <= lo and hi <= b + 1e-6:
            quad = info
            break
    if quad is None:
        return None
    csign, signlem = quad
    try:
        tt = transpile(tan_src, vt, "real")
        goal_t = transpile(node, vt, "real")
    except PyExprError:
        return None
    dd = f"(({tt})^2 + 1)"
    if fn == "sin":
        meta = "sin_from_tan_cneg" if csign == "neg" else "sin_from_tan_cpos"
        num = f"- ({tt})" if csign == "neg" else f"({tt})"
    else:
        meta = "cos_from_tan_cneg" if csign == "neg" else "cos_from_tan_cpos"
        num = "- 1" if csign == "neg" else "1"
    pang = _pv(ang)
    cs_op = "<" if csign == "neg" else ">"
    ids = set(identifiers(goal_t))
    assm_terms = []
    for i, ps in enumerate(assm_srcs):
        try:
            t = transpile(parse_expr(ps), vt, "real")
        except PyExprError:
            return None
        assm_terms.append((f"h{i}", t))
        ids |= identifiers(t)
    tan_ref = next((n for n, t in assm_terms if "tan" in t), None)
    if tan_ref is None:
        return None
    fixhead = (f"  fixes {' '.join(f'{i}::real' for i in sorted(ids))}\n"
               if ids else "")
    ass = ("  assumes "
           + "\n      and ".join(f'{n}: "{t}"' for n, t in assm_terms) + "\n")
    vt_rhs = transpile(rhs, vt, "real")
    if vf >= 0:
        match = (f'  have "{num} / sqrt {dd} = ({vt_rhs})" by {_TRIG_DISCH}\n'
                 f'  thus ?thesis using m by simp\n')
    else:
        match = (f'  have "- ({num}) / sqrt {dd} = - ({vt_rhs})" by {_TRIG_DISCH}\n'
                 f'  thus ?thesis using m by simp\n')
    return (f'theorem chk:\n{fixhead}{ass}  shows "{goal_t}"\nproof -\n'
            f'  have cs: "cos {pang} {cs_op} 0" using assms'
            f' by (auto intro: {signlem})\n'
            f'  have m: "{fn} {pang} = {num} / sqrt {dd}"'
            f' by (rule {meta}[OF cs {tan_ref}])\n'
            f'{match}qed')


# ---------- per-response pipeline ----------

_CASCADE_EX = None
_CASCADE_EX_LOCK = threading.Lock()


def _cascade_executor(pool_workers: int):
    """Process-shared cascade executor (2026-07-11 review #1). Sized well
    above the prover-worker count so the provers never starve; cascade tasks
    spend most of their life parked on pool futures, so slots are cheap.
    Sized once at first use (per reward-worker process)."""
    global _CASCADE_EX
    with _CASCADE_EX_LOCK:
        if _CASCADE_EX is None:
            from concurrent.futures import ThreadPoolExecutor
            _CASCADE_EX = ThreadPoolExecutor(
                max_workers=max(32, 8 * int(pool_workers or 4)),
                thread_name_prefix="isa-cascade")
    return _CASCADE_EX


def process_one(rid, item, pool, config, outdir=None, corrupt=False,
                max_steps: int = 0):
    problem = item["problem"]
    steps_xml = parse_xml_steps(item["response"])
    # Verify only the first max_steps steps (0 = no limit). The step reward
    # manager penalizes steps beyond penalty_max_steps and DISCARDS their
    # verdicts, so translating/verifying them is pure waste -- and
    # step-inflated responses (15-20 steps) are exactly the reward-time
    # stragglers. Truncated steps are never translated, never enter any
    # judge prompt, and are never referenced.
    if max_steps > 0 and steps_xml is not None and len(steps_xml) > max_steps:
        steps_xml = steps_xml[:max_steps]
    box = boxed_answer(item["response"])
    rec = {"rid": rid, "dataset": item["dataset"], "idx": item["idx"],
           "sample": item["sample"], "format_ok": steps_xml is not None,
           "boxed": box, "outcome_correct":
           box is not None and box == item["ground_truth"].strip(),
           "givens_ok": False, "steps_ok": False, "n_steps": 0, "steps": [],
           "premise_inconsistent_at": None, "corrupt_info": None}
    # Per-response wall profile (2026-07-11 review #6). judge_http_s measures
    # only requests.post wall time and therefore does not overlap prover queue
    # or run time. translate_validate_s is retained as the end-to-end cached
    # translate+parse+validate wall and may include prover-backed validation.
    prof = {"judge_http_s": 0.0, "translate_validate_s": 0.0,
            "prove_calls": 0, "prove_queue_s": 0.0,
            "prove_run_s": 0.0, "prove_cache_hits": 0}
    rec["prof"] = prof
    _prof_lock = threading.Lock()
    # Unified scheduling (2026-07-11 review #1): route every prover call
    # through pool.submit's process-wide FIFO when available (real pools).
    # Fallback to direct check for pools without it (mocks, older callers).
    _submit = getattr(pool, "submit", None)

    def _record(r):
        with _prof_lock:   # probes/cascades run concurrently
            prof["prove_calls"] += 1
            prof["prove_queue_s"] += float(r.get("queue_wait") or 0.0)
            prof["prove_run_s"] += float(r.get("check_time") or 0.0)
            if r.get("cache_hit"):
                prof["prove_cache_hits"] += 1

    def _check(thm):
        r = _submit(thm).result() if _submit is not None else pool.check(thm)
        _record(r)
        return r
    if steps_xml is not None and corrupt:
        rec["corrupt_info"] = corrupt_steps(steps_xml, item["ground_truth"])
        if rec["corrupt_info"] is None:
            rec["format_ok"] = False
            return rec
    if steps_xml is None:
        # No per-response print: surfaced via format_ok=0 in metrics and the
        # trainer's [Step Rewards] sample line.
        return rec
    rec["n_steps"] = len(steps_xml)

    base_nums = num_values(problem, words=True)
    # Subscript indices in a_n notation (a_1, a_4, a_{12}) are problem numbers:
    # num_values skips them (the leading `_` is a word char), yet an arithmetic/
    # geometric sequence's defining relation a_k == a_1 + (k-1)*d has coefficients
    # k-1 drawn straight from those indices, so they must be admitted.
    base_nums |= {Fraction(m) for m in re.findall(r"_\{?(\d+)", problem)}
    # `allowed_given_nums` (was givens_window): the numbers a GIVEN may use --
    # the faithfulness allow-list that catches the JUDGE inventing numbers not
    # in the problem. A range phrase ("integers 1 through 9") only yields its
    # endpoints via num_values, so range_ints adds the interior integers (4/6/8)
    # -- else they read as invented. Flows into given_nums (hence the step-level
    # allow-lists too), so range problems translate in givens AND steps.
    problem_nums = set(base_nums) | FREE_NUMS | range_ints(problem)
    allowed_given_nums = set(problem_nums)
    allowed_given_nums |= {Fraction(c) for c in
                           (3, 5, 7, 10, 12, 24, 25, 50, 60, 90, 100, 180, 360,
                            1000)}
    allowed_given_nums |= {Fraction(10) ** k for k in range(4, 10)}
    for v in base_nums:
        for c in (Fraction(60), Fraction(100), Fraction(1000)):
            allowed_given_nums.add(v * c)
            if v != 0:
                allowed_given_nums.add(v / c)

    def validate_givens(parsed):
        fixes, givens = parsed
        vt = dict(fixes)
        errs = []
        terms = []
        for g in givens:
            try:
                term, ids, consts, carrier = expr_info(g, vt)
            except PyExprError as e:
                errs.append(f"GIVEN '{g[:60]}': {e}")
                continue
            bad = [str(v) for v in consts if v not in allowed_given_nums]
            if bad:
                errs.append(f"GIVEN uses numbers not in the problem "
                            f"statement: {sorted(set(bad))}. Only "
                            "transcribe the problem.")
            if re.fullmatch(r"answer\s*==\s*-?[\d./() *]+", g):
                errs.append("Do not state the value of answer as a GIVEN.")
            terms.append((term, ids))
        if errs:
            return errs
        if not any("answer" in ids for _, ids in terms):
            return ["No GIVEN defines answer. Add exactly one GIVEN naming "
                    "the asked quantity (answer == <variable>), or the "
                    "question's own expression - never a derived formula, "
                    "never a numeric value."]
        # auto-declare undeclared identifiers (v4 parity: augment, not
        # reject) so a missing VARS entry never fails a translation
        undeclared = set().union(*(ids for _, ids in terms)) - set(vt)
        for u in sorted(undeclared):
            fixes.append((u, "real"))
            vt[u] = "real"
        if undeclared:
            terms.clear()
            for g in givens:
                term, ids, _, _ = expr_info(g, vt)
                terms.append((term, ids))
        thm = make_theorem(fixes, [(f"g{i}", t) for i, (t, _) in
                                 enumerate(terms)], "True", "simp")
        r = _check(thm)
        if not r["success"]:
            return r["errors"][:2] or ["givens skeleton rejected"]
        return []

    _t_tr = time.time()
    parsed_a, att_a, _ = translate(
        PROMPT_GIVENS_PY.replace("{problem}", problem), parse_givens_py,
        validate_givens,
        judge_url=config.judge_url, judge_model=config.judge_model,
        max_model_len=config.max_model_len,
        api_timeout=config.api_timeout,
        soft_prefix=("GIVEN uses numbers", "No GIVEN defines answer"))
    prof["translate_validate_s"] += time.time() - _t_tr
    prof["judge_http_s"] += sum(
        float(a.get("http_wall_s") or 0.0) for a in att_a
        if isinstance(a, dict))
    rec["translate_a"] = att_a
    if parsed_a is None:
        # No per-response print: surfaced via givens_ok=0 in metrics.
        return rec
    fixes, givens_src = parsed_a
    vt = dict(fixes)
    rec["givens_ok"] = True

    def validate_props(props, required_end=None):
        required_end = required_end or len(steps_xml)
        errs = []
        if not set(range(1, required_end + 1)) <= set(props.keys()):
            return [f"Expected one STEP line per step "
                    f"(1..{required_end}), got steps "
                    f"{sorted(props.keys())}."]
        plist = [props[k + 1]["prop"] for k in range(required_end)]
        # the root of a proposition must be boolean-shaped: comparison,
        # and/or, or unary not. Bare values (8, x, x + 1) are TERMS, not
        # propositions; the transpiler still emits something for them
        # (`(8::int)`) but Isabelle rejects it as an obligation. Reject
        # here so the judge gets a clear retry signal.
        for k, p in enumerate(plist):
            try:
                root = parse_expr(p)
            except PyExprError:
                continue  # the type/syntax error below will fire first
            if not isinstance(root, (ast.Compare, ast.BoolOp))\
                    and not (isinstance(root, ast.UnaryOp)
                             and isinstance(root.op, ast.Not)):
                errs.append(f"STEP {k + 1} '{p[:50]}': prop must be a "
                            "boolean expression (use ==, <, >, <=, >=, "
                            "and, or). A bare value like '8' or 'x' is "
                            "not a proposition.")
        if errs:
            return errs[:5]
        vt2 = dict(vt)
        for k in range(required_end):
            entry = props[k + 1]
            for src in [entry["prop"]] + entry["premises"]:
                try:
                    _, ids, _, _ = expr_info(src, vt2)
                    for i in ids:
                        vt2.setdefault(i, "real")
                except PyExprError as e:
                    errs.append(f"STEP {k + 1} '{src[:50]}': {e}")
        if errs:
            return errs[:5]
        for k, p in enumerate(plist):
            for c in conjuncts(p):
                if is_vacuous_node(c) and len(conjuncts(p)) > 1:
                    errs.append(f"STEP {k + 1}: drop the trivially-true "
                                "conjunct; every conjunct must carry "
                                "content from the conclusion.")
                    break
            if is_vacuous_node(parse_expr(p)):
                errs.append(f"STEP {k + 1}: proposition is trivially true; "
                            "state the conclusion's actual content.")
        var_vals = {}
        for g in givens_src:
            pin_from(g, var_vals)
        t_errs = []
        for k, s in enumerate(steps_xml[:required_end]):
            miss = transcription_missing_py(
                plist[k], s["conclusion"], var_vals,
                " ".join(s["premises"]), list(givens_src) + plist[:k])
            if miss:
                t_errs.append(f"STEP TRANSCRIPTION {k + 1}: the conclusion "
                              f"mentions numbers {miss} absent from your "
                              "expression; include them doing real work "
                              "(as the conclusion computes them), even if "
                              "the conclusion looks wrong - transcribe, "
                              "never fix.")
            pin_from(plist[k], var_vals)
        errs += t_errs[:5]
        return errs[:6]

    # long solutions translate in chunks of CHUNK fresh steps per judge
    # call; earlier chunks ride along as context lines the judge must not
    # re-output (the merge keeps them). Within a chunk, a retry may also
    # re-output only the rejected STEP lines (partial-retry merge).
    # 20, not 12: chunking rescues truncation on very long solutions but
    # costs boundary quality (naming drift) on mid-length ones (v5.5 read:
    # OlympiadBench +5, judge-AIME -6)
    CHUNK = 20
    vars_givens_text = ("VARS: " + ", ".join(f"{n} {t}" for n, t in fixes)
                        + "\n" + "\n".join(f"GIVEN: {g}"
                                           for g in givens_src))
    parsed_all = {}
    att_b_list = []
    failed_b = False
    for c0 in range(0, len(steps_xml), CHUNK):
        block_end = min(c0 + CHUNK, len(steps_xml))
        steps_text = "\n\n".join(
            f"STEP {k + 1}:\n"
            + "\n".join(f"premise: {p}" for p in steps_xml[k]["premises"])
            + f"\nconclusion: {steps_xml[k]['conclusion']}"
            for k in range(c0, block_end))
        if c0:
            steps_text = (
                "Steps already translated (context only, do NOT re-output "
                "them):\n"
                + "\n".join(f"STEP {j + 1} | prop: "
                            f"{parsed_all[j + 1]['prop']}"
                            for j in range(c0))
                + "\n\nNEW STEPS:\n\n" + steps_text)
        merged_steps = dict(parsed_all)

        def parse_props_merge(reply, _m=merged_steps):
            got = parse_props_py(reply)
            if got:
                _m.update(got)
            return dict(_m) if _m else None

        def validate_block(props, _end=block_end):
            return validate_props(props, _end)

        _t_tr = time.time()
        parsed_b, att_b, _ = translate(
            PROMPT_STEPS_PY
            .replace("{vars_givens}", vars_givens_text)
            .replace("{steps}", steps_text),
            parse_props_merge, validate_block,
            judge_url=config.judge_url, judge_model=config.judge_model,
            max_model_len=config.max_model_len,
            api_timeout=config.api_timeout,
            soft_prefix="STEP TRANSCRIPTION")
        prof["translate_validate_s"] += time.time() - _t_tr
        prof["judge_http_s"] += sum(
            float(a.get("http_wall_s") or 0.0) for a in att_b
            if isinstance(a, dict))
        att_b_list.append(att_b)
        if parsed_b is None:
            failed_b = True
            break
        parsed_all = {k: v for k, v in parsed_b.items() if k <= block_end}
    rec["translate_b"] = (att_b_list[0] if len(att_b_list) == 1
                          else att_b_list)
    if failed_b:
        # No per-response print: surfaced via steps_ok=0 in metrics.
        return rec
    rec["steps_ok"] = True
    props_src = [parsed_all[k + 1]["prop"] for k in range(len(steps_xml))]
    prem_srcs = [parsed_all[k + 1]["premises"]
                 for k in range(len(steps_xml))]

    vt_all = dict(vt)
    # A step may introduce a FRESH intermediate name not declared in the givens
    # VARS (current_tiles, subtotal). Its default type must not clash with the
    # declared ones: in a purely INTEGER problem (counting -- tiles, apples),
    # defaulting a new var to real makes `red::int == current_tiles::real -
    # blue::int` a TYPE ERROR (Isabelle has no int<->real coercion) and the step
    # fails to typecheck (2026-07-15: pool tiles `ox`). Default new names to int
    # when EVERY declared var is int, else real (real safely holds a stray
    # integer literal, and a genuinely fractional intermediate is normally
    # declared real by the judge so it is not defaulted here).
    _new_default = "int" if (vt and all(t == "int" for t in vt.values())) \
        else "real"
    for p in props_src + [q for ps in prem_srcs for q in ps]:
        try:
            _, ids, _, _ = py_to_isabelle(p, vt_all)
            for i in ids:
                vt_all.setdefault(i, _new_default)
        except PyExprError:
            pass
    # Uninterpreted functions (f/g/h applied) are declared as function-typed
    # fixes `f :: "real => ... => real"`, NOT scalars -- collected from every
    # given/prop/premise. _canonical_fixes() then picks them up by name like
    # any other fix. (analyze() already keeps these names out of vt_all.)
    func_ar = {}
    for p in givens_src + props_src + [q for ps in prem_srcs for q in ps]:
        try:
            func_ar.update(func_arities(parse_expr(p)))
        except PyExprError:
            pass
    # Fix names carry the same pv_ prefix as the transpiled terms (see
    # pyexpr._pv) so free variables never collide with an Isabelle constant.
    func_fixes = [(_pv(name), '"' + " => ".join(["real"] * (ar + 1)) + '"')
                  for name, ar in sorted(func_ar.items())]
    fixes_all = [(_pv(k), v) for k, v in vt_all.items()] + func_fixes
    terms, consts_per = [], []
    for p in props_src:
        term, ids, consts, carrier = _transpile_conjunctive(p, vt_all)
        terms.append((term, carrier))
        consts_per.append(consts)
    given_terms = [py_to_isabelle(g, vt_all)[0] for g in givens_src]
    given_nums = set(problem_nums)
    known = set()
    for g in givens_src:
        try:
            _, ids, cs, _ = py_to_isabelle(g, vt_all)
            given_nums |= cs
            known |= ids
        except PyExprError:
            pass
    g_prem = [(f"g{i}", t) for i, t in enumerate(given_terms)]
    # A claim that exactly restates a GIVEN is trivially satisfied: the given is
    # an always-true assumption, so the claim holds independent of the step's
    # premise-chain consistency (safe mode too, where g_prem is withheld from
    # the proof). Without this, a step that merely carries a given forward --
    # "the LCM of 1 through 9 is the answer", when the givens define
    # answer == lcm(1,...,9) -- is an unprovable free-variable claim, mislabeled
    # x and denied reward for a valid restatement.
    given_dumps = set()
    for _g in givens_src:
        try:
            given_dumps.add(ast.dump(parse_expr(_g)))
        except PyExprError:
            pass

    if outdir is not None:
        (outdir / f"{rid:03d}_v5.json").write_text(json.dumps(
            {"fixes": fixes_all, "givens_src": givens_src,
             "props_src": props_src, "prem_srcs": prem_srcs,
             "givens": given_terms, "props": [t for t, _ in terms],
             "steps_xml": steps_xml}, indent=2))

    # ---- Step verification in 3 phases (2026-07-10 straggler fix). The old
    # single loop ran every prover check strictly serially per response, so a
    # 12-step straggler held the whole training step (~365s of the tail was
    # serial pool.check chains while other pool workers idled). Split:
    #   phase 1 (serial, pure CPU): per-step prep. Every cross-step input --
    #     premise lists, known ids, concl_nums, var_vals -- derives from the
    #     TRANSLATIONS only, never from verdicts, so it can be precomputed in
    #     step order without touching the prover.
    #   phase 2 (parallel): each step's prover work (consistency probe, nz
    #     checks, claim cascade, tolerance fallback) is an independent task.
    #   phase 3 (serial): premise-inconsistency latch = MIN step whose probe
    #     proved False (identical to the sequential first-hit latch, because
    #     each probe's verdict depends only on that step's premise list), then
    #     rewards. Semantics are byte-identical to the sequential loop
    #     (regression-verified); the only difference is that probes past the
    #     latch also run (their results are discarded, memo absorbs the cost).
    var_vals = {}
    for g in givens_src:
        pin_from(g, var_vals)
    concl_nums = set()
    # sos rescue is gated on nonlinearity; a nonlinear GIVEN (e.g. a geometric
    # relation a_3 = a_1*r**2) is admitted to every step, so compute it once
    # here and OR it into each step's flag below.
    givens_nonlin = False
    for _g in givens_src:
        try:
            if is_nonlinear(parse_expr(_g)):
                givens_nonlin = True
                break
        except PyExprError:
            pass
    # Linear-probe gate (2026-07-14): the consistency probe may use the fast,
    # complete `(linarith|presburger)` (LINEAR_FALSE) instead of the heavy
    # ALTERNATION only when EVERY accumulated premise is affine with constant
    # coefficients. Precompute per-source linearity here; the givens are
    # admitted to every step, and each step also carries all EARLIER steps'
    # conclusions, so a step's premise chain is linear iff the givens, all
    # prior props, and this step's own admitted premises/definitions are each
    # linear (a parse failure -> treated as non-linear, i.e. keep ALTERNATION).
    def _src_linear(src):
        try:
            return is_linear_arith(parse_expr(src))
        except PyExprError:
            return False
    givens_linear = all(_src_linear(g) for g in givens_src)
    props_linear = [_src_linear(p) for p in props_src]
    prepped = []
    for k, (s, (term, carrier)) in enumerate(zip(steps_xml, terms)):
        entry = {"step": k, "neutral": False}
        entry["transcription_missing"] = transcription_missing_py(
            props_src[k], s["conclusion"], var_vals,
            " ".join(s["premises"]), list(givens_src) + props_src[:k])
        pin_from(props_src[k], var_vals)
        # `allowed_step_nums` (was win): the numbers this step's translated
        # proposition may use -- the problem's given_nums PLUS the numbers the
        # MODEL itself wrote in this step's text (so a number the model states,
        # e.g. cos(pi/4), is the model's, not a judge invention). Anything else
        # is the JUDGE inventing a number -> guard_invented.
        step_text_nums = num_values(s["block_text"], words=True)
        allowed_step_nums = step_text_nums | given_nums
        for v in set(step_text_nums):
            allowed_step_nums.add(v * 100)
            if v != 0:
                allowed_step_nums.add(v / 100)
        entry["guard_invented"] = [str(v) for v in consts_per[k]
                                   if v not in allowed_step_nums]
        entry["guard_ok"] = not entry["guard_invented"]

        # (c) definition vs claim split over the prop's top-level conjuncts:
        # a definition (fresh name == expr over known identifiers, never
        # `answer`, never a bare constant) is a conservative extension - it
        # becomes an assumption instead of a proof obligation
        prop_conj = conjuncts(props_src[k])
        defs_t, claims_t, claims_nodes, claims_given = [], [], [], []
        defs_linear = True   # all definition-conjuncts affine (linear probe gate)


        for c in prop_conj:
            is_def = (isinstance(c, ast.Compare) and len(c.ops) == 1
                      and isinstance(c.ops[0], ast.Eq)
                      and isinstance(c.left, ast.Name)
                      and c.left.id != "answer"
                      and c.left.id not in known)
            if is_def:
                try:
                    from verl.utils.isabelle_utils.pyexpr import analyze as _an
                    rids, _, _ = _an(c.comparators[0])
                    is_def = bool(rids & known)
                except PyExprError:
                    is_def = False
            ccar = _node_carrier(c, vt_all, "real")
            if is_def:
                defs_t.append(transpile(c, vt_all, ccar))
                if not is_linear_arith(c):
                    defs_linear = False
            else:
                claims_t.append(transpile(c, vt_all, ccar))
                claims_nodes.append(c)
                claims_given.append(ast.dump(c) in given_dumps)
        entry["n_definitions"] = len(defs_t)

        # (b) the step's own premises, admitted only with provenance: their
        # numbers must trace to the problem/givens, earlier conclusions, OR the
        # MODEL's own text for this step -- a number the model wrote (cos(pi/4)
        # = sqrt2/2, "an odd integer between 3 and 7") is the model's citation,
        # not a judge invention, so admit it; a false cited premise is caught by
        # the consistency probe, not this number check. Copying this step's own
        # conclusion is still rejected (below).
        admitted = []
        prem_nonlin = False
        admitted_linear = True   # all admitted premises affine (linear probe gate)
        prop_dumps = {ast.dump(c) for c in prop_conj}
        allowed_prem_nums = (given_nums | concl_nums | FREE_NUMS
                             | step_text_nums)
        for psrc in prem_srcs[k]:
            try:
                from verl.utils.isabelle_utils.pyexpr import analyze as _an
                node = parse_expr(psrc)
                pids, pcs, _ = _an(node)
            except PyExprError:
                continue
            if any(ast.dump(c) in prop_dumps for c in conjuncts(psrc)):
                continue
            if not all(v in allowed_prem_nums for v in pcs):
                continue
            try:
                admitted.append(py_to_isabelle(psrc, vt_all)[0])
            except PyExprError:
                continue
            if is_nonlinear(node):
                prem_nonlin = True
            if not is_linear_arith(node):
                admitted_linear = False
        entry["n_admitted_premises"] = len(admitted)

        prem = g_prem + [(f"s{j}", terms[j][0]) for j in range(k)]
        prem += [(f"p{k}_{i}", t) for i, t in enumerate(admitted)]
        prem += [(f"d{k}_{i}", t) for i, t in enumerate(defs_t)]
        nz_list = []
        if claims_t:
            for claim_node in claims_nodes:
                claim_carrier = _node_carrier(claim_node, vt_all, "real")
                try:
                    claim_src = ast.unparse(claim_node)
                except AttributeError:
                    claim_src = props_src[k]
                nz_list.extend(nz_divisors(
                    claim_src, vt_all, claim_carrier))
        # Rational equations put the UNKNOWN in a premise denominator (work-rate:
        # "1/comb = 1/pipe - 1/leak", solve pipe), not the claim. Collect premise
        # divisors too so pipe != 0 gets proved and admitted for FIELD_RESCUE.
        for psrc in prem_srcs[k]:
            try:
                pcar = _node_carrier(parse_expr(psrc), vt_all, "real")
                nz_list.extend(nz_divisors(psrc, vt_all, pcar))
            except PyExprError:
                pass
        nz_list = list(dict.fromkeys(nz_list))
        step_nonlin = (givens_nonlin or prem_nonlin
                       or any(is_nonlinear(c) for c in claims_nodes))
        # Linear-probe gate: this step's premise chain (givens + ALL earlier
        # conclusions + this step's admitted premises + definitions) is affine,
        # so the consistency probe can use the fast complete `(linarith|
        # presburger)` instead of grinding ALTERNATION to the watchdog.
        linear_prem = (givens_linear and all(props_linear[:k])
                       and admitted_linear and defs_linear)
        # linear_step also requires every CLAIM to be affine: then
        # (linarith|presburger) is a COMPLETE decision procedure for the
        # with-premise proof too (proves it fast or the goal is genuinely not
        # entailed), so the claim cascade skips the ALTERNATION grind. Measured:
        # a 10-relation sequence goal a_3+a_6+a_9=32 took ALTERNATION 17.6s ->
        # watchdog -> x, but linarith 0.96s -> proved.
        linear_step = (linear_prem and bool(claims_nodes)
                       and all(is_linear_arith(c) for c in claims_nodes))
        # Quadrant-trig: a single-claim `sin(a)/cos(a) == V` step with a
        # `tan(a) == T` premise and quadrant bounds gets a self-contained
        # theorem built from the precompiled meta-theorems (bounds live in the
        # givens, so feed those in too). None unless the pattern matches.
        trig_thm = None
        if len(claims_nodes) == 1:
            try:
                _csrc = ast.unparse(claims_nodes[0])
            except AttributeError:
                _csrc = None
            if _csrc and ("sin" in _csrc or "cos" in _csrc):
                trig_thm = trig_quadrant_theorem(
                    _csrc, list(prem_srcs[k]) + list(givens_src), vt_all)
        rec["steps"].append(entry)
        prepped.append({"k": k, "entry": entry, "prem": prem,
                        "claims_t": claims_t, "claims_nodes": claims_nodes,
                        "claims_given": claims_given, "trig_thm": trig_thm,
                        "nz_list": nz_list, "carrier": carrier,
                        "nonlinear": step_nonlin, "linear_prem": linear_prem,
                        "linear_step": linear_step})
        # cross-step state advances on TRANSLATIONS, exactly as before
        try:
            known |= py_to_isabelle(props_src[k], vt_all)[1]
        except PyExprError:
            pass
        concl_nums |= consts_per[k]

    def _canonical_fixes(goal):
        """Minimal, sorted fixes for a premise-free goal: only the variables
        that actually occur in the goal text. Unused universally-quantified
        fixes are vacuous for provability, but they make otherwise-identical
        bare theorems (the same arithmetic claim across rollouts) textually
        different, defeating the theorem cache. Sorting canonicalizes the
        text across responses, so identical bare claims share ONE cache
        entry (memory + disk) instead of 16."""
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", goal))
        return sorted((n, t) for (n, t) in fixes_all if n in toks)

    def _probe_step_raw(p, tac=ALTERNATION):
        """Premise-consistency probe (goal False from the accumulated axioms).
        Returns the RAW check result so the caller can classify it ternary
        (inconsistent / consistent / undetermined). Used only by the mock /
        legacy non-submit path; the submit path builds the theorem inline. `tac`
        is LINEAR_FALSE for an affine premise chain, else ALTERNATION."""
        return _check(make_theorem(fixes_all, p["prem"], "False", tac))

    def _cascade_step(p):
        prem, claims_t = p["prem"], p["claims_t"]
        # Giant-number guard (2026-07-11): a claim or premise carrying a huge
        # literal / >=1000 exponent / factorial / power tower makes the
        # leading simp/presburger of ALTERNATION grind 60-75s past the 15s
        # watchdog. Route the whole step to eval alone (watchdog-respecting,
        # still proves legit moderate computations); skip the nz and tolerance
        # refinements, which would re-introduce the grinding tactics.
        danger = is_dangerous_isabelle(*claims_t, *[t for _, t in prem])
        out = {"verified": None, "tolerance": False, "danger": danger}
        # Safe mode (2026-07-11): the consistency of this step's premise chain
        # is UNKNOWN (an earlier probe was undetermined), so ANY with-premises
        # proof could be ex-falso. Prove the claim INDEPENDENTLY only -- a
        # premise-free (canonical) proof is a tautology, true regardless of
        # premise consistency, so it can still earn reward; a claim that needs
        # the suspect premises earns nothing. This MUST gate the nz loop and
        # the tolerance fallback too, not just the main _verify_single proof:
        # both build make_theorem(..., prem, ...) and ALTERNATION's
        # linarith/presburger/auto close any goal ex-falso from inconsistent
        # premises (2026-07-11 review: HIGH ex-falso reward leak).
        safe = bool(p.get("safe_mode"))
        # sos rescue is gated on nonlinearity, computed at prep time over the
        # claims AND the admitted premises AND the givens -- the nonlinear
        # fact usually lives in a premise (e.g. a_3 = a_1*r**2), not the
        # (linear) conclusion a_2 = 6. sos grinds ~15s on goals it cannot
        # close and proves any goal ex-falso from inconsistent premises, so
        # it is also gated on non-danger (here) and non-safe (below).
        nonlin = (not danger) and bool(p.get("nonlinear"))
        nz_prem = []
        if not danger and not safe:
            for d in p["nz_list"]:
                # ALTERNATION proves most d != 0; a denominator that is itself
                # only pinned through a rational premise (pipe, from
                # 1/comb = 1/pipe - 1/leak) needs field_simps, so fall back to
                # FIELD_RESCUE.
                for nztac in (ALTERNATION, FIELD_RESCUE):
                    if _check(make_theorem(fixes_all, prem,
                                           f"{d} \\<noteq> 0", nztac))["success"]:
                        nz_prem.append((f"nz{len(nz_prem)}",
                                        f"{d} \\<noteq> 0"))
                        break
        full_prem = prem + nz_prem
        tac = SAFE_DANGEROUS if danger else ALTERNATION

        def _verify_single(g):
            # Quadrant-trig recipe: a self-contained theorem (sin/cos from
            # tan + quadrant, via the precompiled meta-theorems). Sound in safe
            # mode too -- it derives the true value from the stated premises via
            # real theorems, never ex-falso.
            tthm = p.get("trig_thm")
            if tthm is not None and _check(tthm)["success"]:
                return True
            if not safe:
                if p.get("linear_step"):
                    # Affine premises + affine goal: prove with (argo|smt)
                    # (LINEAR_CLAIM). This REPLACES the with-premise ALTERNATION,
                    # whose leading fastforce/powr branches grind a many-relation
                    # linear goal to the 15s watchdog. On the ACTUAL engine
                    # premise set (givens attached to every step -> ~19 premises)
                    # linarith/force/auto all hit the watchdog (Fourier-Motzkin /
                    # search blow-up); argo (built-in, simplex) closes both the
                    # difference step and the 11-variable value step in ~0.4s and
                    # fails fast on an unentailed goal (2026-07-14). The premise-
                    # free canonical / eval rescues below still run (a ground
                    # tautology can close without the premises).
                    if _check(make_theorem(fixes_all, full_prem, g,
                                           LINEAR_CLAIM))["success"]:
                        return True
                else:
                    r = _check(make_theorem(fixes_all, full_prem, g, tac))
                    if r["success"]:
                        return True
            else:
                # SAFE MODE (2026-07-15 rework). Consistency is UNKNOWN because
                # the `|- False` probe was UNDETERMINED -- and it is undetermined
                # in exactly the cases where the SAME tactic (ALTERNATION, sos/
                # smt are gated OFF in safe mode) could not derive `False` within
                # the watchdog. Measured invariant: an inconsistency ALTERNATION
                # CAN exploit is found FAST for the `False` goal too -> the probe
                # SUCCEEDS -> the step is hard-cut `c`, never safe mode (verified:
                # a1=9,k=40,a1=k/b1^2 -> probe proves False in 0.4s -> hard cut).
                # So a with-premise proof reaching safe mode cannot route through
                # ex-falso: ex-falso needs `False`, which this tactic provably
                # cannot reach here (else no safe mode). A completing success is
                # therefore a DIRECT derivation (any contradiction found within
                # the watchdog would have made the probe succeed) -> sound to
                # reward. Recovers inverse/direct variation (a = k/b^2: a2 = 4)
                # that the old premise-free-only safe mode zeroed to `u`, without
                # the perturbed/negation probe (which itself grinds 15s on the
                # nonlinear premises whatever the goal).
                if _check(make_theorem(fixes_all, full_prem, g, tac))["success"]:
                    return True
            r0 = _check(make_theorem(_canonical_fixes(g), [], g, tac))
            if r0["success"]:
                return True
            if danger:
                # eval already tried; do NOT fall through to the grinding
                # ALTERNATION/EVAL_RESCUE rescue on a giant goal.
                return False
            # EVAL_RESCUE already imported at module level
            r_eval = _check(make_theorem(_canonical_fixes(g), [], g,
                                         EVAL_RESCUE))
            if r_eval["success"]:
                return True
            # Rational-equation rescue: an unknown in a denominator (work-rate /
            # mixture: 1/comb = 1/pipe - 1/leak => pipe = 12). The denominators
            # were proved nonzero into nz_prem above; field_simps then solves.
            # ALTERNATION cannot (its field_simps sits behind `eval`, which
            # aborts on the free-variable goal). Uses the premises, so non-safe.
            if not safe and has_division(g, *[t for _, t in full_prem]):
                if _check(make_theorem(fixes_all, full_prem, g,
                                       FIELD_RESCUE))["success"]:
                    return True
            # Trig-identity rescue: tan(pi - a) = -tan(a), sin(pi/2 - a) = cos a,
            # angle-sum/-difference -- expand tan to sin/cos and apply the laws.
            if not safe and has_trig(g):
                if _check(make_theorem(fixes_all, full_prem, g,
                                       TRIG_RESCUE))["success"]:
                    return True
            # Polynomial-identity rescue: a factoring / expansion identity
            # (58*x^5 - 203*x^11 = 29*x^5*(2 - 7*x^6)) that algebra_simps
            # expands but ALTERNATION cannot reach (its fastforce/powr branches
            # grind on the polynomial first).
            if not safe and has_poly(g):
                if _check(make_theorem(fixes_all, full_prem, g,
                                       ALGEBRA_RESCUE))["success"]:
                    return True
            # Definitional-eval rescue: unfold the definitional givens
            # (answer = gcd 32 48) INTO the goal, then eval the now-ground
            # result (answer = 16). ALTERNATION's eval sits on the
            # un-substituted goal, where the free `answer` blocks evaluation.
            if not safe:
                uthm = make_theorem_unfold(fixes_all, full_prem, g)
                if uthm is not None and _check(uthm)["success"]:
                    return True
            # Integer rescue: presburger/arith decide a linear integer goal
            # (side3 = 5 from side3 mod 2 = 1 and 3 < side3 < 7). ALTERNATION
            # contains presburger but its earlier `eval` aborts on the
            # free-variable goal before reaching it; run it standalone with the
            # premises. Gated on integer flavour (mod/dvd or an int/nat carrier).
            if not safe:
                _gt = " ".join([g] + [t for _, t in full_prem])
                if ("mod" in _gt or "dvd" in _gt
                        or p.get("carrier") in ("int", "nat")):
                    if _check(make_theorem(fixes_all, full_prem, g,
                                           "(presburger | arith)"))["success"]:
                        return True
            # Nonlinear real rescue: sos WITH the premises (the nonlinear fact
            # -- r^2 = 9, a squared distance -- lives in them). Forbidden in
            # safe mode: sos would close the goal ex-falso from premises whose
            # consistency is unknown.
            if nonlin and not safe:
                if _check(make_theorem(fixes_all, full_prem, g,
                                       SOS_RESCUE))["success"]:
                    return True
            # Uninterpreted-function rescue: smt (verit) discharges functional
            # equations (forall x. f(x+1)=f(x)+2 => f 2=9) and functional
            # existentials that nothing else closes; make_theorem_with_logs
            # first supplies any log-numeral facts (so f(log b x)=x => f N=V
            # composes). Gated to a function in the step (else smt only wastes
            # ~15s grinding); non-safe (ex-falso) and non-danger (above).
            if func_ar and not safe:
                if _check(make_theorem_with_logs(fixes_all, full_prem, g,
                                                 SMT_RESCUE))["success"]:
                    return True
            # Log-numeral rescue: a bare `log B V = N` (V = B^N) that no single
            # tactic closes (numeral<->powr gap). The log toolbox proves it via
            # the powr recipe and hands it to the normal tactic. General over
            # every log computation, composed here with the premises.
            if not safe and has_log(g, *[t for _, t in full_prem]):
                if _check(make_theorem_with_logs(fixes_all, full_prem, g,
                                                 ALTERNATION))["success"]:
                    return True
            # Exponential-equation rescue: `b powr f = c => f-value` by the
            # injectivity of a base>1 power (4 powr (2y) = 4 => 2y = 1). smt with
            # powr_inj + powr_one closes it in 0.6s where auto/sos/plain-smt
            # fail/grind. Gated to a variable-exponent power in the step.
            if not safe and has_powr(g, *[t for _, t in full_prem]):
                if _check(make_theorem(fixes_all, full_prem, g,
                                       EXP_RESCUE))["success"]:
                    return True
            return False

        # A claim identical to a GIVEN is auto-satisfied (see given_dumps): the
        # given holds by assumption, so this is sound even in safe mode.
        gflags = p.get("claims_given") or [False] * len(claims_t)
        if len(claims_t) > 1:
            out["verified"] = all(gf or _verify_single(g)
                                  for gf, g in zip(gflags, claims_t))
        else:
            out["verified"] = gflags[0] or _verify_single(claims_t[0])
        if not out["verified"] and not danger and not safe:
            # NOT in safe mode: the tolerance fallback proves an approximate
            # goal FROM the premises, so it must be forbidden when premise
            # consistency is unknown (ex-falso leak, 2026-07-11 review).
            tol = tolerance_goal(p["claims_nodes"], vt_all, p["carrier"])
            if tol:
                rt = _check(make_theorem(fixes_all, prem + nz_prem, tol,
                                         "(approximation 20)"))
                if not rt["success"]:
                    rt = _check(make_theorem(fixes_all, prem + nz_prem,
                                             tol, ALTERNATION))
                if rt["success"]:
                    out["verified"] = True
                    out["tolerance"] = True
        return out

    par = max(1, int(os.environ.get("ISABELLE_STEP_CHECK_PAR", "4")))

    def _pmap(fn, items):
        if not items:
            return []
        if par == 1 or len(items) == 1:
            return [fn(p) for p in items]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(par, len(items))) as ex:
            return list(ex.map(fn, items))

    # Wave 1: consistency probes. TERNARY outcome per step (2026-07-11):
    #   inconsistent  -- proved `|- False` from the accumulated premises;
    #   consistent    -- the check COMPLETED and did not prove False;
    #   undetermined  -- we could NOT decide (dangerous premise chain skipped;
    #                    or timeout / worker_error / watchdog-incomplete).
    # A timeout must NEVER be read as "consistent" (that would let ex-falso
    # steps built on unverifiable premises earn reward). Dangerous premise
    # chains skip the prover entirely (proving False from a giant term would
    # grind). The first inconsistent OR undetermined step is the cutoff:
    # every step at/after it gets NO positive reward.
    def _classify_probe(r):
        if r.get("success"):
            return "inconsistent"
        if (r.get("worker_error") or r.get("incomplete")
                or r.get("undetermined")):
            return "undetermined"
        return "consistent"

    probe_oc = [None] * len(prepped)
    _pending = []
    for i, p in enumerate(prepped):
        # A purely linear premise chain is decided completely and fast by
        # (linarith|presburger); ALTERNATION grinds its powr/ln/exp branches to
        # the watchdog on a many-equation linear goal (2026-07-14 regression).
        probe_tac = LINEAR_FALSE if p.get("linear_prem") else ALTERNATION
        if is_dangerous_isabelle(*[t for _, t in p["prem"]]):
            probe_oc[i] = "undetermined"
        elif _submit is not None:
            _pending.append((i, _submit(make_theorem(
                fixes_all, p["prem"], "False", probe_tac))))
        else:
            probe_oc[i] = _classify_probe(_probe_step_raw(p, probe_tac))
    for i, fut in _pending:
        r = fut.result()
        _record(r)
        probe_oc[i] = _classify_probe(r)

    # Two boundaries (2026-07-11), both latched at their FIRST occurrence:
    #   hard_cut = first INCONSISTENT step. Premises are provably contradictory
    #     from here, so this and every later step earn NO reward (symbol 'c').
    #   safe_from = first UNDETERMINED step. Consistency is UNKNOWN from here,
    #     so this and every later step enter SAFE MODE: the with-premises proof
    #     is forbidden (could be ex-falso), but an INDEPENDENT premise-free
    #     proof still earns reward; a claim that needs the suspect premises (or
    #     times out) earns nothing (symbol 'u'). Inconsistent dominates unknown
    #     (a proven contradiction is a harder cut than an undecidable one).
    # Asymmetry rationale: an inconsistency is the MODEL's error (it asserted a
    # contradiction) -> penalize the whole tail; an undetermined check is the
    # VERIFIER's limitation -> stay lenient (rescue independently-true steps).
    # FUTURE (needs an Isabelle unsat core + a step-dependency graph, neither
    # available today): for an inconsistency, invalidate only the FIRST
    # conflict-introducing conclusion and block only steps that DEPEND on it,
    # letting conflict-independent later steps keep earning reward -- more
    # precise than the current conservative tail cut, but unsafe to attempt
    # without reliably identifying which conclusion to drop.
    hard_cut = None
    safe_from = None
    for p, oc in zip(prepped, probe_oc):
        if oc == "inconsistent" and hard_cut is None:
            hard_cut = p["k"]
        if oc == "undetermined" and safe_from is None:
            safe_from = p["k"]
    rec["premise_inconsistent_at"] = hard_cut
    if safe_from is not None:
        rec["premise_undetermined_at"] = safe_from

    def _mode(k):
        if hard_cut is not None and k >= hard_cut:
            return "hard"
        if safe_from is not None and k >= safe_from:
            return "safe"
        return "normal"

    # Wave 2: claim cascades for every step NOT hard-cut. Safe-mode steps DO
    # cascade (their independent proof can still earn reward); only the proven-
    # inconsistent tail is skipped as dead work.
    to_cascade = []
    for p in prepped:
        if not p["claims_t"]:
            continue
        m = _mode(p["k"])
        if m == "hard":
            continue
        p["safe_mode"] = (m == "safe")
        to_cascade.append(p)
    if _submit is not None and to_cascade:
        # Shared cascade executor (review #1): cascades are sequential
        # chains, so each needs a thread of control -- but a process-wide
        # bounded pool replaces one fresh 4-thread executor per response
        # (hundreds of short-lived threads per training step). Slots mostly
        # park on pool futures, so far fewer than the old peak suffice.
        _ex = _cascade_executor(getattr(pool, "num_workers", 4))
        cascade_results = list(_ex.map(_cascade_step, to_cascade))
    else:
        cascade_results = _pmap(_cascade_step, to_cascade)
    cascade_out = dict(zip((p["k"] for p in to_cascade), cascade_results))

    for p in prepped:
        entry = p["entry"]
        m = _mode(p["k"])
        entry["premises_inconsistent"] = (m == "hard")
        if m == "safe":
            entry["premises_undetermined"] = True
        if not p["claims_t"]:
            # definition-only step: nothing to prove (conservative), but
            # also nothing to reward - neutral, excluded from the metric
            entry["neutral"] = True
            entry["verified"] = True
            entry["rewarded"] = False
            continue
        if m == "hard":
            entry["verified"] = False   # hard cut: cascade skipped
            entry["rewarded"] = False
            continue
        res = cascade_out.get(p["k"])
        if res is None:
            entry["verified"] = False
        else:
            entry["verified"] = res["verified"]   # safe mode: independent-only
            if res["tolerance"]:
                entry["tolerance"] = True
        # A safe-mode step that VERIFIED did so with a premise-free proof, so
        # it earns reward; one that could not is 'u'. Reward is NOT gated on
        # premises_undetermined (safe mode already restricted the proof path).
        entry["rewarded"] = (entry["verified"] and entry["guard_ok"]
                             and not entry["transcription_missing"])

    # Per-step symbol string (priority o>c>u>m>g>x): o=rewarded,
    # c=premises-inconsistent (hard cut), u=premises-undetermined AND could not
    # be proven independently (safe-mode miss), m=verified-but-transcription-
    # missing, g=verified-but-guard-failed/neutral, x=unverified.
    pat = "".join(("o" if e["rewarded"] else
                   ("c" if e["premises_inconsistent"] else
                    ("u" if e.get("premises_undetermined") else
                     ("m" if e["verified"] and e["transcription_missing"] else
                      ("g" if e["verified"] else "x"))))) for e in rec["steps"])
    rec["pattern"] = pat
    # No per-response print: the pattern + outcome converge into the trainer's
    # single [Step Rewards] sample line (ray_trainer.py) once per step.
    return rec


def block_stats_v5(rs, label):
    """Like pipeline_v4.block_stats but definition-only steps are NEUTRAL:
    excluded from the verified/rewarded denominators (they carry no proof
    obligation and earn no reward)."""
    n = len(rs)
    if n == 0:
        print(f"{label}: no data")
        return
    fmt = [r for r in rs if r["format_ok"]]
    tr = [r for r in fmt if r["steps_ok"]]
    steps = [e for r in tr for e in r["steps"] if not e.get("neutral")]
    neutral = sum(1 for r in tr for e in r["steps"] if e.get("neutral"))
    nv = sum(1 for e in steps if e["verified"])
    nr = sum(1 for e in steps if e["rewarded"])
    t_rate = len(tr) / len(fmt) if fmt else 0
    v_rate = nv / len(steps) if steps else 0
    r_rate = nr / len(steps) if steps else 0
    print(f"{label}: n={n} format={len(fmt)}/{n} "
          f"translation={len(tr)}/{len(fmt)} ({100 * t_rate:.1f}%) "
          f"verified={nv}/{len(steps)} ({100 * v_rate:.1f}%) "
          f"rewarded={nr}/{len(steps)} ({100 * r_rate:.1f}%) "
          f"neutral={neutral}")
    print(f"  GOAL METRIC translation x verified = "
          f"{100 * t_rate * v_rate:.1f}%   (x rewarded = "
          f"{100 * t_rate * r_rate:.1f}%)")


class IsabelleEngine:
    """Isabelle/HOL verification engine for math step-level rewards.

    Wraps process_one: given a problem + policy response + ground truth,
    returns per-step verification results.
    """

    def __init__(self, config: IsabelleConfig | None = None):
        self.config = config or IsabelleConfig()
        # Honor fol_timeout-aligned check deadline.
        from verl.utils.isabelle_utils import server_pool as _sp
        _sp.CHECK_DEADLINE = float(self.config.check_deadline)
        # base_dir MUST be per-process: multiple RewardLoopWorker processes
        # each build their own engine, and IsabelleWorker.start() rmtree's
        # its master_dir — a shared path lets a later process wipe an earlier
        # process's live worker dirs (ENOENT on the next theory write).
        import os
        self.pool = IsabelleServerPool(
            num_workers=self.config.pool_workers,
            base_dir=f"/tmp/isabelle_pool_engine_{os.getpid()}",
            rss_cap_gb=self.config.rss_cap_gb)
        self.pool.start()
        import atexit
        atexit.register(self._safe_shutdown)

    def _safe_shutdown(self):
        try:
            self.pool.shutdown()
        except Exception:
            pass

    def verify_solution(self, problem: str, response: str,
                        ground_truth: str, dataset: str = "math",
                        idx: int = 0, sample: int = 0,
                        max_steps: int = 0) -> dict:
        """Verify a complete solution. Returns the same dict as process_one."""
        item = {
            "problem": problem,
            "response": response,
            "ground_truth": ground_truth,
            "dataset": dataset,
            "idx": idx,
            "sample": sample,
        }
        t0 = time.time()
        out = process_one(idx, item, self.pool, self.config,
                          max_steps=max_steps)
        # Total reward wall for this response (includes translate + prove +
        # queue waits + CPU prep); the gap vs the parts is scheduling overhead.
        if isinstance(out.get("prof"), dict):
            out["prof"]["reward_wall_s"] = time.time() - t0
        return out

    def shutdown(self):
        self.pool.shutdown()


# NOTE: no standalone CLI here. Training uses IsabelleEngine.verify_solution
# via the step reward manager; for offline batch measurement use the PoC
# pipeline (scripts/isabelle_poc_math500/pipeline_v5.py), which keeps the
# file-based --inputs/--outdir/--corrupt workflow. A previous main() lifted
# from the PoC was unrunnable in this module (undefined POC_DIR /
# ThreadPoolExecutor imports) and was removed on 2026-07-03.
