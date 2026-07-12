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
    ALTERNATION, EVAL_RESCUE, FALSE_TACTIC, NZ_TACTIC, SAFE_DANGEROUS,
    is_dangerous_isabelle, make_theorem, num_values, fixes_clause,
    identifiers, anchor_ground_numerals, FREE_NUMS,
)
from verl.utils.isabelle_utils.judge import translate
from verl.utils.isabelle_utils.xml_utils import (
    block_stats, boxed_answer, corrupt_steps, parse_xml_steps,
)
from verl.utils.isabelle_utils.pyexpr import (
    FUNCS, PyExprError, analyze, parse_expr, py_to_isabelle, transpile,
)

_PKG_DIR = Path(__file__).parent.resolve()
FREE_NUMS = {Fraction(0), Fraction(1), Fraction(2)}
UNIT_CLOSURE = (Fraction(60), Fraction(100), Fraction(1000))

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


def transcription_missing_py(src, con_text, var_vals, premise_text,
                             context_srcs):
    """AST version of the anti-repair check (v4.3 semantics)."""
    from verl.utils.isabelle_utils.pyexpr import _const_eval, analyze as _an
    try:
        node = parse_expr(src)
    except PyExprError:
        return []
    ctx_dumps = set()
    for c_src in context_srcs:
        try:
            ctx_dumps.add(ast.dump(parse_expr(c_src)))
        except PyExprError:
            pass
    pn, idents = set(), set()
    for c in (node.values if isinstance(node, ast.BoolOp) else [node]):
        if is_vacuous_node(c) or ast.dump(c) in ctx_dumps:
            continue
        try:
            ids, cs, _ = _an(c)
        except PyExprError:
            continue
        idents |= ids
        pn |= cs
    pn |= {var_vals[i] for i in idents if i in var_vals}
    prem_nums = num_values(premise_text, words=True)
    con_text = re.sub(r"_\{?\d+\}?", "", con_text)
    out = []
    for v in num_values(con_text, words=True):
        if v in FREE_NUMS or v in prem_nums:
            continue
        forms = {v}
        for c in UNIT_CLOSURE:
            forms.add(v * c)
            forms.add(v / c)
        if not (forms & pn):
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
    problem_nums = set(base_nums) | FREE_NUMS
    givens_window = set(problem_nums)
    givens_window |= {Fraction(c) for c in
                      (3, 5, 7, 10, 12, 24, 25, 50, 60, 90, 100, 180, 360,
                       1000)}
    givens_window |= {Fraction(10) ** k for k in range(4, 10)}
    for v in base_nums:
        for c in (Fraction(60), Fraction(100), Fraction(1000)):
            givens_window.add(v * c)
            if v != 0:
                givens_window.add(v / c)

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
            bad = [str(v) for v in consts if v not in givens_window]
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
    for p in props_src + [q for ps in prem_srcs for q in ps]:
        try:
            _, ids, _, _ = py_to_isabelle(p, vt_all)
            for i in ids:
                vt_all.setdefault(i, "real")
        except PyExprError:
            pass
    fixes_all = list(vt_all.items())
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
    prepped = []
    for k, (s, (term, carrier)) in enumerate(zip(steps_xml, terms)):
        entry = {"step": k, "neutral": False}
        entry["transcription_missing"] = transcription_missing_py(
            props_src[k], s["conclusion"], var_vals,
            " ".join(s["premises"]), list(givens_src) + props_src[:k])
        pin_from(props_src[k], var_vals)
        win = (num_values(s["block_text"], words=True) | given_nums)
        for v in set(num_values(s["block_text"], words=True)):
            win.add(v * 100)
            if v != 0:
                win.add(v / 100)
        entry["guard_invented"] = [str(v) for v in consts_per[k]
                                   if v not in win]
        entry["guard_ok"] = not entry["guard_invented"]

        # (c) definition vs claim split over the prop's top-level conjuncts:
        # a definition (fresh name == expr over known identifiers, never
        # `answer`, never a bare constant) is a conservative extension - it
        # becomes an assumption instead of a proof obligation
        prop_conj = conjuncts(props_src[k])
        defs_t, claims_t, claims_nodes = [], [], []


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
            else:
                claims_t.append(transpile(c, vt_all, ccar))
                claims_nodes.append(c)
        entry["n_definitions"] = len(defs_t)

        # (b) the step's own premises, admitted only with provenance: their
        # numbers must come from the problem/givens or earlier conclusions,
        # and copying this step's conclusion is rejected (Z3-path parity)
        admitted = []
        prop_dumps = {ast.dump(c) for c in prop_conj}
        prem_window = given_nums | concl_nums | FREE_NUMS
        for psrc in prem_srcs[k]:
            try:
                from verl.utils.isabelle_utils.pyexpr import analyze as _an
                node = parse_expr(psrc)
                pids, pcs, _ = _an(node)
            except PyExprError:
                continue
            if any(ast.dump(c) in prop_dumps for c in conjuncts(psrc)):
                continue
            if not all(v in prem_window for v in pcs):
                continue
            try:
                admitted.append(py_to_isabelle(psrc, vt_all)[0])
            except PyExprError:
                continue
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
        rec["steps"].append(entry)
        prepped.append({"k": k, "entry": entry, "prem": prem,
                        "claims_t": claims_t, "claims_nodes": claims_nodes,
                        "nz_list": nz_list, "carrier": carrier})
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

    def _probe_step_raw(p):
        """Premise-consistency probe (goal False from the accumulated axioms).
        Returns the RAW check result so the caller can classify it ternary
        (inconsistent / consistent / undetermined). Used only by the mock /
        legacy non-submit path; the submit path builds the theorem inline."""
        return _check(make_theorem(fixes_all, p["prem"], "False",
                                   ALTERNATION))

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
        nz_prem = []
        if not danger:
            for d in p["nz_list"]:
                rnz = _check(make_theorem(fixes_all, prem,
                                          f"{d} \\<noteq> 0", ALTERNATION))
                if rnz["success"]:
                    nz_prem.append((f"nz{len(nz_prem)}", f"{d} \\<noteq> 0"))
        full_prem = prem + nz_prem
        tac = SAFE_DANGEROUS if danger else ALTERNATION
        # Safe mode (2026-07-11): the consistency of this step's premise chain
        # is UNKNOWN (an earlier probe was undetermined), so a with-premises
        # proof could be ex-falso. Prove the claim INDEPENDENTLY only -- a
        # premise-free (canonical) proof is a tautology, true regardless of
        # premise consistency, so it can still earn reward; a claim that needs
        # the suspect premises earns nothing.
        safe = bool(p.get("safe_mode"))

        def _verify_single(g):
            if not safe:
                r = _check(make_theorem(fixes_all, full_prem, g, tac))
                if r["success"]:
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
            return r_eval["success"]

        if len(claims_t) > 1:
            out["verified"] = all(_verify_single(g) for g in claims_t)
        else:
            out["verified"] = _verify_single(claims_t[0])
        if not out["verified"] and not danger:
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
        if is_dangerous_isabelle(*[t for _, t in p["prem"]]):
            probe_oc[i] = "undetermined"
        elif _submit is not None:
            _pending.append((i, _submit(make_theorem(
                fixes_all, p["prem"], "False", ALTERNATION))))
        else:
            probe_oc[i] = _classify_probe(_probe_step_raw(p))
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
            base_dir=f"/tmp/isabelle_pool_engine_{os.getpid()}")
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
