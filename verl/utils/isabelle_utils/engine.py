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
    ALTERNATION, EVAL_RESCUE, FALSE_TACTIC, NZ_TACTIC,
    make_theorem, num_values, fixes_clause, identifiers,
    anchor_ground_numerals, FREE_NUMS,
)
from verl.utils.isabelle_utils.judge import translate
from verl.utils.isabelle_utils.xml_utils import (
    block_stats, boxed_answer, corrupt_steps, parse_xml_steps,
)
from verl.utils.isabelle_utils.pyexpr import (
    FUNCS, PyExprError, parse_expr, py_to_isabelle, transpile,
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
    """User-approved rounding semantics (2026-06-14): an equality against a
    written decimal is read as |lhs - c| < half-ulp of the written
    precision; when both sides are constants the COARSER precision wins
    (13.8 == 14 means tolerance 0.5). Returns the rewritten Isabelle goal,
    or None if no equality involves a written decimal."""
    parts, changed = [], False
    for c in claims_nodes:
        dec = None
        if (isinstance(c, ast.Compare) and len(c.ops) == 1
                and isinstance(c.ops[0], ast.Eq)):
            sides = [c.left, c.comparators[0]]
            places = []
            for s in sides:
                if isinstance(s, ast.Constant) and isinstance(s.value, float):
                    rep = repr(s.value)
                    if "." in rep and "e" not in rep and "E" not in rep:
                        places.append(len(rep.split(".")[1]))
                elif isinstance(s, ast.Constant) and isinstance(s.value, int):
                    places.append(0)
            float_sides = [s for s in sides
                           if isinstance(s, ast.Constant)
                           and isinstance(s.value, float)]
            if float_sides and places:
                cst = float_sides[0]
                other = sides[1] if cst is sides[0] else sides[0]
                dec = (other, Fraction(repr(cst.value)), min(places))
        if dec is None:
            parts.append(transpile(c, var_types, carrier))
            continue
        other, val, pl = dec
        lhs_t = transpile(other, var_types, "real")
        tol_den = 2 * 10 ** pl
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
    # Per-response wall profile (2026-07-11 review #6): where the reward time
    # actually goes -- translation wall vs prover queue wait vs prover run --
    # so the step-time tail is attributable from W&B instead of live
    # forensics. Mutated by _check/translate wrappers below; threaded through
    # rec["prof"] -> formal_verify debug -> isabelle/* metrics.
    prof = {"translate_s": 0.0, "prove_calls": 0, "prove_queue_s": 0.0,
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
    prof["translate_s"] += time.time() - _t_tr
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
        prof["translate_s"] += time.time() - _t_tr
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
        term, ids, consts, carrier = py_to_isabelle(p, vt_all)
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

        def _conj_carrier(cnode):
            # Per-conjunct carrier (2026-07-11): the prop-level carrier is
            # the union of the whole proposition's needs, so a mixed prop
            # like `x / 2 == 1 and n % 2 == 0` annotated EVERY numeral real
            # and mistyped the integer half. Each top-level conjunct now
            # picks the carrier its OWN content requires. (Different-typed
            # variables inside ONE conjunct still need a typed IR -- known
            # remaining limitation.)
            try:
                from verl.utils.isabelle_utils.pyexpr import analyze as _an
                cids, _, creal = _an(cnode)
            except PyExprError:
                return carrier
            if creal or any(vt_all.get(i) == "real" for i in cids):
                return "real"
            return "int"

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
            ccar = _conj_carrier(c)
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
        nz_list = (nz_divisors(props_src[k], vt_all, carrier)
                   if claims_t else [])
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

    def _probe_step(p):
        """Premise-consistency probe only (goal False from the accumulated
        axioms). Split from the claim cascade so the inconsistency point can
        be resolved BEFORE cascades run: steps at/after that point are forced
        to rewarded=False / pattern 'c' regardless of their proofs, so their
        cascades (3-6 prover calls each) are dead work."""
        rf = _check(make_theorem(fixes_all, p["prem"], "False",
                                 ALTERNATION))
        return bool(rf["success"])

    def _cascade_step(p):
        prem, claims_t = p["prem"], p["claims_t"]
        out = {"verified": None, "tolerance": False}
        nz_prem = []
        for d in p["nz_list"]:
            rnz = _check(make_theorem(fixes_all, prem,
                                      f"{d} \\<noteq> 0", ALTERNATION))
            if rnz["success"]:
                nz_prem.append((f"nz{len(nz_prem)}", f"{d} \\<noteq> 0"))
        full_prem = prem + nz_prem

        def _verify_single(g):
            r = _check(make_theorem(fixes_all, full_prem, g, ALTERNATION))
            if r["success"]:
                return True
            r0 = _check(make_theorem(_canonical_fixes(g), [], g,
                                     ALTERNATION))
            if r0["success"]:
                return True
            # EVAL_RESCUE already imported at module level
            r_eval = _check(make_theorem(_canonical_fixes(g), [], g,
                                         EVAL_RESCUE))
            return r_eval["success"]

        if len(claims_t) > 1:
            out["verified"] = all(_verify_single(g) for g in claims_t)
        else:
            out["verified"] = _verify_single(claims_t[0])
        if not out["verified"]:
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

    # Wave 1: consistency probes for every step; the inconsistency point is
    # the FIRST step whose probe proves False (identical to the sequential
    # first-hit rule -- each probe depends only on its own premise list).
    # With a submitting pool the whole wave enters the FIFO at once -- no
    # per-response executor threads; collection order == step order.
    if _submit is not None:
        _probe_futs = [_submit(make_theorem(fixes_all, p["prem"], "False",
                                            ALTERNATION)) for p in prepped]
        probe_hits = []
        for _f in _probe_futs:
            _r = _f.result()
            _record(_r)
            probe_hits.append(bool(_r["success"]))
    else:
        probe_hits = _pmap(_probe_step, prepped)
    latch = None
    for p, hit in zip(prepped, probe_hits):
        if hit:
            latch = p["k"]
            break
    rec["premise_inconsistent_at"] = latch

    # Wave 2: claim cascades ONLY for steps before the inconsistency point.
    # Steps at/after it get rewarded=False and pattern 'c' regardless of any
    # proof result (reward reads only `rewarded`), so their cascades are
    # skipped entirely; their `verified` stays False. This changes ONLY the
    # verified_steps diagnostic metric for post-inconsistency steps, never a
    # reward.
    to_cascade = [p for p in prepped
                  if p["claims_t"] and (latch is None or p["k"] < latch)]
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
        entry["premises_inconsistent"] = (latch is not None
                                          and p["k"] >= latch)
        if not p["claims_t"]:
            # definition-only step: nothing to prove (conservative), but
            # also nothing to reward - neutral, excluded from the metric
            entry["neutral"] = True
            entry["verified"] = True
            entry["rewarded"] = False
            continue
        res = cascade_out.get(p["k"])
        if res is None:
            entry["verified"] = False   # cascade skipped past the latch
        else:
            entry["verified"] = res["verified"]
            if res["tolerance"]:
                entry["tolerance"] = True
        entry["rewarded"] = (entry["verified"] and entry["guard_ok"]
                             and not entry["premises_inconsistent"]
                             and not entry["transcription_missing"])

    # Per-step symbol string (priority o>c>m>g>x): o=rewarded,
    # c=premises-inconsistent, m=verified-but-transcription-missing,
    # g=verified-but-guard-failed/neutral, x=unverified.
    # NOTE: c is NOT a cascade from a failed step. It is the separate
    # `premise_inconsistent_at` latch above (a real `|- False` proof against
    # the accumulated axioms); an x leaves premises consistent, so x does not
    # force following steps to c. Returned in rec so the reward manager can
    # surface it (e.g. in the [Step Rewards] sample print).
    pat = "".join(("o" if e["rewarded"] else
                   ("c" if e["premises_inconsistent"] else
                    ("m" if e["verified"] and e["transcription_missing"] else
                     ("g" if e["verified"] else "x")))) for e in rec["steps"])
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
