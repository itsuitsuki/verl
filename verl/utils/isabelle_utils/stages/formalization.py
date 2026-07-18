"""The formalization stage of the step-verification pipeline.

The per-response pipeline in engine.process_one runs formalization -> preparation -> verification under engine.py's orchestration; this module is the formalization part. It turns one parsed response from natural language into validated constrained pyexpr (forms A -> B, natural language -> constrained pyexpr; naming contract at the top of state_classes.py) without touching a prover pool: parsing the translator's replies, validating givens and step propositions, the source-faithfulness rules (transcription completeness, invented-number rejection) wired into the translate retry loop, and direct-domain statement parsing.

formalize() is the stage entry point and returns a state_classes.FormalizationOutput.

The parse and faithfulness helpers stay module-level. The preparation stage imports only `conjuncts` for its definition split.

1. parse translator output
2. validate givens
3. validate step propositions
4. enforce source faithfulness
5. parse direct-domain statements

`formalize` returns a complete FormalizationOutput and does not own a prover pool.
"""
import ast
import re
import time
from fractions import Fraction
from pathlib import Path

import verl.utils.isabelle_utils.number_provenance as number_provenance
import verl.utils.isabelle_utils.pyexpr as pyexpr
import verl.utils.isabelle_utils.stages.direct_verify as direct_verify
import verl.utils.isabelle_utils.state_classes as state_classes
import verl.utils.isabelle_utils.theorem_builders as theorem_builders
import verl.utils.isabelle_utils.translator as translator

UNIT_CLOSURE = (Fraction(60), Fraction(100), Fraction(1000))

# Faithfulness allow-list constants: a translated GIVEN may use these even when the problem text does not print them literally, 
# because they are conventional scales/units rather than invented numbers. 
# Grouped by meaning so the "why" is explicit (the boundaries are heuristic, e.g. decimal scales stop at 1e9 = "up to a billion", which is a pragmatic cap, not a principled one).
_DECIMAL_SCALES = {Fraction(10) ** k for k in range(1, 10)}                  # 10, 100, ... 1e9: scientific-notation magnitudes
_UNIT_CONSTANTS = {Fraction(c) for c in (60, 90, 180, 360, 12, 24, 25, 50)}  # angle deg / min-sec / dozen-months / hours / percent
_SMALL_FACTORS = {Fraction(c) for c in (3, 5, 7)}                            # small primes common as bare coefficients
_ALLOWED_UNIT_NUMS = _DECIMAL_SCALES | _UNIT_CONSTANTS | _SMALL_FACTORS


def _eval_values(node, vals):
    """Return numeric subexpression values after substituting pinned variables.

    Only operations explicitly present in the translated proposition are evaluated. This lets the transcription check recognize a number computed by the proposition, regardless of which supported arithmetic operation produced it.
    """
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
        # Compare, BoolOp, and Call are not numeric themselves. Recurse into them so numeric operand subexpressions are still evaluated.
        for ch in ast.iter_child_nodes(n):
            ev(ch)
        return None

    ev(node)
    return out


# Prompt files ship with verl under verl/prompts/, the same root the fol/z3 pipeline uses (fol_utils/common.py PROMPT_ROOT); the isabelle_ prefix keeps them apart from the z3 translate prompts there.
PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts"
PROMPT_GIVENS_PY = (PROMPT_ROOT / "isabelle_translate_givens.txt").read_text(encoding="utf-8")
PROMPT_STEPS_PY = (PROMPT_ROOT / "isabelle_translate_steps.txt").read_text(encoding="utf-8")


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
    """Return True for commentary or placeholder lines that are not translator output. 
    A line with comparison syntax remains eligible for validation so a mathematically wrong expression receives useful feedback instead of being silently discarded."""
    if not p or PLACEHOLDER_RE.search(p):
        return True
    if re.search(r"[=<>]", p):
        return False
    try:
        pyexpr.parse_expr(_normalize_src(p))
        return False
    except pyexpr.PyExprError:
        return True


def parse_givens_vars_to_pyexpr(reply: str):
    lines = reply.splitlines()
    # the canonical block starts at the LAST VARS line; 
    # everything before it is the model's commentary (which loves to quote GIVEN lines)
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
    return state_classes.PyExprGiven(pyexpr_variable_types=fixes,
                                         pyexpr_givens=givens)


def parse_step_translation_to_pyexpr(translator_resp: str):
    """{k: PyExprStep}; the premises part of a STEP line is optional. Form A (natural language) -> form B (pyexpr sources), verbatim for direct-domain statements."""
    out = {}
    step_line_re = re.compile(
        r"\s*STEP\s*(\d+)\s*\|\s*(?:premises?:\s*(.*?)\s*\|\s*)?prop:\s*(.+)",
    re.IGNORECASE)
    for line in translator_resp.splitlines():
        m = step_line_re.match(line.strip())
        if not m:
            continue
        raw = m.group(3).strip().replace('"', ' ').strip()
        praws = [p.strip().replace('"', ' ').strip()
                 for p in (m.group(2) or "").split(";")]
        praws = [p for p in praws if p and p != "-"]
        # A direct-path (group, ...) statement is raw Isabelle, not pyexpr. Routing reads the WHOLE step (conclusion and premises together), so an arithmetic-valued claim such as `order G = 15` still rides with its carrier premises, and a numeric-looking premise inside a direct step keeps its single `=` (normalizing it to `==` would break the Isabelle theorem).
        if direct_verify.match_domain(raw, *praws):
            out[int(m.group(1))] = state_classes.PyExprStep(
                pyexpr_conclusion=raw, pyexpr_premises=praws)
            continue
        prop = _normalize_src(raw)
        if _junk_line(prop):
            continue
        prems = []
        for praw in praws:
            pn = _normalize_src(praw)
            if not _junk_line(pn):
                prems.append(pn)
        out[int(m.group(1))] = state_classes.PyExprStep(
            pyexpr_conclusion=prop, pyexpr_premises=prems)
    return out or None


def expr_info(src: str, pyexpr_variable_types: dict):
    """(isabelle_term, idents, consts, carrier) or raise pyexpr.PyExprError."""
    return pyexpr.py_to_isabelle(src, pyexpr_variable_types)


def conjuncts(src: str):
    node = pyexpr.parse_expr(src)
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
        node = pyexpr.parse_expr(src)
    except pyexpr.PyExprError:
        return
    for c in (node.values if isinstance(node, ast.BoolOp) else [node]):
        if (isinstance(c, ast.Compare) and len(c.ops) == 1
                and isinstance(c.ops[0], ast.Eq)
                and isinstance(c.left, ast.Name)):
            v = pyexpr._const_eval(c.comparators[0])
            if v is not None:
                vals[c.left.id] = v


def transcription_missing_py(
    pyexpr_conclusion,
    nl_conclusion,
    var_vals,
    nl_premises_text,
    context_srcs,
):
    """Return model conclusion numbers omitted or changed by translation.

    Every number stated by the model must occur literally in the translated proposition or be computed by it.
    For example, a conclusion containing ``40`` must not be translated as ``answer == 41``.
    A proposition such as ``price * quantity == 10`` may preserve ``10`` through computation rather than as a standalone literal.
    """
    try:
        node = pyexpr.parse_expr(pyexpr_conclusion)
    except pyexpr.PyExprError:
        return []
    ctx_dumps = set()
    for c_src in context_srcs:
        try:
            ctx_dumps.add(ast.dump(pyexpr.parse_expr(c_src)))
        except pyexpr.PyExprError:
            pass
    present_nums, idents = set(), set()
    for c in (node.values if isinstance(node, ast.BoolOp) else [node]):
        if is_vacuous_node(c) or ast.dump(c) in ctx_dumps:
            continue
        try:
            ids, cs, _ = pyexpr.analyze(c)
        except pyexpr.PyExprError:
            continue
        idents |= ids
        present_nums |= cs
    present_nums |= {var_vals[i] for i in idents if i in var_vals}
    # A variable subscript counts as present. For example, the conclusion may say "the third term" or "the sixth term" while the proposition names ``a3`` or ``a_6``; ``num_values`` does not see the digit inside the variable name.
    for i in idents:
        present_nums |= {Fraction(m) for m in re.findall(r"\d+", i)}
    # Count values actually computed by the translated proposition. For example, ``side1 - side2`` may compute 3 and ``price * quantity`` may compute 10 even when those result values do not occur as standalone literals.
    present_nums |= _eval_values(node, var_vals)
    prem_nums = number_provenance.num_values(nl_premises_text, words=True)
    con_text = re.sub(r"_\{?\d+\}?", "", nl_conclusion)
    out = []
    for v in number_provenance.num_values(con_text, words=True):
        if v in number_provenance.FREE_NUMS or v in prem_nums:
            continue
        forms = {v}
        for c in UNIT_CLOSURE:
            forms.add(v * c)
            forms.add(v / c)
        if forms & present_nums:
            continue
        # An ordinal ("the third / sixth term") is read by num_values as the fraction 1/3, 1/6 -- a position, not a value: if its reciprocal (the position) is present as a subscript (a3, a6), it is transcribed.
        if v.numerator == 1 and Fraction(v.denominator) in present_nums:
            continue
        out.append(str(v))
    return out


def formalize(problem, nl_steps, config, verify, profile):
    """Formalization entry point: translate one parsed response into validated constrained pyexpr.

    Translates the problem conditions and response steps with retries for invalid proposition shape, unfaithful numbers, and incomplete transcription. A direct-domain statement becomes a DirectDomainStep containing verbatim Isabelle; every other statement remains a PyExprStep. The ordered `steps` list is the single source of truth.

    The returned FormalizationOutput leaves fields as ``None`` when their translation did not complete. This stage owns no prover pool; the givens typecheck uses the normalized ``verify`` callable supplied by the orchestrator. ``profile`` must already contain the translator timing keys.
    """
    base_nums = number_provenance.num_values(problem, words=True)
    # num_values skips digits inside names such as `a_1`, `a_4`, and `a_{12}`. Include these indices because sequence relations legitimately use coefficients derived from them.
    base_nums |= {Fraction(m) for m in re.findall(r"_\{?(\d+)", problem)}
    # GIVEN translations may use problem numbers, interior values from stated ranges, and the small fixed constants below. This allow-list rejects numbers invented by the translator and also feeds the step-level checks.
    problem_nums = set(base_nums) | number_provenance.FREE_NUMS | number_provenance.range_ints(problem)
    allowed_given_nums = set(problem_nums) | _ALLOWED_UNIT_NUMS
    # A base number scaled by a unit factor (2 hours -> 120 minutes, 0.3 -> 30 percent) is faithful too.
    for v in base_nums:
        for c in UNIT_CLOSURE:
            allowed_given_nums.add(v * c)
            if v != 0:
                allowed_given_nums.add(v / c)

    def validate_givens(parsed):
        declared, givens = parsed.pyexpr_variable_types, parsed.pyexpr_givens
        pyexpr_variable_types_declared = dict(declared)
        errs, given_term_ids = [], []
        for g in givens:
            try:
                term, ids, consts, carrier = expr_info(g, pyexpr_variable_types_declared)
            except pyexpr.PyExprError as e:
                errs.append(f"GIVEN '{g[:60]}': {e}")
                continue
            bad = [str(v) for v in consts if v not in allowed_given_nums]
            if bad:
                errs.append(f"GIVEN uses numbers not in the problem "
                            f"statement: {sorted(set(bad))}. Only "
                            "transcribe the problem.")
            if re.fullmatch(r"answer\s*==\s*-?[\d./() *]+", g):
                errs.append("Do not state the value of answer as a GIVEN.")
            given_term_ids.append((term, ids))
        if errs:
            return errs
        if not any("answer" in ids for _, ids in given_term_ids):
            return ["No GIVEN defines answer. Add exactly one GIVEN naming "
                    "the asked quantity (answer == <variable>), or the "
                    "question's own expression - never a derived formula, "
                    "never a numeric value."]
        # Auto-declare identifiers omitted from VARS so a missing declaration does not reject an otherwise usable translation.
        undeclared = set().union(*(ids for _, ids in given_term_ids)) - set(pyexpr_variable_types_declared)
        for u in sorted(undeclared):
            declared.append((u, "real"))
            pyexpr_variable_types_declared[u] = "real"
        if undeclared:
            given_term_ids.clear()
            for g in givens:
                term, ids, _, _ = expr_info(g, pyexpr_variable_types_declared)
                given_term_ids.append((term, ids))
        thm = theorem_builders.make_theorem(declared, [(f"g{i}", t) for i, (t, _) in
                                   enumerate(given_term_ids)], "True", "simp")
        r = verify(thm)
        if not r.proved:
            return r.errors[:2] or ["givens skeleton rejected"]
        return []

    result = state_classes.FormalizationOutput(nl_steps=nl_steps, problem_nums=problem_nums)

    _time_start = time.time()
    parsed_a, attempt_a, _ = translator.translate(
        PROMPT_GIVENS_PY.replace("{problem}", problem), parse_givens_vars_to_pyexpr,
        validate_givens,
        translator_url=config.translator_url,
        translator_model=config.translator_model,
        max_model_len=config.max_model_len,
        api_timeout=config.api_timeout,
        soft_prefix=("GIVEN uses numbers", "No GIVEN defines answer"))
    profile["translate_validate_time"] += time.time() - _time_start
    profile["translator_http_time"] += sum(
        float(a.get("http_wall_time") or 0.0) for a in attempt_a
        if isinstance(a, dict))
    result.translation_record_from_problem = attempt_a
    if parsed_a is None:
        return result
    pyexpr_givens, pyexpr_variable_types_declared = parsed_a.pyexpr_givens, dict(parsed_a.pyexpr_variable_types)
    result.pyexpr_givens, result.pyexpr_variable_types_declared, result.givens_ok = pyexpr_givens, pyexpr_variable_types_declared, True

    def validate_props(propositions, required_end=None):
        required_end = required_end or len(nl_steps)
        errs = []
        if not set(range(1, required_end + 1)) <= set(propositions.keys()):
            return [f"Expected one STEP line per step "
                    f"(1..{required_end}), got steps "
                    f"{sorted(propositions.keys())}."]
        plist = [propositions[k + 1].pyexpr_conclusion for k in range(required_end)]
        # the root of a proposition must be boolean-shaped: comparison,
        # and/or, or unary not. Bare values (8, x, x + 1) are TERMS, not
        # propositions; the transpiler still emits something for them
        # (`(8::int)`) but Isabelle rejects it as an obligation. Reject
        # here so the translator gets a clear retry signal.
        for k, p in enumerate(plist):
            try:
                root = pyexpr.parse_expr(p)
            except pyexpr.PyExprError:
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
        pyexpr_variable_types_local = dict(pyexpr_variable_types_declared)
        for k in range(required_end):
            translated = propositions[k + 1]
            if direct_verify.match_domain(translated.pyexpr_conclusion, *translated.pyexpr_premises):
                continue   # direct-path step: verbatim Isabelle, not pyexpr (routing reads the whole step)
            for src in [translated.pyexpr_conclusion] + translated.pyexpr_premises:
                try:
                    _, ids, _, _ = expr_info(src, pyexpr_variable_types_local)
                    for i in ids:
                        pyexpr_variable_types_local.setdefault(i, "real")
                except pyexpr.PyExprError as e:
                    errs.append(f"STEP {k + 1} '{src[:50]}': {e}")
        if errs:
            return errs[:5]
        for k, p in enumerate(plist):
            if direct_verify.match_domain(p, *propositions[k + 1].pyexpr_premises):
                continue   # direct-path step: vacuity handled by direct_verify
            for c in conjuncts(p):
                if is_vacuous_node(c) and len(conjuncts(p)) > 1:
                    errs.append(f"STEP {k + 1}: drop the trivially-true "
                                "conjunct; every conjunct must carry "
                                "content from the conclusion.")
                    break
            if is_vacuous_node(pyexpr.parse_expr(p)):
                errs.append(f"STEP {k + 1}: proposition is trivially true; "
                            "state the conclusion's actual content.")
        var_vals = {}
        for g in pyexpr_givens:
            pin_from(g, var_vals)
        t_errs = []
        for k, nl_step in enumerate(nl_steps[:required_end]):
            miss = transcription_missing_py(
                plist[k], nl_step.nl_conclusion, var_vals,
                " ".join(nl_step.nl_premises),
                list(pyexpr_givens) + plist[:k])
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

    # Long solutions are translated in chunks of fresh steps. Earlier chunks remain in the prompt as context but are not re-emitted. A retry within a chunk may re-emit only the rejected step lines; the parser merges them into the existing chunk result.
    CHUNK = config.translate_chunk_steps
    vars_givens_text = ("VARS: "
        + ", ".join(f"{n} {t}" for n, t in parsed_a.pyexpr_variable_types)
        + "\n" + "\n".join(f"GIVEN: {g}" for g in pyexpr_givens))
    parsed_all, attempt_b_list, failed_b = {}, [], False
    for c0 in range(0, len(nl_steps), CHUNK):
        block_end = min(c0 + CHUNK, len(nl_steps))
        steps_text = "\n\n".join(
            f"STEP {k + 1}:\n"
            + "\n".join(f"premise: {p}" for p in nl_steps[k].nl_premises)
            + f"\nconclusion: {nl_steps[k].nl_conclusion}"
            for k in range(c0, block_end))
        if c0:
            steps_text = (
                "Steps already translated (context only, do NOT re-output "
                "them):\n"
                + "\n".join(f"STEP {j + 1} | prop: "
                            f"{parsed_all[j + 1].pyexpr_conclusion}"
                            for j in range(c0))
                + "\n\nNEW STEPS:\n\n" + steps_text)
        merged_steps = dict(parsed_all)

        def parse_props_merge(reply, _m=merged_steps):
            got = parse_step_translation_to_pyexpr(reply)
            if got:
                _m.update(got)
            return dict(_m) if _m else None

        def validate_block(props, _end=block_end):
            return validate_props(props, _end)

        _time_start = time.time()
        parsed_b, att_b, _ = translator.translate(
            PROMPT_STEPS_PY.replace("{vars_givens}", vars_givens_text).replace("{steps}", steps_text),
            parse_props_merge, validate_block,
            translator_url=config.translator_url,
        translator_model=config.translator_model,
            max_model_len=config.max_model_len,
            api_timeout=config.api_timeout,
            soft_prefix="STEP TRANSCRIPTION")
        profile["translate_validate_time"] += time.time() - _time_start
        profile["translator_http_time"] += sum(
            float(a.get("http_wall_time") or 0.0) for a in att_b
            if isinstance(a, dict))
        attempt_b_list.append(att_b)
        if parsed_b is None:
            failed_b = True
            break
        parsed_all = {k: v for k, v in parsed_b.items() if k <= block_end}
    result.translation_record_from_steps = (attempt_b_list[0] if len(attempt_b_list) == 1
                          else attempt_b_list)
    if failed_b:
        return result
    result.steps_ok = True
    # One typed entry per response step, in order. Routing checks conclusion and premises together: a match becomes a DirectDomainStep with its verbatim Isabelle text (pyexpr cannot parse it); everything else stays the parser's PyExprStep.
    steps = []
    for k in range(len(nl_steps)):
        parsed_step = parsed_all[k + 1]
        spec = direct_verify.match_domain(parsed_step.pyexpr_conclusion, *parsed_step.pyexpr_premises)
        if spec is None:
            steps.append(parsed_step)
        else:
            steps.append(state_classes.DirectDomainStep(
                spec=spec, premises=list(parsed_step.pyexpr_premises),
                claim=parsed_step.pyexpr_conclusion, nl_step_text=nl_steps[k].nl_step_text))
    # The authoritative per-step transcription record. The identical computation inside validate_props only feeds the translator retry loop (soft-accepted misses survive it, and a disk-cache hit skips validation entirely), so the values that reach the solution record are computed exactly once, here.
    # Transcription applies to the general path only: a direct-domain step records [] and is skipped entirely; its verbatim Isabelle is not pyexpr, its numbers must not exempt later general steps from transcription, and its faithfulness check is the entity anchor in direct_verify.
    var_vals, transcription_missing = {}, []
    for g in pyexpr_givens:
        pin_from(g, var_vals)
    general_context = list(pyexpr_givens)
    for k, step in enumerate(steps):
        if isinstance(step, state_classes.DirectDomainStep):
            transcription_missing.append([])
            continue
        transcription_missing.append(transcription_missing_py(
            step.pyexpr_conclusion, nl_steps[k].nl_conclusion, var_vals,
            " ".join(nl_steps[k].nl_premises), list(general_context)))
        pin_from(step.pyexpr_conclusion, var_vals)
        general_context.append(step.pyexpr_conclusion)
    result.steps, result.transcription_missing = steps, transcription_missing
    return result
