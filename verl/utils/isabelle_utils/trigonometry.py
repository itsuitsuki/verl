"""Structured trigonometric evidence extraction and bounded Isabelle proof attempts.

The general trigonometry system (design note: Notion 三角tactic的泛用化) replaces the old single-purpose quadrant theorem.
Preparation extracts a typed TrigContext from ADMISSION-PASSED constrained-pyexpr sources;
verification asks plan_attempts for a bounded, deterministically ordered list of independent proof attempts and submits each one on its own.
Quadrant reasoning is only a SIGN-FACT PROVIDER here:
strict principal-quadrant bounds yield cos/sin/tan sign and cos-nonzero auxiliary facts that the value families consume;
it is not a proof family itself.

Soundness rules this module enforces by construction:
- evidence binds to the exact named Isabelle assumptions it came from;
  nothing re-selects premises by string search, and conflicting values for one trig application make that key ambiguous instead of picking a side;
- Python never inspects the claimed value to choose a sign branch;
  the meta lemma orientation follows the PROVED cosine/sine sign fact, and every Python-computed candidate (an exact square root) is restated as a local fact the kernel checks;
- render_attempt is the single renderer for an attempt's positive theorem and its `False` guard twin:
  the two renderings are byte-identical except the goal, so the guard carries exactly the strength of the positive proof (assumptions, auxiliary facts, final tactic);
- tangent values are only ever derived through lemmas whose hypotheses include the cosine sign (hence nonzero);
  no attempt clears a denominator without that fact.
"""

import ast
import math
from dataclasses import dataclass
from fractions import Fraction

import verl.utils.isabelle_utils.pyexpr as pyexpr
import verl.utils.isabelle_utils.state_classes as state_classes
import verl.utils.isabelle_utils.theorem_builders as theorem_builders

TRIG_FUNCTIONS = frozenset({"sin", "cos", "tan"})

# One closing tactic for every guarded attempt:
# linarith chains the local value facts (and, in the False twin, refutes a tangent premise against the derived tangent sign),
# and the close-or-fail simp handles the rational/radical arithmetic the value facts reduce claims to.
GUARDED_FINAL_TACTIC = ("((linarith) | (simp add: power2_eq_square real_sqrt_pow2 "
                        "real_sqrt_divide field_simps; fail))")

# Premise-free identity groups, each one small and orientation-fixed (a single grown simp set measured as timeouts/unfinished goals on the real pool).
# Expansion covers angle addition and, through the [simp] values of sin/cos at 0, pi/2, pi, and 2*pi, also pi-symmetry, cofunction, and literal-period shifts.
# Families that are mutual inverses stay in SEPARATE attempts so the rewrites can never cycle in one set:
# the two Pythagorean squared orientations, product-to-sum versus sum-to-product, and the two cos double-angle forms.
PREMISE_FREE_GROUPS = (
    ("SPECIAL_ANGLE", "(simp add: sin_30 cos_30 sin_45 cos_45 sin_60 cos_60)"),
    ("IDENTITY_EXPANSION", "(simp add: sin_add sin_diff cos_add cos_diff)"),
    ("SPECIAL_ANGLE_EXPANDED",
     "(simp add: sin_add sin_diff cos_add cos_diff "
     "sin_30 cos_30 sin_45 cos_45 sin_60 cos_60)"),
    ("PYTHAGOREAN_SIN_SQUARED", "(simp add: sin_squared_eq power2_eq_square algebra_simps)"),
    ("PYTHAGOREAN_COS_SQUARED", "(simp add: cos_squared_eq power2_eq_square algebra_simps)"),
    ("PRODUCT_TO_SUM",
     "(simp add: sin_times_sin sin_times_cos cos_times_sin cos_times_cos)"),
    ("SUM_TO_PRODUCT",
     "(simp add: sin_plus_sin sin_diff_sin cos_plus_cos cos_diff_cos)"),
    # No algebra_simps here:
    # it commutes `2 * x` into `x * 2` BEFORE the double-angle rules can match their `sin (2 * x)` left-hand sides (measured on the real pool).
    ("DOUBLE_ANGLE_COS_FORM", "(simp add: sin_double cos_double_cos)"),
    ("DOUBLE_ANGLE_SIN_FORM", "(simp add: sin_double cos_double_sin)"),
    ("DOUBLE_ANGLE_DIFF_FORM", "(simp add: sin_double cos_double power2_eq_square)"),
    # Inverse trig compositions:
    # every rewrite in this set is CONDITIONAL on its range side condition (sin_arcsin needs -1 <= y <= 1, arcsin_sin needs -pi/2 <= x <= pi/2, ...),
    # and simp discharges those conditions itself for literal in-range arguments;
    # out-of-range arguments leave the rewrite unfired and the claim unproved
    # (HOL's arcsin outside the range is unspecified but proves nothing, so no totalization pre-check is needed here).
    ("INVERSE_TRIG", "(simp add: sin_arcsin cos_arccos tan_arctan "
     "cos_arcsin sin_arccos arcsin_sin arccos_cos arctan_tan)"),
    # Tangent identity rewrites over variable angles (tan(-a) = -tan(a), tan(pi - a) = -tan(a), tan(a) = sin(a)/cos(a)):
    # unfolding tan_def turns both sides into sine/cosine ratios that the expansion evaluates;
    # HOL's sign-through-division rules close them without any nonzero condition;
    # value pinning stays excluded by the tangent definedness check.
    ("TAN_IDENTITY_EXPANSION", "(simp add: tan_def sin_add sin_diff cos_add cos_diff)"),
)

# The four quadrant sign lemmas the cosine-nonzero check tries for a with-premises condition, each carrying its cosine-sign orientation.
# Flat tactic forms were all measured failing in this rendering position
# (smt: verit rejects, z3 exceeds the watchdog; rule chains: the builder's chained assumptions derail `rule` even behind `-` or `insert`),
# so the check renders cosine_nonzero_theorem instead, built ONLY from attempt-proven shapes.
QUADRANT_SIGN_LEMMAS = (("cos_pos_q1", True), ("cos_pos_q4", True),
                        ("cos_neg_q2", False), ("cos_neg_q3", False))


def cosine_nonzero_theorem(fixes, premises, angle_term: str, lemma: str,
                           positive: bool, goal: str) -> str:
    """Hand-rendered Isar theorem proving `goal` from one quadrant sign lemma:
    the sign fact s is derived exactly like the attempt-internal cs/csd facts (`using <premises> pi_gt_zero by (intro L; linarith)`), then simp turns the strict sign into the nonzero goal.
    ONE renderer serves the positive condition and its False guard twin (only `goal` changes), keeping the equal-strength doctrine's byte-identity."""
    relation = f"0 < cos {angle_term}" if positive else f"cos {angle_term} < 0"
    assumptions = theorem_builders._assumes_clause(premises)
    names = " ".join(name for name, _ in premises)
    using = f"using {names} pi_gt_zero " if names else "using pi_gt_zero "
    return (f"theorem chk:\n{theorem_builders.fixes_clause(fixes)}{assumptions}"
            f'  shows "{goal}"\n  proof -\n'
            f'  have s: "{relation}" {using}by (intro {lemma}; linarith)\n'
            f"  show ?thesis using s by simp\nqed")

# One extra premise-free tactic verification may use to discharge a tangent definedness condition at a special angle
# (cos(pi/4) != 0 needs the special values ALTERNATION lacks);
# cos_arctan covers the arctan composition (cos(arctan 5) = 1/sqrt 26 != 0).
SPECIAL_VALUE_TACTIC = "(simp add: sin_30 cos_30 sin_45 cos_45 sin_60 cos_60 cos_arctan)"

# Principal quadrants over exact pi multiples: (lower, upper, cos sign lemma, cos positive, sin positive).
# The sin lemmas are Isa_Step_Base's sin_pos_upper / sin_neg_lower;
# all four cos lemmas and both sin lemmas take STRICT bounds, matched by the strict-only admission in _bound_evidence's consumers.
_QUADRANTS = (
    (Fraction(0), Fraction(1, 2), "cos_pos_q1", True, True),
    (Fraction(1, 2), Fraction(1), "cos_neg_q2", False, True),
    (Fraction(1), Fraction(3, 2), "cos_neg_q3", False, False),
    (Fraction(3, 2), Fraction(2), "cos_pos_q4", True, False),
)

MAX_ATTEMPTS = 24       # hard cap across every family for one claim

# Pi coefficients whose sin/cos the plain groups already evaluate ([simp] values at 0, pi/2, pi, 2*pi and the special 30/45/60 facts);
# literal angles OUTSIDE this set go through the LITERAL_ANGLE decomposition attempt.
_BASE_PI_COEFFS = frozenset((Fraction(0), Fraction(1, 6), Fraction(1, 4), Fraction(1, 3),
                             Fraction(1, 2), Fraction(1), Fraction(2)))

# Whole-2*pi-period cap for the nested decomposition/transfer layers (literal angles and interval sign transfer):
# each period is one nested `+/- 2*pi` layer folded by one sin_add/cos_diff application, so the bound keeps terms and simp work small;
# angles beyond it stay unproved (fail closed, a FN only).
_MAX_PERIOD_SHIFTS = 8


@dataclass(frozen=True)
class TrigAuxiliaryFact:
    """One deterministic local Isabelle fact used by a trigonometric attempt."""

    name: str
    proposition: str
    proof: str


@dataclass(frozen=True)
class TrigProofAttempt:
    """One bounded trigonometric proof attempt, independent of other tactics."""

    name: str
    premises: tuple[tuple[str, str], ...]
    auxiliary_facts: tuple[TrigAuxiliaryFact, ...]
    tactic: str
    # Names of auxiliary facts applied through the Isar `unfolding` keyword instead of the `using` list:
    # unfolding rewrites the goal left-to-right at the meta level, immune to simp's reorientation and to the arithmetic simproc re-collapsing the rewritten form.
    unfold_fact_names: tuple[str, ...] = ()
    # Rendered terms marking what this attempt is about
    # (an evidence family stores the angle or pinned-variable term its facts concern).
    # Empty means always relevant: the premise-free groups and the claim-derived attempts.
    # _goal_relevant compares a term's free-identifier SET against the goal's, never substrings
    # (pv_x1 as a substring would wrongly count as relevant to a pv_x11 goal, measured);
    # plan_attempts orders relevant attempts ahead of the rest BEFORE the MAX_ATTEMPTS cap,
    # so unrelated premises can no longer crowd the claim's own evidence out of the window.
    relevance_terms: tuple[str, ...] = ()

    @property
    def premise_dependent(self) -> bool:
        """True only when the attempt ASSUMES something.
        An attempt whose auxiliary facts are proven from nothing (a literal-angle decomposition equality, a kernel-checked square root) has no assumptions a contradiction could hide in,
        so it needs no False guard twin and stays available under unknown premise consistency, exactly like the plain premise-free groups."""
        return bool(self.premises)


def _linear_form(node: ast.AST):
    """Exact linear form of an expression as {name: coefficient} over Fractions, with the rational constant under the "" key and pi under its own name;
    None for anything the exact rational arithmetic cannot express (a nonlinear term, a function call, a division by a non-constant).
    Zero coefficients are dropped, so equal forms mean equal real functions."""
    if isinstance(node, ast.Constant):
        try:
            value = pyexpr._const_frac(node)
        except (pyexpr.PyExprError, TypeError, ValueError):
            return None
        return {"": value} if value != 0 else {}
    if isinstance(node, ast.Name):
        return {node.id: Fraction(1)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        form = _linear_form(node.operand)
        if form is None or isinstance(node.op, ast.UAdd):
            return form
        return {name: -coefficient for name, coefficient in form.items()}
    if not isinstance(node, ast.BinOp):
        return None
    left, right = _linear_form(node.left), _linear_form(node.right)
    if left is None or right is None:
        return None
    if isinstance(node.op, (ast.Add, ast.Sub)):
        sign = 1 if isinstance(node.op, ast.Add) else -1
        merged = dict(left)
        for name, coefficient in right.items():
            merged[name] = merged.get(name, Fraction(0)) + sign * coefficient
        return {name: c for name, c in merged.items() if c != 0}
    if isinstance(node.op, ast.Mult):
        for constant, other in ((left, right), (right, left)):
            if set(constant) <= {""}:
                scale = constant.get("", Fraction(0))
                return {name: c * scale for name, c in other.items() if c * scale != 0}
        return None
    if isinstance(node.op, ast.Div) and set(right) <= {""} and right.get("", 0) != 0:
        return {name: c / right[""] for name, c in left.items()}
    return None


def _angle_key(node: ast.AST) -> str:
    """Canonical identity of an angle for evidence joining and conflict detection:
    the exact linear form when one exists, so the algebraically equal spellings 2*a, a*2, and a+a share one key
    (their bounds, signs, and values connect; the rendered proofs still use each premise's own spelling),
    with the raw AST dump as the fallback identity for nonlinear angles.
    Distinct linear forms are distinct real functions, so a merge never conflates genuinely different angles."""
    form = _linear_form(node)
    if form is not None:
        return "lin:" + ";".join(f"{name}={coefficient}"
                                 for name, coefficient in sorted(form.items()))
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _source(node: ast.AST) -> str:
    return ast.unparse(node)


def _transpile(node: ast.AST, variable_types: dict) -> str | None:
    try:
        return pyexpr.transpile(node, variable_types, "real")
    except pyexpr.PyExprError:
        return None


def _trig_call(node: ast.AST):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in TRIG_FUNCTIONS and len(node.args) == 1):
        return node.func.id, node.args[0]
    return None


def _constant_sign(node: ast.AST) -> int | None:
    value = pyexpr._const_eval(node)
    if value is None:
        return None
    return (value > 0) - (value < 0)


def _pi_linear(node: ast.AST):
    """Return `(constant, pi coefficient)` over exact rationals, or None."""

    if isinstance(node, ast.Constant):
        try:
            return pyexpr._const_frac(node), Fraction(0)
        except (pyexpr.PyExprError, TypeError, ValueError):
            return None
    if isinstance(node, ast.Name):
        return (Fraction(0), Fraction(1)) if node.id == "pi" else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _pi_linear(node.operand)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else (-value[0], -value[1])
    if not isinstance(node, ast.BinOp):
        return None
    left, right = _pi_linear(node.left), _pi_linear(node.right)
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Add):
        return left[0] + right[0], left[1] + right[1]
    if isinstance(node.op, ast.Sub):
        return left[0] - right[0], left[1] - right[1]
    if isinstance(node.op, ast.Mult):
        if left[1] == 0:
            return left[0] * right[0], left[0] * right[1]
        if right[1] == 0:
            return left[0] * right[0], left[1] * right[0]
        return None
    if isinstance(node.op, ast.Div) and right[1] == 0 and right[0] != 0:
        return left[0] / right[0], left[1] / right[0]
    return None


def exact_pi_coefficient(node: ast.AST) -> Fraction | None:
    """Exact coefficient when the expression is a rational multiple of pi (a bare 0 counts);
    a nonzero pi-free component (a radian literal, pi + 1) is NOT a pi multiple."""

    value = _pi_linear(node)
    if value is None or value[0] != 0:
        return None
    return value[1]


def _exact_value(node: ast.AST) -> Fraction | None:
    """Exact rational value of a closed numeric expression, else None."""
    return pyexpr._const_eval(node)


def _exact_sqrt_int_arg(node: ast.AST) -> Fraction | None:
    """The integer n when the node is exactly sqrt(n) or c*sqrt(n)? No: only bare sqrt(n)."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt" and len(node.args) == 1):
        return pyexpr._const_eval(node.args[0])
    return None


def _rational_sqrt(value: Fraction) -> Fraction | None:
    """The exact rational square root of `value`, or None when it is not a perfect square."""
    if value < 0:
        return None
    num, den = value.numerator, value.denominator
    rn, rd = _isqrt_exact(num), _isqrt_exact(den)
    if rn is None or rd is None:
        return None
    return Fraction(rn, rd)


def _isqrt_exact(n: int):
    """Exact integer square root or None.
    math.isqrt keeps this pure integer arithmetic:
    a float detour would silently return wrong roots around 100 digits and raise OverflowError around 400 digits,
    and trig planning runs before the dangerous-term check, so a giant literal must degrade to None, never crash."""
    if n < 0:
        return None
    root = math.isqrt(n)
    return root if root * root == n else None


def _hypotenuse_root(value_node: ast.AST) -> Fraction | None:
    """The exact value of sqrt(t^2 + 1) for the tangent value t, when Python can compute it and Isabelle can re-check it
    (t a rational with a Pythagorean-triple hypotenuse, or t = sqrt(n) with n + 1 a perfect square).
    The root is only a CANDIDATE:
    it is restated as a local fact proved through real_sqrt_unique, so a wrong candidate fails in the kernel."""
    t = _exact_value(value_node)
    if t is not None:
        return _rational_sqrt(t * t + 1)
    sqrt_arg = _exact_sqrt_int_arg(value_node)
    if sqrt_arg is not None and sqrt_arg >= 0:
        return _rational_sqrt(sqrt_arg + 1)
    return None


def _unit_complement_root(value_node: ast.AST) -> Fraction | None:
    """The exact value of sqrt(1 - v^2) for a rational sin/cos value v with |v| <= 1, when it is rational;
    same candidate-then-kernel-check contract as _hypotenuse_root."""
    v = _exact_value(value_node)
    if v is None or abs(v) > 1:
        return None
    return _rational_sqrt(1 - v * v)


def _fraction_term(value: Fraction) -> str:
    if value.denominator == 1:
        return f"({value.numerator}::real)"
    return f"(({value.numerator}::real) / ({value.denominator}::real))"


def _comparison_parts(node: ast.Compare):
    terms = [node.left] + list(node.comparators)
    for index, op in enumerate(node.ops):
        yield terms[index], op, terms[index + 1]


def _conjuncts(node: ast.AST):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        for value in node.values:
            yield from _conjuncts(value)
    else:
        yield node


def _value_call(node: ast.AST):
    """Like _trig_call but also matching the inverse functions, so `arctan(x) == pi/6` premises become value evidence (function \"arctan\");
    the direct trig families filter by function name and never consume these."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and len(node.args) == 1
            and node.func.id in ("sin", "cos", "tan", "arcsin", "arccos", "arctan")):
        return node.func.id, node.args[0]
    return None


def _value_evidence(node: ast.AST, source_name: str, variable_types: dict):
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)):
        return None
    left, right = node.left, node.comparators[0]
    call = _value_call(left)
    value = right
    if call is None:
        call = _value_call(right)
        value = left
    if call is None:
        return None
    function, angle = call
    angle_term = _transpile(angle, variable_types)
    value_term = _transpile(value, variable_types)
    if angle_term is None or value_term is None:
        return None
    return state_classes.TrigValueEvidence(
        source_names=(source_name,), function=function,
        angle_key=_angle_key(angle), angle_source=_source(angle),
        angle_term=angle_term, value_source=_source(value),
        value_term=value_term)


def _sign_evidence(left, op, right, source_name, variable_types):
    call = _trig_call(left)
    sign = _constant_sign(right)
    positive = None
    if call is not None and sign == 0:
        if isinstance(op, ast.Gt):
            positive = True
        elif isinstance(op, ast.Lt):
            positive = False
    if positive is None:
        call = _trig_call(right)
        sign = _constant_sign(left)
        if call is not None and sign == 0:
            if isinstance(op, ast.Lt):
                positive = True
            elif isinstance(op, ast.Gt):
                positive = False
    if positive is None or call is None:
        return None
    function, angle = call
    angle_term = _transpile(angle, variable_types)
    if angle_term is None:
        return None
    return state_classes.TrigSignEvidence(
        source_names=(source_name,), function=function,
        angle_key=_angle_key(angle), angle_source=_source(angle),
        angle_term=angle_term, positive=positive)


def _bound_evidence(left, op, right, source_name, variable_types):
    if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        return None
    left_pi, right_pi = exact_pi_coefficient(left), exact_pi_coefficient(right)
    if left_pi is None and right_pi is None:
        return None
    strict = isinstance(op, (ast.Lt, ast.Gt))
    if left_pi is not None and right_pi is None:
        angle, coefficient = right, left_pi
        side = "lower" if isinstance(op, (ast.Lt, ast.LtE)) else "upper"
    elif right_pi is not None and left_pi is None:
        angle, coefficient = left, right_pi
        side = "upper" if isinstance(op, (ast.Lt, ast.LtE)) else "lower"
    else:
        return None
    angle_term = _transpile(angle, variable_types)
    if angle_term is None:
        return None
    proposition = _transpile(
        ast.Compare(left=left, ops=[op], comparators=[right]), variable_types)
    if proposition is None:
        return None
    return state_classes.PiBoundEvidence(
        source_names=(source_name,), angle_key=_angle_key(angle),
        angle_source=_source(angle), angle_term=angle_term,
        side=side, coefficient=coefficient, strict=strict,
        proposition=proposition)


def extract_trig_context(sources, variable_types: dict) -> state_classes.TrigContext:
    """Extract trig values, signs, and exact-pi bounds from admitted sources.
    Conflicting values for one trig application mark that key ambiguous instead of picking either side;
    whether the assumptions are contradictory stays the consistency machinery's business."""

    values, signs, bounds = [], [], []
    for source in sources:
        try:
            root = pyexpr.parse_expr(source.pyexpr_source)
        except pyexpr.PyExprError:
            continue
        for node in _conjuncts(root):
            value = _value_evidence(node, source.name, variable_types)
            if value is not None:
                values.append(value)
            if not isinstance(node, ast.Compare):
                continue
            for left, op, right in _comparison_parts(node):
                sign = _sign_evidence(left, op, right, source.name, variable_types)
                if sign is not None:
                    signs.append(sign)
                bound = _bound_evidence(left, op, right, source.name, variable_types)
                if bound is not None:
                    bounds.append(bound)

    grouped = {}
    for value in values:
        grouped.setdefault((value.function, value.angle_key), set()).add(value.value_term)
    ambiguous = tuple(sorted(key for key, value_terms in grouped.items()
                             if len(value_terms) > 1))
    values = [value for value in values
              if (value.function, value.angle_key) not in ambiguous]
    unique = lambda items: tuple(dict.fromkeys(items))  # noqa: E731
    return state_classes.TrigContext(
        values=unique(values), signs=unique(signs), pi_bounds=unique(bounds),
        ambiguous_value_keys=ambiguous)


def render_attempt(fixes, goal: str, attempt: TrigProofAttempt) -> str:
    """Render one deterministic attempt.
    The positive theorem and the `False` guard twin come from this ONE function with only `goal` changed,
    so they are byte-identical in fixes, assumptions, local facts, fact names, `using` list, and final tactic."""

    assumptions = theorem_builders._assumes_clause(attempt.premises)
    facts = "\n".join(
        f'  have {fact.name}: "{fact.proposition}" {fact.proof}'
        for fact in attempt.auxiliary_facts)
    using_names = [name for name, _ in attempt.premises]
    using_names.extend(fact.name for fact in attempt.auxiliary_facts
                       if fact.name not in attempt.unfold_fact_names)
    using = "using " + " ".join(using_names) + " " if using_names else ""
    unfolding = ("unfolding " + " ".join(attempt.unfold_fact_names) + " "
                 if attempt.unfold_fact_names else "")
    if facts:
        body = (f"proof -\n{facts}\n  show ?thesis {unfolding}{using}"
                f"by {attempt.tactic}\nqed")
    else:
        body = f"{using}by {attempt.tactic}"
    return (f"theorem chk:\n{theorem_builders.fixes_clause(fixes)}{assumptions}"
            f'  shows "{goal}"\n  {body}')


def attempt_fixes(step_fixes, goal: str, attempt: TrigProofAttempt):
    """Minimal sorted fixes for one attempt:
    only the step variables that occur in the goal, the attempt's premises, or its auxiliary facts.
    Sorting canonicalizes the theorem text so identical attempts share one theorem-cache entry."""
    texts = [goal]
    texts.extend(term for _, term in attempt.premises)
    texts.extend(fact.proposition for fact in attempt.auxiliary_facts)
    tokens = set()
    for text in texts:
        tokens |= theorem_builders.identifiers(text)
    return sorted((name, sort) for name, sort in step_fixes if name in tokens)


# ---------------------------------------------------------------------------
# Sign-fact provider (quadrant bounds -> cos/sin/tan sign auxiliary facts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QuadrantSigns:
    """Sign facts derivable for one angle from strict quadrant bounds.
    `shift` counts the whole 2*pi periods subtracted to reach the principal quadrant (any integer up to _MAX_PERIOD_SHIFTS):
    0 means the bounds already sit inside [0, 2*pi);
    otherwise the sign facts are proved on the nested shifted term first (one `- 2*pi` / `+ 2*pi` layer per period),
    then transferred to the original angle through the angle-addition expansion, one layer per simp pass (cos((a - 2*pi) - 2*pi) folds to cos a)."""

    angle_term: str
    cos_positive: bool
    sin_positive: bool
    bound_names: tuple[str, ...]     # assumption names of the contributing bounds
    shift: int = 0


def _quadrant_signs(context: state_classes.TrigContext):
    """{angle_key: _QuadrantSigns} for every angle whose STRICT admitted bounds fit inside one principal quadrant,
    directly or after shifting by whole 2*pi periods (e.g. 5*pi/2 < a < 3*pi is Q2 shifted by +1, 9*pi/2 < a < 5*pi is Q2 shifted by +2).
    Non-strict bounds never contribute (an endpoint leaves the signs open),
    and an empty or cross-quadrant interval yields nothing (fail closed)."""
    per_angle = {}
    for bound in context.pi_bounds:
        if not bound.strict:
            continue
        entry = per_angle.setdefault(bound.angle_key, {"term": bound.angle_term,
                                                       "lower": None, "upper": None,
                                                       "names": []})
        if bound.side == "lower":
            if entry["lower"] is None or bound.coefficient > entry["lower"]:
                entry["lower"] = bound.coefficient
        else:
            if entry["upper"] is None or bound.coefficient < entry["upper"]:
                entry["upper"] = bound.coefficient
        entry["names"].extend(bound.source_names)
    signs = {}
    for angle_key, entry in per_angle.items():
        lower, upper = entry["lower"], entry["upper"]
        if lower is None or upper is None or lower >= upper:
            continue
        # the unique whole-period shift bringing the lower bound into [0, 2*pi);
        # an interval inside one quadrant never straddles a period boundary, so this shift is the only candidate worth checking
        shift = lower // 2
        if abs(shift) > _MAX_PERIOD_SHIFTS:
            continue
        shifted_lower, shifted_upper = lower - 2 * shift, upper - 2 * shift
        for q_lower, q_upper, _lemma, cos_positive, sin_positive in _QUADRANTS:
            if q_lower <= shifted_lower and shifted_upper <= q_upper:
                signs[angle_key] = _QuadrantSigns(
                    angle_term=entry["term"], cos_positive=cos_positive,
                    sin_positive=sin_positive,
                    bound_names=tuple(dict.fromkeys(entry["names"])),
                    shift=int(shift))
                break
    return signs


def _cos_sign_lemma(lower: Fraction, upper: Fraction) -> str | None:
    for q_lower, q_upper, lemma, _cos_positive, _sin_positive in _QUADRANTS:
        if q_lower <= lower and upper <= q_upper:
            return lemma
    return None


def _sign_fact_chain(angle_term: str, signs: _QuadrantSigns, lower, upper):
    """The deterministic cs/sn/nz/ts auxiliary chain for one quadrant-bounded angle.
    Validated shapes:
    the cos lemma and the sin lemma both discharge their strict bound premises through `(intro L; linarith)` with the bound assumptions and `0 < pi` in scope;
    the tangent sign follows from tan_def and the matching divide sign lemma;
    each fact cites only names that exist in the rendering.
    For a period-shifted interval (signs.shift of +1/-1), the sign lemmas prove csr/snr on the SHIFTED term first (its bounds are what linarith derives from the assumptions),
    and the angle-addition expansion transfers each sign onto the original angle (cos(a - 2*pi) simplifies to cos a through cos_diff plus the [simp] values at 2*pi);
    the shifted quadrant selects the lemma."""
    shift = signs.shift
    cos_lemma = _cos_sign_lemma(lower - 2 * shift, upper - 2 * shift)
    sin_lemma = "sin_pos_upper" if signs.sin_positive else "sin_neg_lower"
    divide_lemma = {(True, True): "divide_pos_pos", (True, False): "divide_pos_neg",
                    (False, False): "divide_neg_neg",
                    (False, True): "divide_neg_pos"}[(signs.sin_positive, signs.cos_positive)]
    bound_names = " ".join(signs.bound_names)

    def relation(function, term, positive):
        return f"0 < {function} {term}" if positive else f"{function} {term} < 0"

    cs_relation = relation("cos", angle_term, signs.cos_positive)
    sn_relation = relation("sin", angle_term, signs.sin_positive)
    tan_positive = signs.sin_positive == signs.cos_positive
    ts_relation = f"0 < tan {angle_term}" if tan_positive else f"tan {angle_term} < 0"
    tail = (
        TrigAuxiliaryFact("nz", f"cos {angle_term} \\<noteq> 0", "using cs by simp"),
        TrigAuxiliaryFact("ts", ts_relation,
                          f"using cs sn by (auto simp: tan_def intro: {divide_lemma})"),
    )
    if shift == 0:
        return (
            TrigAuxiliaryFact("cs", cs_relation,
                              f"using {bound_names} pi_gt_zero by (intro {cos_lemma}; linarith)"),
            TrigAuxiliaryFact("sn", sn_relation,
                              f"using {bound_names} pi_gt_zero by (intro {sin_lemma}; linarith)"),
        ) + tail
    shifted_term = angle_term
    for _ in range(abs(shift)):
        shifted_term = (f"({shifted_term} - (2::real) * pi)" if shift > 0
                        else f"({shifted_term} + (2::real) * pi)")
    transfer = "by (simp add: cos_diff cos_add sin_diff sin_add)"
    return (
        TrigAuxiliaryFact("csr", relation("cos", shifted_term, signs.cos_positive),
                          f"using {bound_names} pi_gt_zero by (intro {cos_lemma}; linarith)"),
        TrigAuxiliaryFact("snr", relation("sin", shifted_term, signs.sin_positive),
                          f"using {bound_names} pi_gt_zero by (intro {sin_lemma}; linarith)"),
        TrigAuxiliaryFact("cs", cs_relation, f"using csr {transfer}"),
        TrigAuxiliaryFact("sn", sn_relation, f"using snr {transfer}"),
    ) + tail


def _quadrant_interval(context: state_classes.TrigContext, angle_key: str):
    lower = upper = None
    for bound in context.pi_bounds:
        if bound.angle_key != angle_key or not bound.strict:
            continue
        if bound.side == "lower":
            lower = bound.coefficient if lower is None else max(lower, bound.coefficient)
        else:
            upper = bound.coefficient if upper is None else min(upper, bound.coefficient)
    return lower, upper


# ---------------------------------------------------------------------------
# Attempt planner
# ---------------------------------------------------------------------------


def _sqrt_candidate_fact(name: str, radicand_term: str, root: Fraction) -> TrigAuxiliaryFact:
    """A Python-computed square root restated as a kernel-checked fact:
    real_sqrt_unique demands root^2 = radicand and 0 <= root, so a wrong candidate cannot survive."""
    return TrigAuxiliaryFact(
        name, f"sqrt {radicand_term} = {_fraction_term(root)}",
        "by (intro real_sqrt_unique; simp add: power2_eq_square field_simps)")


def _premises_for(names, name_to_term):
    ordered = tuple(dict.fromkeys(names))
    return tuple((name, name_to_term[name]) for name in ordered if name in name_to_term)


def _explicit_sign_fact(name: str, function: str, angle_term: str,
                        positive: bool, source_name: str) -> TrigAuxiliaryFact:
    """Normalize an explicit admitted sign premise into a local fact of the exact `0 < f a` / `f a < 0` shape the session meta lemmas unify against (the transpiled premise may carry the flipped `>` spelling)."""
    relation = (f"0 < {function} {angle_term}" if positive
                else f"{function} {angle_term} < 0")
    return TrigAuxiliaryFact(name, relation, f"using {source_name} by simp")


def _tan_value_attempts(context, signs_by_angle, name_to_term):
    """Guarded attempts deriving sin/cos values from an admitted tangent value plus a cosine sign, the generalization of the old quadrant route.
    The claim itself is free-form:
    the value facts land in scope and the final tactic closes whatever the claim combines them into."""
    attempts = []
    for value in context.values:
        if value.function != "tan":
            continue
        try:
            value_node = pyexpr.parse_expr(value.value_source)
        except pyexpr.PyExprError:
            continue
        angle = value.angle_term
        tan_name = value.source_names[0]
        dd_term = f"(({value.value_term})^2 + 1)"
        root = _hypotenuse_root(value_node)
        # Variant A: quadrant bounds provide the cosine sign
        # (full cs/sn/nz/ts chain, so the guard twin can refute a tangent premise whose sign contradicts its quadrant).
        quadrant = signs_by_angle.get(value.angle_key)
        if quadrant is not None:
            lower, upper = _quadrant_interval(context, value.angle_key)
            facts = list(_sign_fact_chain(angle, quadrant, lower, upper))
            facts.extend(_meta_value_facts(angle, value.value_term, dd_term,
                                           quadrant.cos_positive, tan_name))
            if root is not None:
                facts.append(_sqrt_candidate_fact("dr", dd_term, root))
            attempts.append(TrigProofAttempt(
                name="TAN_VALUE_FROM_QUADRANT",
                premises=_premises_for((tan_name,) + quadrant.bound_names, name_to_term),
                auxiliary_facts=tuple(facts),
                tactic=GUARDED_FINAL_TACTIC,
                relevance_terms=(angle,)))
        # Variant B: an explicit admitted cosine-sign premise for the same angle.
        # The premise is normalized into the local fact `cs`, which the meta lemmas and the nonzero fact both consume.
        for sign in context.signs:
            if sign.function != "cos" or sign.angle_key != value.angle_key:
                continue
            facts = [_explicit_sign_fact("cs", "cos", angle, sign.positive,
                                         sign.source_names[0]),
                     TrigAuxiliaryFact("nz", f"cos {angle} \\<noteq> 0",
                                       "using cs by simp")]
            facts.extend(_meta_value_facts(angle, value.value_term, dd_term,
                                           sign.positive, tan_name))
            if root is not None:
                facts.append(_sqrt_candidate_fact("dr", dd_term, root))
            attempts.append(TrigProofAttempt(
                name="TAN_VALUE_FROM_SIGN",
                premises=_premises_for((tan_name,) + sign.source_names, name_to_term),
                auxiliary_facts=tuple(facts),
                tactic=GUARDED_FINAL_TACTIC,
                relevance_terms=(angle,)))
            break
    return attempts


def _meta_value_facts(angle_term, tan_term, dd_term, cos_positive, tan_name):
    """The sin and cos value facts from the tangent value, via the session meta lemmas whose orientation follows the PROVED cosine sign fact `cs` (never the claimed value)."""
    if cos_positive:
        sin_meta, cos_meta = "sin_from_tan_cpos", "cos_from_tan_cpos"
        sin_value = f"({tan_term}) / sqrt {dd_term}"
        cos_value = f"1 / sqrt {dd_term}"
    else:
        sin_meta, cos_meta = "sin_from_tan_cneg", "cos_from_tan_cneg"
        sin_value = f"- ({tan_term}) / sqrt {dd_term}"
        cos_value = f"- 1 / sqrt {dd_term}"
    return (
        TrigAuxiliaryFact("ms", f"sin {angle_term} = {sin_value}",
                          f"by (rule {sin_meta}[OF cs {tan_name}])"),
        TrigAuxiliaryFact("mc", f"cos {angle_term} = {cos_value}",
                          f"by (rule {cos_meta}[OF cs {tan_name}])"),
    )


def _pythagorean_value_attempts(context, signs_by_angle, name_to_term):
    """Guarded attempts deriving the complementary sin/cos value from one admitted sin or cos value plus the other function's sign, via the session cos_from_sin_* / sin_from_cos_* lemmas.
    The sign fact comes from quadrant bounds (cs/sn chain) or from an explicit sign premise;
    Python contributes only the kernel-checked square-root candidate."""
    attempts = []
    for value in context.values:
        if value.function not in ("sin", "cos"):
            continue
        try:
            value_node = pyexpr.parse_expr(value.value_source)
        except pyexpr.PyExprError:
            continue
        other = "cos" if value.function == "sin" else "sin"
        angle = value.angle_term
        value_name = value.source_names[0]
        complement_term = f"(1 - ({value.value_term})^2)"
        root = _unit_complement_root(value_node)
        quadrant = signs_by_angle.get(value.angle_key)
        if quadrant is not None:
            lower, upper = _quadrant_interval(context, value.angle_key)
            facts = list(_sign_fact_chain(angle, quadrant, lower, upper))
            other_positive = (quadrant.cos_positive if other == "cos"
                              else quadrant.sin_positive)
            facts.append(_complement_meta_fact(other, angle, value.value_term,
                                               complement_term, other_positive,
                                               value_name,
                                               sign_fact="cs" if other == "cos" else "sn"))
            if root is not None:
                facts.append(_sqrt_candidate_fact("dr", complement_term, root))
            attempts.append(TrigProofAttempt(
                name=f"{other.upper()}_VALUE_FROM_QUADRANT",
                premises=_premises_for((value_name,) + quadrant.bound_names, name_to_term),
                auxiliary_facts=tuple(facts),
                tactic=GUARDED_FINAL_TACTIC,
                relevance_terms=(angle,)))
        for sign in context.signs:
            if sign.function != other or sign.angle_key != value.angle_key:
                continue
            facts = [_explicit_sign_fact("fs", other, angle, sign.positive,
                                         sign.source_names[0]),
                     _complement_meta_fact(other, angle, value.value_term,
                                           complement_term, sign.positive,
                                           value_name, sign_fact="fs")]
            if root is not None:
                facts.append(_sqrt_candidate_fact("dr", complement_term, root))
            attempts.append(TrigProofAttempt(
                name=f"{other.upper()}_VALUE_FROM_SIGN",
                premises=_premises_for((value_name,) + sign.source_names, name_to_term),
                auxiliary_facts=tuple(facts),
                tactic=GUARDED_FINAL_TACTIC,
                relevance_terms=(angle,)))
            break
    return attempts


def _complement_meta_fact(other, angle_term, value_term, complement_term,
                          other_positive, value_name, sign_fact):
    lemma = {("cos", True): "cos_from_sin_cpos", ("cos", False): "cos_from_sin_cneg",
             ("sin", True): "sin_from_cos_spos",
             ("sin", False): "sin_from_cos_sneg"}[(other, other_positive)]
    rendered = (f"sqrt {complement_term}" if other_positive
                else f"- sqrt {complement_term}")
    return TrigAuxiliaryFact(
        "mo", f"{other} {angle_term} = {rendered}",
        f"by (rule {lemma}[OF {sign_fact} {value_name}])")


def _tan_from_values_attempts(context, name_to_term):
    """Guarded attempts deriving the tangent from admitted sin AND cos values of one angle:
    the cosine value must be a nonzero closed rational (kernel-checked via simp) before tan_def clears the denominator."""
    attempts = []
    sin_values = {value.angle_key: value for value in context.values
                  if value.function == "sin"}
    for value in context.values:
        if value.function != "cos":
            continue
        sin_value = sin_values.get(value.angle_key)
        if sin_value is None:
            continue
        cos_rational = _exact_value_of_source(value.value_source)
        if cos_rational is None or cos_rational == 0:
            continue
        angle = value.angle_term
        facts = (
            TrigAuxiliaryFact("nz", f"cos {angle} \\<noteq> 0",
                              f"using {value.source_names[0]} by simp"),
            TrigAuxiliaryFact("mt",
                              f"tan {angle} = ({sin_value.value_term}) / ({value.value_term})",
                              f"using {sin_value.source_names[0]} {value.source_names[0]} nz "
                              "by (simp add: tan_def)"),
        )
        attempts.append(TrigProofAttempt(
            name="TAN_FROM_VALUES",
            premises=_premises_for(sin_value.source_names + value.source_names, name_to_term),
            auxiliary_facts=facts,
            tactic=GUARDED_FINAL_TACTIC,
            relevance_terms=(angle,)))
    return attempts


def _exact_value_of_source(source: str) -> Fraction | None:
    try:
        return pyexpr._const_eval(pyexpr.parse_expr(source))
    except pyexpr.PyExprError:
        return None


def _sqrt_multiple(node: ast.AST):
    """(c, n) with the node's exact value c * sqrt(n) for rational c and a rational radicand n >= 0 (rationals themselves come back as (v, 1));
    None when the node is not of that shape.
    Handles sqrt(n), -x, x/d, c*x, x*c, so sqrt(2)/2, 3*sqrt(3), and 1/sqrt(3) all normalize."""
    value = pyexpr._const_eval(node)
    if value is not None:
        return value, Fraction(1)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt" and len(node.args) == 1):
        radicand = pyexpr._const_eval(node.args[0])
        if radicand is None or radicand < 0:
            return None
        return Fraction(1), radicand
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _sqrt_multiple(node.operand)
        if inner is None:
            return None
        c, n = inner
        return (c, n) if isinstance(node.op, ast.UAdd) else (-c, n)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        left, right = _sqrt_multiple(node.left), _sqrt_multiple(node.right)
        if left is None or right is None:
            return None
        (cl, nl), (cr, nr) = left, right
        if isinstance(node.op, ast.Mult):
            if nl != 1 and nr != 1:
                return None                    # sqrt * sqrt leaves the c*sqrt(n) shape
            return cl * cr, nl if nl != 1 else nr
        if cr == 0:
            return None
        if nr == 1:
            return cl / cr, nl
        if nl == 1 and nr != 0:
            # x / (cr * sqrt(nr)) = (x / (cr * nr)) * sqrt(nr)
            return cl / (cr * nr), nr
        return None
    return None


def _quadratic_parts(node: ast.AST):
    """(p, q, n) with the node's exact value p + q*sqrt(n) for rationals p, q;
    None when the node is not a sum/difference of rational and sqrt-multiple parts over ONE radicand."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _quadratic_parts(node.operand)
        if inner is None:
            return None
        p, q, n = inner
        return (p, q, n) if isinstance(node.op, ast.UAdd) else (-p, -q, n)
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))):
        return None
    sides = []
    for part in (node.left, node.right):
        multiple = _sqrt_multiple(part)
        if multiple is not None:
            c, n = multiple
            sides.append((c, Fraction(0), Fraction(1)) if n == 1
                         else (Fraction(0), c, n))
            continue
        sub = _quadratic_parts(part)
        if sub is None:
            return None
        sides.append(sub)
    (p1, q1, n1), (p2, q2, n2) = sides
    if isinstance(node.op, ast.Sub):
        p2, q2 = -p2, -q2
    if q1 != 0 and q2 != 0 and n1 != n2:
        return None
    n = n1 if q1 != 0 else n2
    return p1 + p2, q1 + q2, (n if (q1 != 0 or q2 != 0) else Fraction(1))


def _tan_double_value_term(value_node: ast.AST, value_term: str):
    """(rendered doubled-tangent value 2t/(1-t^2), sq-fact info or None), or None when the value is not exactly expressible.
    Rational t gives a plain fraction;
    t = c*sqrt(n) has rational t^2, so the value is a rational coefficient times the premise's own rendered term;
    a full quadratic irrational t = p + q*sqrt(n) is divided exactly in Q(sqrt n) through the conjugate,
    and the mv proof then needs the kernel-checked fact sqrt(n)*sqrt(n) = n (returned as sq info).
    t^2 = 1 and a vanishing conjugate denominator yield None (singular)."""
    t = _exact_value(value_node)
    if t is not None:
        if t * t == 1:
            return None
        return _fraction_term(2 * t / (1 - t * t)), None, None
    parts = _sqrt_multiple(value_node)
    if parts is not None:
        c, n = parts
        t_squared = c * c * n
        if t_squared == 1:
            return None
        return f"({_fraction_term(2 / (1 - t_squared))} * ({value_term}))", None, None
    quad = _quadratic_parts(value_node)
    if quad is None:
        return None
    p, q, n = quad
    if q == 0 or n.denominator != 1 or n <= 1:
        return None
    rational_square, sqrt_square = p * p + q * q * n, 2 * p * q
    a_part, b_part = 1 - rational_square, -sqrt_square
    conjugate_norm = a_part * a_part - b_part * b_part * n
    if conjugate_norm == 0 or (a_part == 0 and b_part == 0):
        return None
    # the denominator-nonzero fact dz proves by simp only when 1 - t^2 = A + B*sqrt(n) has A and B on ONE side of zero (then sqrt(n) >= 0 settles the sign);
    # a mixed-sign denominator would need the conjugate-norm argument in the kernel, so skip (FN only)
    if a_part != 0 and b_part != 0 and (a_part > 0) != (b_part > 0):
        return None
    p_out = 2 * (p * a_part - q * b_part * n) / conjugate_norm
    q_out = 2 * (q * a_part - p * b_part) / conjugate_norm
    sqrt_term = f"(sqrt (({n.numerator}::real)))"
    return (_qsqrt_term(p_out, q_out, sqrt_term), (sqrt_term, n.numerator),
            (_qsqrt_term(a_part, b_part, sqrt_term), a_part < 0 or b_part < 0))


def _qsqrt_term(rational: Fraction, coefficient: Fraction, sqrt_term: str) -> str:
    """Rendering of `rational + coefficient * sqrt_term` with zero parts elided."""
    if coefficient == 0:
        return _fraction_term(rational)
    if rational == 0:
        return f"({_fraction_term(coefficient)} * {sqrt_term})"
    return f"({_fraction_term(rational)} + {_fraction_term(coefficient)} * {sqrt_term})"


def literal_condition_theorem(angle_term: str, decomp_term: str, goal: str) -> str:
    """Premise-free theorem proving `goal` (a tangent definedness condition over a literal pi-multiple angle, e.g. cos(5*pi/4) != 0) through the kernel-checked decomposition:
    the same unfolding + staged-simp shape as the LITERAL_ANGLE attempt, because one plain simp round reorients the decomposition equality and re-collapses the decomposed form (measured).
    No assumptions, so no guard twin is needed."""
    return ("theorem chk:\n"
            f'  shows "{goal}"\n  proof -\n'
            f'  have e0: "{angle_term} = {decomp_term}" by (linarith | simp add: field_simps)\n'
            "  show ?thesis unfolding e0 by (simp only: sin_add sin_diff cos_add cos_diff, "
            "simp add: sin_30 cos_30 sin_45 cos_45 sin_60 cos_60)\nqed")


# Inverse-trig literal values over the PRINCIPAL ranges, keyed by (function, (c, n)) with the argument value c*sqrt(n):
# the angle the composition rewrite returns and the simp facts that prove the value equality `<arg> = <fn>(<angle>)`.
# Only angles inside arcsin_sin / arccos_cos / arctan_tan's ranges appear, so the range side conditions discharge;
# the negative arccos side (arccos(-1/2) = 2*pi/3) proves its value equality through a decomposition-backed reflection (the decomp column below).
_HALF = Fraction(1, 2)

# The ONE registry of arctan special angles,
# shared by the literal direction (arctan(sqrt(3)) = pi/3, rows of _INVERSE_LITERAL_TABLE)
# and the premise direction (`arctan(x) == pi/3` pins x = sqrt(3), _ARCTAN_PREMISE_TABLE),
# so the two directions can never drift apart.
# Per positive pi-coefficient:
# the rendered principal angle, the value key (c, n) for c*sqrt(n), the literal direction's value-equality lemmas,
# the premise direction's exact right-hand side, and the premise direction's lemmas.
# The two lemma sets differ where the spelling does:
# the literal direction states 1/sqrt(3) via real_sqrt_divide,
# while the premise direction states the rationalized sqrt(3)/3 (the final tactic cannot bridge the two spellings, measured) and proves it through the tan_def expansion.
_ARCTAN_SPECIAL_ANGLES = {
    Fraction(0): ("0", (Fraction(0), Fraction(1)), "tan_zero", "0", "tan_zero"),
    Fraction(1, 6): ("(pi / (6::real))", (Fraction(1, 3), Fraction(3)),
                     "tan_30 real_sqrt_divide", "sqrt 3 / 3",
                     "tan_def sin_30 cos_30 real_sqrt_divide field_simps "
                     "real_sqrt_pow2 power2_eq_square"),
    Fraction(1, 4): ("(pi / (4::real))", (Fraction(1), Fraction(1)), "tan_45", "1",
                     "tan_45"),
    Fraction(1, 3): ("(pi / (3::real))", (Fraction(1), Fraction(3)), "tan_60", "sqrt 3",
                     "tan_60"),
}

# value key -> (principal angle, value-equality simp lemmas, angle decomposition or None);
# a decomposition marks arccos's negative side, where the value equality itself needs the reflection (-1/2 = cos(2*pi/3) proves as cos(pi - pi/3) through the staged expansion).
# The arctan rows come from the shared registry above.
_INVERSE_LITERAL_TABLE = {
    ("arcsin", (Fraction(0), Fraction(1))): ("0", "sin_zero", None),
    ("arcsin", (_HALF, Fraction(1))): ("(pi / (6::real))", "sin_30", None),
    ("arcsin", (_HALF, Fraction(2))): ("(pi / (4::real))", "sin_45", None),
    ("arcsin", (_HALF, Fraction(3))): ("(pi / (3::real))", "sin_60", None),
    ("arcsin", (Fraction(1), Fraction(1))): ("(pi / (2::real))", "sin_pi_half", None),
    ("arccos", (Fraction(1), Fraction(1))): ("0", "cos_zero", None),
    ("arccos", (_HALF, Fraction(3))): ("(pi / (6::real))", "cos_30", None),
    ("arccos", (_HALF, Fraction(2))): ("(pi / (4::real))", "cos_45", None),
    ("arccos", (_HALF, Fraction(1))): ("(pi / (3::real))", "cos_60", None),
    ("arccos", (Fraction(0), Fraction(1))): ("(pi / (2::real))", "cos_pi_half", None),
    ("arccos", (-_HALF, Fraction(1))): ("((2::real) * pi / (3::real))", "cos_60",
                                        "(pi - (pi / (3::real)))"),
    ("arccos", (-_HALF, Fraction(2))): ("((3::real) * pi / (4::real))", "cos_45",
                                        "(pi - (pi / (4::real)))"),
    ("arccos", (-_HALF, Fraction(3))): ("((5::real) * pi / (6::real))", "cos_30",
                                        "(pi - (pi / (6::real)))"),
    ("arccos", (Fraction(-1), Fraction(1))): ("pi", "cos_pi", None),
    **{("arctan", value_key): (angle, literal_lemmas, None)
       for angle, value_key, literal_lemmas, _rhs, _premise_lemmas
       in _ARCTAN_SPECIAL_ANGLES.values()},
}
# odd functions: negative arguments map to the negated principal angle (still in range)
_ODD_INVERSE = frozenset({"arcsin", "arctan"})


# Per inverse function:
# the inner function, the range-conditional composition lemma, and its two range bound forms (arcsin_sin/arccos_cos take non-strict bounds, arctan_tan strict).
_INVERSE_COMPOSITION = {
    "arcsin": ("sin", "arcsin_sin", "- (pi / 2) \\<le> {angle}", "{angle} \\<le> pi / 2"),
    "arccos": ("cos", "arccos_cos", "0 \\<le> {angle}", "{angle} \\<le> pi"),
    "arctan": ("tan", "arctan_tan", "- (pi / 2) < {angle}", "{angle} < pi / 2"),
}


def _inverse_literal_attempts(claim_node):
    """One aux-only attempt per claim resolving inverse-trig LITERAL values (arcsin(1/2) = pi/6):
    the value equality e (`<arg> = sin(pi/6)`, kernel-checked from the special-value lemmas) is applied through `unfolding`,
    and the composition identity m comes from DIRECT rule instantiation `{lemma}[OF ba bb]` over the range bounds proved as explicit linarith facts;
    measured on the real pool: simp's conditional-rewrite solver discharges those bounds NEITHER from pi_gt_zero in the simpset NOR from chained facts, so the OF form (the session meta-lemma pattern) is the only working shape.
    Assumes nothing, so premise-free-classified like LITERAL_ANGLE."""
    if claim_node is None:
        return []
    facts, unfold_names, seen = [], [], set()
    for part in ast.walk(claim_node):
        if not (isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                and part.func.id in _INVERSE_COMPOSITION and len(part.args) == 1):
            continue
        function = part.func.id
        argument = part.args[0]
        key = (function, _angle_key(argument))
        if key in seen:
            continue
        seen.add(key)
        parts = _sqrt_multiple(argument)
        if parts is None:
            continue
        c, n = parts
        negated = c < 0 and function in _ODD_INVERSE
        entry = _INVERSE_LITERAL_TABLE.get((function, (-c if negated else c, n)))
        if entry is None:
            continue
        angle, lemmas, angle_decomp = entry
        if negated:
            angle = f"(- {angle})"
        argument_term = _transpile(argument, {})
        if argument_term is None:
            continue
        inner, composition_lemma, lower_form, upper_form = _INVERSE_COMPOSITION[function]
        k = len(unfold_names)
        if angle_decomp is None:
            value_proof = f"by (simp add: {lemmas})"
        else:
            facts.append(TrigAuxiliaryFact(
                f"d{k}", f"{angle} = {angle_decomp}",
                "by (linarith | simp add: field_simps)"))
            value_proof = (f"unfolding d{k} by (simp only: sin_add sin_diff cos_add "
                           f"cos_diff, simp add: {lemmas})")
        facts.extend((
            TrigAuxiliaryFact(f"e{k}", f"{argument_term} = {inner} {angle}", value_proof),
            TrigAuxiliaryFact(f"ba{k}", lower_form.format(angle=angle),
                              "using pi_gt_zero by linarith"),
            TrigAuxiliaryFact(f"bb{k}", upper_form.format(angle=angle),
                              "using pi_gt_zero by linarith"),
            TrigAuxiliaryFact(f"m{k}", f"{function} ({inner} {angle}) = {angle}",
                              f"by (rule {composition_lemma}[OF ba{k} bb{k}])"),
        ))
        unfold_names.append(f"e{k}")
    if not facts:
        return []
    return [TrigProofAttempt(
        name="INVERSE_LITERAL", premises=(), auxiliary_facts=tuple(facts),
        tactic="(simp)", unfold_fact_names=tuple(unfold_names))]


def _tan_double_attempts(context, signs_by_angle, name_to_term):
    """Guarded attempts for double-angle tangent values:
    an admitted tangent value with exactly expressible square (rational t, or t = c*sqrt(n) whose square c^2*n is rational),
    whose angle has principal-quadrant bounds AND whose DOUBLED interval also fits one principal quadrant,
    yields kernel-proved cos-nonzero facts for the angle (nz, from the cs chain) and for its double (nzd, from the doubled interval's own sign lemma);
    those discharge tan_double's hypotheses (mt),
    and the Python-computed doubled value 2t/(1-t^2), derived from the PREMISE value t (never from the claim), is restated as the kernel-checked fact mv,
    so the final linarith closes the claim by chaining mt with mv.
    mv is separate from mt because field_simps inside one simp round restructures the tan_double equation itself before it can rewrite the goal (measured on the real pool).
    Without the doubled-quadrant fit, with t^2 = 1 (the singular denominator), or with an inexpressible value no attempt is emitted
    (the tangent definedness check then fails the claim closed, never unsoundly open)."""
    attempts = []
    for value in context.values:
        if value.function != "tan":
            continue
        quadrant = signs_by_angle.get(value.angle_key)
        if quadrant is None or quadrant.shift != 0:
            continue
        lower, upper = _quadrant_interval(context, value.angle_key)
        if lower is None or upper is None:
            continue
        doubled_lemma = _cos_sign_lemma(2 * lower, 2 * upper)
        if doubled_lemma is None:
            continue
        try:
            value_node = pyexpr.parse_expr(value.value_source)
        except pyexpr.PyExprError:
            continue
        doubled = _tan_double_value_term(value_node, value.value_term)
        if doubled is None:
            continue
        doubled_value, sq_info, den_info = doubled
        angle = value.angle_term
        doubled = f"((2::real) * {angle})"
        doubled_expr = f"2 * tan {angle} / (1 - (tan {angle})^2)"
        doubled_cos_positive = doubled_lemma in ("cos_pos_q1", "cos_pos_q4")
        csd_relation = (f"0 < cos {doubled}" if doubled_cos_positive
                        else f"cos {doubled} < 0")
        bound_names = " ".join(quadrant.bound_names)
        mv_simp = "power2_eq_square real_sqrt_pow2 field_simps"
        mv_only = value.source_names[0]
        facts = list(_sign_fact_chain(angle, quadrant, lower, upper))
        if sq_info is not None:
            sqrt_term, radicand = sq_info
            facts.append(TrigAuxiliaryFact(
                "sq", f"{sqrt_term} * {sqrt_term} = ({radicand}::real)",
                "by (metis real_sqrt_pow2 power2_eq_square zero_le_numeral)"))
            mv_simp = "sq power2_eq_square field_simps"
        mv_proof = f"by (simp only: {mv_only}, simp add: {mv_simp})"
        if den_info is not None:
            den_rhs, den_negative = den_info
            # No simp round ever clears this division
            # (measured: the nonzero side condition resolves inside a NESTED conditional-rewrite context where neither simpset facts nor chained premises are visible).
            # So avoid division solving:
            # dn evaluates the denominator,
            # dp/ds/dz establish its nonzero sign through the proven linarith-then-simp chain,
            # mp proves the PRODUCT identity (pure ring, no division),
            # metis with nonzero_divide_eq_eq converts it to the division form mw,
            # and mv is three deterministic simp-only rewrites.
            facts.extend((
                TrigAuxiliaryFact(
                    "dn", f"(1::real) - ({value.value_term})^2 = {den_rhs}",
                    "by (simp add: sq power2_eq_square field_simps)"),
                TrigAuxiliaryFact("dp", f"0 \\<le> {sqrt_term}", "by simp"),
                TrigAuxiliaryFact("ds",
                                  (f"{den_rhs} < 0" if den_negative
                                   else f"0 < {den_rhs}"),
                                  "using dp by linarith"),
                TrigAuxiliaryFact("dz", f"{den_rhs} \\<noteq> 0", "using ds by simp"),
                TrigAuxiliaryFact(
                    "mp", f"({doubled_value}) * {den_rhs} = 2 * {value.value_term}",
                    "by (simp add: sq power2_eq_square field_simps)"),
                TrigAuxiliaryFact(
                    "mw", f"2 * {value.value_term} / {den_rhs} = {doubled_value}",
                    "using mp dz by (metis nonzero_divide_eq_eq)"),
            ))
            mv_proof = f"by (simp only: {value.source_names[0]} dn mw)"
        facts.extend((
            TrigAuxiliaryFact("csd", csd_relation,
                              f"using {bound_names} pi_gt_zero "
                              f"by (intro {doubled_lemma}; linarith)"),
            TrigAuxiliaryFact("nzd", f"cos {doubled} \\<noteq> 0", "using csd by simp"),
            TrigAuxiliaryFact("mt", f"tan {doubled} = {doubled_expr}",
                              "using nz nzd by (simp add: tan_double)"),
            # simp only substitutes the tangent value FIRST;
            # handing the premise to one field_simps round instead lets it restructure `tan a = 3/4` into `tan a * 4 = 3`, destroying the rewrite (measured on the real pool).
            # real_sqrt_pow2 evaluates (sqrt n)^2 for the sqrt-valued tangents.
            TrigAuxiliaryFact("mv", f"{doubled_expr} = {doubled_value}", mv_proof),
        ))
        attempts.append(TrigProofAttempt(
            name="TAN_DOUBLE_FROM_QUADRANT",
            premises=_premises_for((value.source_names[0],) + quadrant.bound_names,
                                   name_to_term),
            auxiliary_facts=tuple(facts),
            tactic=GUARDED_FINAL_TACTIC,
            relevance_terms=(angle,)))
    return attempts


_INVERSE_TRIG_SIMP_SET = ("sin_arcsin cos_arccos tan_arctan "
                         "cos_arcsin sin_arccos arcsin_sin arccos_cos arctan_tan")


def _inverse_trig_attempts(goal_term: str, name_to_term):
    """One guarded attempt applying the conditional inverse-composition rewrites with every admitted premise in scope:
    each rewrite's range side condition (e.g. -1 <= y <= 1 for sin_arcsin, -pi/2 <= x <= pi/2 for arcsin_sin) must be discharged from those premises or the rewrite never fires,
    and the equal-strength guard voids a success that leans on contradictory premises."""
    if not any(fn in goal_term for fn in ("arcsin", "arccos", "arctan")):
        return []
    if not name_to_term:
        return []
    return [TrigProofAttempt(
        name="INVERSE_TRIG_WITH_PREMISES",
        premises=tuple(sorted(name_to_term.items())),
        auxiliary_facts=(),
        tactic=f"(auto simp add: {_INVERSE_TRIG_SIMP_SET})")]


def _inverse_composition_value_attempts(claim_node):
    """Aux-only attempts evaluating inverse-trig compositions to NUMBERS (cos(arcsin(3/5)) = 4/5, sin(arctan(3/4)) = 3/5):
    the composition lemma gives the closed sqrt form by direct instantiation (cos_arcsin/sin_arccos over their [-1,1] bounds via OF; cos_arctan/sin_arctan are unconditional),
    and the Python-computed rational root is restated through the real_sqrt_unique-checked candidate fact, so a wrong root cannot survive.
    Rational arguments with rational roots only; anything else emits nothing."""
    if claim_node is None:
        return []
    facts, unfold_names, seen = [], [], set()
    for part in ast.walk(claim_node):
        if not (isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                and part.func.id in ("sin", "cos") and len(part.args) == 1):
            continue
        inner_call = part.args[0]
        if not (isinstance(inner_call, ast.Call) and isinstance(inner_call.func, ast.Name)
                and inner_call.func.id in ("arcsin", "arccos", "arctan")
                and len(inner_call.args) == 1):
            continue
        outer, inverse = part.func.id, inner_call.func.id
        argument = inner_call.args[0]
        v = _exact_value(argument)
        if v is None:
            continue
        key = (outer, inverse, _angle_key(argument))
        if key in seen:
            continue
        seen.add(key)
        v_term = _transpile(argument, {})
        if v_term is None:
            continue
        k = len(unfold_names)
        if (outer, inverse) in (("cos", "arcsin"), ("sin", "arccos")):
            if abs(v) > 1:
                continue
            root = _unit_complement_root(argument)
            if root is None:
                continue
            radicand = f"(1 - ({v_term})^2)"
            lemma = "cos_arcsin" if outer == "cos" else "sin_arccos"
            facts.extend((
                TrigAuxiliaryFact(f"ca{k}", f"- 1 \\<le> {v_term}", "by simp"),
                TrigAuxiliaryFact(f"cb{k}", f"{v_term} \\<le> 1", "by simp"),
                TrigAuxiliaryFact(f"w{k}",
                                  f"{outer} ({inverse} {v_term}) = sqrt {radicand}",
                                  f"by (rule {lemma}[OF ca{k} cb{k}])"),
                _sqrt_candidate_fact(f"wr{k}", radicand, root),
            ))
        elif inverse == "arctan":
            root = _rational_sqrt(1 + v * v)
            if root is None:
                continue
            radicand = f"(1 + ({v_term})^2)"
            lemma = "cos_arctan" if outer == "cos" else "sin_arctan"
            rhs = (f"1 / sqrt {radicand}" if outer == "cos"
                   else f"({v_term}) / sqrt {radicand}")
            facts.extend((
                TrigAuxiliaryFact(f"w{k}", f"{outer} ({inverse} {v_term}) = {rhs}",
                                  f"by (rule {lemma})"),
                _sqrt_candidate_fact(f"wr{k}", radicand, root),
            ))
        else:
            continue
        # wr goes through `unfolding` too:
        # handed to simp as a fact instead, the field rules restructure `sqrt X = 5/4` into `sqrt X * 4 = 5` and the rewrite dies (the same measured trap as the tan_double mv fact)
        unfold_names.extend((f"w{k}", f"wr{k}"))
    if not facts:
        return []
    return [TrigProofAttempt(
        name="INVERSE_COMPOSITION_VALUE", premises=(), auxiliary_facts=tuple(facts),
        tactic="(simp)", unfold_fact_names=tuple(unfold_names))]


# angle coefficient (of pi, positive side) -> (the tangent value as the exact right-hand side, the proof lemmas),
# the premise direction of the shared _ARCTAN_SPECIAL_ANGLES registry.
# Negative angles ride on tan_minus [simp] plus the same lemmas.
_ARCTAN_PREMISE_TABLE = {
    coefficient: (rhs, premise_lemmas)
    for coefficient, (_angle, _value_key, _literal_lemmas, rhs, premise_lemmas)
    in _ARCTAN_SPECIAL_ANGLES.items()}


def _arctan_premise_value_attempts(context, name_to_term):
    """Guarded attempts turning an admitted `arctan(x) == <table angle>` premise into the exact value of x:
    tan_arctan is UNCONDITIONAL in HOL (tan(arctan v) = v for every real v), so mx (`x = tan(angle)`) follows by metis with no range assumption,
    mt evaluates the tangent through the stock value lemma,
    and mv chains them for the final tactic.
    arcsin/arccos premises pin nothing in HOL without range information (outside [-1,1] the inverse is an unspecified value), so they stay unsupported (fail closed)."""
    attempts = []
    for value in context.values:
        if value.function != "arctan":
            continue
        try:
            angle_node = pyexpr.parse_expr(value.value_source)
        except pyexpr.PyExprError:
            continue
        coefficient = exact_pi_coefficient(angle_node)
        if coefficient is None:
            continue
        negated = coefficient < 0
        entry = _ARCTAN_PREMISE_TABLE.get(-coefficient if negated else coefficient)
        if entry is None:
            continue
        rhs, lemmas = entry
        tan_value = f"- ({rhs})" if negated else rhs
        facts = (
            TrigAuxiliaryFact("mt", f"tan {value.value_term} = {tan_value}",
                              f"by (simp add: {lemmas})"),
            TrigAuxiliaryFact("mx", f"{value.angle_term} = tan {value.value_term}",
                              f"using {value.source_names[0]} by (metis tan_arctan)"),
            TrigAuxiliaryFact("mv", f"{value.angle_term} = {tan_value}",
                              "using mx mt by linarith"),
        )
        attempts.append(TrigProofAttempt(
            name="ARCTAN_PREMISE_VALUE",
            premises=_premises_for(value.source_names, name_to_term),
            auxiliary_facts=facts,
            tactic="((linarith) | (simp add: real_sqrt_divide field_simps "
                   "real_sqrt_pow2 power2_eq_square; fail))",
            # the pinned variable's own term: the closable claims are about x itself
            relevance_terms=(value.angle_term,)))
    return attempts


def _pi_term(coefficient: Fraction) -> str:
    """Isabelle rendering of a POSITIVE rational multiple of pi (0 and 1 special-cased)."""
    if coefficient == 0:
        return "0"
    if coefficient == 1:
        return "pi"
    num, den = coefficient.numerator, coefficient.denominator
    if den == 1:
        return f"(({num}::real) * pi)"
    if num == 1:
        return f"(pi / ({den}::real))"
    return f"(({num}::real) * pi / ({den}::real))"


def _decomposed_pi_term(coefficient: Fraction):
    """The canonical decomposition of `coefficient * pi` into full periods plus a reflection off a base angle, or None beyond the period cap:
    m = coefficient mod 2 picks the in-period residue,
    the quarter of m picks pi - x / pi + x / 2*pi - x,
    and each whole 2*pi period becomes ONE NESTED `2*pi + (...)` / `(...) - 2*pi` layer.
    Nesting matters:
    the default simpset evaluates sin/cos only at 2*pi itself, so a flat `4*pi + x` would strand the expansion, while each nested layer folds by one sin_add/cos_diff application.
    The equality `angle = decomposition` is only a CANDIDATE;
    it is restated as a kernel-checked local fact, so a wrong decomposition cannot survive."""
    residue = coefficient % 2
    periods = int((coefficient - residue) / 2)
    if abs(periods) > _MAX_PERIOD_SHIFTS:
        return None
    if residue <= Fraction(1, 2):
        core = _pi_term(residue)
    elif residue <= 1:
        core = f"(pi - {_pi_term(1 - residue)})"
    elif residue <= Fraction(3, 2):
        core = f"(pi + {_pi_term(residue - 1)})"
    else:
        core = f"((2::real) * pi - {_pi_term(2 - residue)})"
    term = core
    for _ in range(abs(periods)):
        term = (f"((2::real) * pi + {term})" if periods > 0
                else f"({term} - (2::real) * pi)")
    return term


def _literal_angle_attempts(claim_node):
    """One attempt rewriting every literal non-base pi-multiple angle in the claim (sin(5*pi/6), cos(7*pi/4), sin(13*pi/6)) onto its periodicity/reflection decomposition, then closing with the expansion + special-value simp set.
    The decomposition equalities are pure linear pi arithmetic proved by the kernel;
    a literal angle contains no step variables, so transpilation needs no variable types."""
    if claim_node is None:
        return []
    facts, seen_keys = [], set()
    for part in ast.walk(claim_node):
        call = _trig_call(part)
        if call is None:
            continue
        _function, angle = call
        key = _angle_key(angle)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        coefficient = exact_pi_coefficient(angle)
        if coefficient is None or coefficient in _BASE_PI_COEFFS:
            continue
        decomposition = _decomposed_pi_term(coefficient)
        angle_term = _transpile(angle, {})
        if decomposition is None or angle_term is None:
            continue
        facts.append(TrigAuxiliaryFact(
            f"e{len(facts)}", f"{angle_term} = {decomposition}",
            "by (linarith | simp add: field_simps)"))
    if not facts:
        return []
    # Three directed stages, because one simp round cannot do this (measured on the real pool):
    # simp REORIENTS the decomposition equality (the decomposed side is larger)
    # and its arithmetic simproc re-collapses `pi - pi/6` back to the literal `5*pi/6`, so the expansion rules race the collapse and never finish.
    # The `unfolding` keyword rewrites the angle left-to-right at the meta level (no simprocs),
    # `simp only` expands with nothing else active,
    # and only then does the full simpset evaluate the base values.
    return [TrigProofAttempt(
        name="LITERAL_ANGLE", premises=(), auxiliary_facts=tuple(facts),
        tactic="(simp only: tan_def sin_add sin_diff cos_add cos_diff, "
               "simp add: sin_30 cos_30 sin_45 cos_45 sin_60 cos_60)",
        unfold_fact_names=tuple(fact.name for fact in facts))]


def _mentions_trig(text: str) -> bool:
    return any(fn in (text or "") for fn in TRIG_FUNCTIONS)


def _has_variable(node: ast.AST) -> bool:
    return any(isinstance(part, ast.Name) and part.id != "pi" for part in ast.walk(node))


def _bare_tan_argument(node: ast.AST):
    call = _trig_call(node)
    return call[1] if call is not None and call[0] == "tan" else None


def _negated_tan_argument(node: ast.AST):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _bare_tan_argument(node.operand)
    return None


def _linear_combination_is_pi_integer(u: ast.AST, v: ast.AST, add: bool) -> bool:
    """True when u - v (or u + v with add) is exactly k*pi for an integer k, decided over the exact linear forms;
    an inexpressible angle decides False (fail closed)."""
    fu, fv = _linear_form(u), _linear_form(v)
    if fu is None or fv is None:
        return False
    delta = dict(fu)
    for name, coefficient in fv.items():
        delta[name] = delta.get(name, Fraction(0)) + (coefficient if add else -coefficient)
    pi_coefficient = delta.pop("pi", Fraction(0))
    return (all(c == 0 for c in delta.values())
            and pi_coefficient.denominator == 1)


def _same_angle(u: ast.AST, v: ast.AST) -> bool:
    fu, fv = _linear_form(u), _linear_form(v)
    if fu is not None and fv is not None:
        return fu == fv
    return ast.dump(u) == ast.dump(v)


def _sin_over_cos_arguments(node: ast.AST):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        sin_call, cos_call = _trig_call(node.left), _trig_call(node.right)
        if (sin_call is not None and cos_call is not None
                and sin_call[0] == "sin" and cos_call[0] == "cos"):
            return sin_call[1], cos_call[1]
    return None


def _tan_identity_conjunct(conjunct: ast.AST) -> bool:
    """True when the conjunct is one of the allowed tangent identity REWRITE shapes over VARIABLE angles, matched structurally on the whole equation
    (never by the mere presence of some trig application on the other side:
    that earlier test also matched value pins like tan(a) == sin(b) and tan(a) == cos(a), which premises pinning a == pi/2 turn into HOL-provable totalized values, a measured FP door).
    The shapes:
    periodicity `tan(u) == tan(v)` with u - v an integer multiple of pi,
    odd/reflection `tan(u) == -tan(v)` with u + v an integer multiple of pi (covering tan(-a) == -tan(a) and tan(pi - a) == -tan(a)),
    and the definition unfold `tan(u) == sin(u)/cos(u)` over the SAME angle.
    These rewrite a tangent rather than pinning its value, so they carry no cosine-nonzero condition.
    Constant angles never qualify (tan(pi/2 + pi) == tan(pi/2) is HOL-provable as 0 == 0 but mathematically undefined);
    everything unmatched requires its conditions,
    including the tan_add/tan_double formula shapes, whose right side turns into a totalized 0 == 0 at a premise-pinned singular angle."""
    if not (isinstance(conjunct, ast.Compare) and len(conjunct.ops) == 1
            and isinstance(conjunct.ops[0], ast.Eq)):
        return False
    left, right = conjunct.left, conjunct.comparators[0]
    for x, y in ((left, right), (right, left)):
        u = _bare_tan_argument(x)
        if u is None or not _has_variable(u):
            continue
        v = _bare_tan_argument(y)
        if (v is not None and _has_variable(v)
                and _linear_combination_is_pi_integer(u, v, add=False)):
            return True
        v = _negated_tan_argument(y)
        if (v is not None and _has_variable(v)
                and _linear_combination_is_pi_integer(u, v, add=True)):
            return True
        quotient = _sin_over_cos_arguments(y)
        if (quotient is not None and _same_angle(u, quotient[0])
                and _same_angle(u, quotient[1])):
            return True
    return False


def _nonzero_constant_value(node: ast.AST) -> bool:
    """True when the node is a variable-free expression with a provably NONZERO exact value (a rational, c*sqrt(n), or p + q*sqrt(n));
    zero and inexpressible values are False."""
    parts = _sqrt_multiple(node)
    if parts is not None:
        c, n = parts
        return c != 0 and n != 0
    quad = _quadratic_parts(node)
    if quad is None:
        return False
    p, q, n = quad
    root = _rational_sqrt(n)
    if root is not None:
        return p + q * root != 0
    return p != 0 or q != 0


def _tan_nonzero_value_conjunct(conjunct: ast.AST) -> bool:
    """True for `tan(u) == <nonzero constant>` in either orientation:
    in HOL this equation itself FORCES cos(u) != 0 (cos u = 0 evaluates tan u to 0, contradicting the nonzero value),
    so as an ASSUMPTION it is self-certifying and needs no separate cosine-nonzero condition;
    without this, an inherited `tan(x) == 3/4` chain conclusion would demand a quadrant re-proof in every later step of the chain.
    Assumption-side only:
    a CLAIM of this shape keeps its condition, because a claim may be proved under decimal tolerance, where the totalized 0 can sit within tolerance of a small nonzero claimed value."""
    if not (isinstance(conjunct, ast.Compare) and len(conjunct.ops) == 1
            and isinstance(conjunct.ops[0], ast.Eq)):
        return False
    left, right = conjunct.left, conjunct.comparators[0]
    for x, y in ((left, right), (right, left)):
        if _bare_tan_argument(x) is not None and _nonzero_constant_value(y):
            return True
    return False


def tangent_definedness_conditions(claim_nodes, variable_types,
                                   assumption_nodes=()) -> tuple:
    """(angle term, decomposition term or None) for every VALUE-SENSITIVE tangent application the step brings into its theorem;
    verification builds and proves `cos <angle> != 0` for each BEFORE any claim attempt,
    and a non-base literal pi-multiple angle carries its kernel-checkable decomposition (cos(5*pi/4) != 0 has no premise-free route otherwise).
    HOL totalizes tan (the kernel evaluates tan(pi/2) to 0), and the FP surface is VALUE PINNING:
    a definite value entering the proof for a tangent at an undefined point.
    Scanned sources:
    the claim conjuncts, and through `assumption_nodes` every model-authored assumption the theorem admits,
    i.e. this step's definitions and admitted step premises plus every earlier chain conclusion
    (the measured bypasses otherwise: `x == tan(a)` splits into a definition, leaving the tan-free claim `x == 0` unconditioned,
    and a REJECTED `x == tan(pi/2)` step still enters the chain as s*, handing the next step the same pinned 0).
    Problem givens stay unscanned: the model does not author them (translation trust).
    Classification is per conjunct:
    an allowed identity rewrite carries no condition (_tan_identity_conjunct);
    an assumption-side `tan(u) == <nonzero constant>` is self-certifying (_tan_nonzero_value_conjunct);
    every OTHER conjunct containing a tangent requires the condition for each tangent application in it, constant or variable angle alike
    (constant also blocks disguised values like tan(pi/2) == sin(pi/2) * 0; comparison chains and non-comparison conjuncts always require).
    An untranspilable angle degrades to a None angle term, failing the step closed rather than silently waiving the check."""
    angles = []
    sources = [(node, False) for node in claim_nodes]
    sources.extend((node, True) for node in assumption_nodes)
    for node, assumption_side in sources:
        for conjunct in _conjuncts(node):
            if _tan_identity_conjunct(conjunct):
                continue
            if assumption_side and _tan_nonzero_value_conjunct(conjunct):
                continue
            for part in ast.walk(conjunct):
                call = _trig_call(part)
                if call is None or call[0] != "tan":
                    continue
                angle = call[1]
                angle_term = _transpile(angle, variable_types)
                decomp = None
                if angle_term is not None:
                    coefficient = exact_pi_coefficient(angle)
                    if coefficient is not None and coefficient not in _BASE_PI_COEFFS:
                        decomp = _decomposed_pi_term(coefficient)
                angles.append((angle_term, decomp))
    return tuple(dict.fromkeys(angles))


def plan_attempts(context, goal_term: str, name_to_term: dict, claim_node=None):
    """The bounded, deterministically ordered attempt list for one claim:
    premise-free identity and special-angle groups first (no guard needed, allowed even under unknown premise consistency),
    the aux-only literal-angle decomposition next (same premise-free status: it assumes nothing),
    then the guarded value families over admitted evidence.
    Before the MAX_ATTEMPTS cap, evidence attempts whose relevance_terms miss the goal drop behind those that hit it (stable within each class),
    so evidence about the claim's own angles survives the cap regardless of how many unrelated trig premises the step admitted
    (measured: 14 unrelated tangent premises previously pushed the one relevant family past the cap, making provability depend on premise order).
    Relevance compares free-identifier SETS (_goal_relevant), never substrings:
    variable names that are prefixes of one another (pv_x1 inside pv_x11) must not count as hits (measured to crowd the window the same way).
    Order and the cap are part of the verifier contract;
    local fact names are fixed per family so identical attempts share theorem-cache entries."""
    if not _mentions_trig(goal_term) and not any(
            value.function == "arctan" for value in context.values):
        return ()   # an arctan premise value can close trig-free claims about x itself
    attempts = [TrigProofAttempt(name=name, premises=(), auxiliary_facts=(), tactic=tactic)
                for name, tactic in PREMISE_FREE_GROUPS]
    attempts.extend(_literal_angle_attempts(claim_node))
    attempts.extend(_inverse_literal_attempts(claim_node))
    attempts.extend(_inverse_composition_value_attempts(claim_node))
    signs_by_angle = _quadrant_signs(context)
    attempts.extend(_tan_value_attempts(context, signs_by_angle, name_to_term))
    attempts.extend(_pythagorean_value_attempts(context, signs_by_angle, name_to_term))
    attempts.extend(_tan_from_values_attempts(context, name_to_term))
    attempts.extend(_tan_double_attempts(context, signs_by_angle, name_to_term))
    attempts.extend(_inverse_trig_attempts(goal_term, name_to_term))
    attempts.extend(_arctan_premise_value_attempts(context, name_to_term))
    seen, unique_attempts = set(), []
    for attempt in attempts:
        key = (attempt.premises, attempt.auxiliary_facts, attempt.tactic)
        if key in seen:
            continue
        seen.add(key)
        unique_attempts.append(attempt)
    goal_identifiers = theorem_builders.identifiers(goal_term)
    unique_attempts.sort(
        key=lambda attempt: not _goal_relevant(attempt, goal_identifiers))
    return tuple(unique_attempts[:MAX_ATTEMPTS])


def _goal_relevant(attempt: TrigProofAttempt, goal_identifiers: set) -> bool:
    """Whether the attempt concerns the goal, decided on free-identifier SETS:
    some relevance term's identifiers (theorem_builders.identifiers, which already drops the reserved Isabelle vocabulary and type annotations) must all occur among the goal's.
    Never a substring test: pv_x1 occurs inside pv_x11 as text, so prefix-colliding variable names would classify as hits and crowd the cap window exactly like the unrelated-premise case (measured).
    A term with no free identifiers (a literal angle) stays always-relevant, the safe over-inclusive direction."""
    if not attempt.relevance_terms:
        return True
    return any(theorem_builders.identifiers(term) <= goal_identifiers
               for term in attempt.relevance_terms)
