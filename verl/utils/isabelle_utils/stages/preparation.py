"""Prepare translated response steps for Isabelle verification.

A general PyExprStep becomes an IsabelleStep containing its typed terms, separated premise sources, guards, and consistency theorem. A DirectDomainStep keeps its statement and receives the earlier conclusions it may use. This stage creates no verification results and owns no prover pool.
"""
import ast
import dataclasses
import re
from fractions import Fraction

import verl.utils.isabelle_utils.number_provenance as number_provenance
import verl.utils.isabelle_utils.pyexpr as pyexpr
import verl.utils.isabelle_utils.state_classes as state_classes
import verl.utils.isabelle_utils.tactics as tactics
import verl.utils.isabelle_utils.theorem_builders as theorem_builders
import verl.utils.isabelle_utils.trigonometry as trigonometry
from verl.utils.isabelle_utils.stages.formalization import conjuncts

# Numerals inside a verbatim direct-domain claim, e.g. the 15 in `order G = 15`. They feed number provenance only (a number's admissibility for later general steps' premises), never a logical premise.
_DIRECT_CLAIM_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")

# The direct->general bridge whitelist: an arithmetic-valued direct claim of the shape `order G = 15`, `card (rcosets H) = 5`, `ord a = 6`. Only these cross the session boundary, as the nominal pyexpr equation over the token-joined name (order_G == 15; the translator prompt instructs later general steps to use exactly that joined name). Dropping nat nonnegativity only weakens the premise, so the bridge cannot make anything provable the claim did not assert.
_BRIDGE_FN_RE = re.compile(r"^\s*(order|card|ord)\b\s*(\(?[A-Za-z][A-Za-z0-9_ ()]*?\)?)\s*=\s*(\d+)\s*$")


def _bridge_direct_claim(claim):
    """Bridged pyexpr source for a whitelisted direct claim (`order G = 15` -> "order_G == 15"), else None."""
    m = _BRIDGE_FN_RE.match(claim or "")
    if m is None:
        return None
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", m.group(1) + " " + m.group(2))
    name = "_".join(tokens)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return None
    return "%s == %s" % (name, m.group(3))


def _node_carrier(node, pyexpr_variable_types: dict, fallback: str = "real") -> str:
    """Choose the numeral carrier for one typed expression subtree."""
    try:
        ids, _, needs_real = pyexpr.analyze(node)
    except pyexpr.PyExprError:
        return fallback
    if needs_real or any(pyexpr_variable_types.get(i) == "real" for i in ids):
        return "real"
    return "int"


def _transpile_conjunctive(src: str, pyexpr_variable_types: dict):
    """Transpile top-level conjunctions while choosing each conjunct's carrier.

    A proposition such as ``x / 2 == 1 and n % 2 == 0`` cannot use one global carrier: the first conjunct is real while the second must remain int.
    The combined term is used both as the current claim and as a later step's premise, so it must preserve both sorts when constructed.
    """
    node = pyexpr.parse_expr(src)
    nodes = (node.values if isinstance(node, ast.BoolOp)
             and isinstance(node.op, ast.And) else [node])
    terms = []
    carriers = []
    for part in nodes:
        carrier = _node_carrier(part, pyexpr_variable_types)
        terms.append(pyexpr.transpile(part, pyexpr_variable_types, carrier))
        carriers.append(carrier)
    term = "(" + " & ".join(terms) + ")" if len(terms) > 1 else terms[0]
    ids, consts, _ = pyexpr.analyze(node)
    carrier = carriers[0] if len(set(carriers)) == 1 else "mixed"
    return term, ids, consts, carrier


def _nonconstant_divisors(src: str, var_types: dict, carrier: str):
    """Transpile nonconstant Div, FloorDiv, and Mod right-hand sides."""
    out = []
    try:
        node = pyexpr.parse_expr(src)
    except pyexpr.PyExprError:
        return out
    for part in ast.walk(node):
        if (isinstance(part, ast.BinOp)
                and isinstance(part.op, (ast.Div, ast.FloorDiv, ast.Mod))
                and pyexpr._const_eval(part.right) is None):
            try:
                out.append(pyexpr.transpile(part.right, var_types, carrier))
            except pyexpr.PyExprError:
                pass
    return out


def prepare(formalized):
    """Convert one formalized response into its ordered verification inputs.

    A PyExprStep becomes an IsabelleStep. A DirectDomainStep receives earlier conclusions from its compatible direct family and from the general path, then otherwise passes through unchanged.

    Premises depend on translated givens and earlier translated conclusions, not on earlier proof results. The entire response can therefore be prepared before any prover call.
    """
    nl_steps, pyexpr_givens = formalized.nl_steps, formalized.pyexpr_givens
    pyexpr_variable_types_declared = formalized.pyexpr_variable_types_declared
    # General-path sources from the typed step list. A DirectDomainStep contributes nothing to type inference or function arities (its verbatim Isabelle is not pyexpr); its claim's numerals join number provenance in the step loop below.
    general_conclusions = [s.pyexpr_conclusion for s in formalized.steps
                           if isinstance(s, state_classes.PyExprStep)]
    general_premises = [q for s in formalized.steps
                        if isinstance(s, state_classes.PyExprStep) for q in s.pyexpr_premises]
    # Whitelisted direct claims bridged to pyexpr (the direct->general half of the cross-session bridge); they join type inference so the bridged name gets a declared sort.
    bridged_by_slot = {}
    for k, s in enumerate(formalized.steps):
        if isinstance(s, state_classes.DirectDomainStep):
            bridged_src = _bridge_direct_claim(s.claim)
            if bridged_src is not None:
                bridged_by_slot[k] = bridged_src
    bridged_sources = list(bridged_by_slot.values())
    pyexpr_variable_types = dict(pyexpr_variable_types_declared)
    # A step may introduce an intermediate name that was not declared in the givens, such as `current_tiles` or `subtotal`.
    # In an all-integer problem, defaulting that name to real can produce an invalid mixed-type equation such as `red::int == current_tiles::real - blue::int`.
    # Use int when every declared variable is int; otherwise use real. Fractional intermediates are normally declared explicitly by the translator and therefore do not use this default.
    _new_default = "int" if (pyexpr_variable_types_declared and all(t == "int" for t in pyexpr_variable_types_declared.values()))\
        else "real"
    for p in general_conclusions + general_premises + bridged_sources:
        try:
            _, ids, _, _ = pyexpr.py_to_isabelle(p, pyexpr_variable_types)
            for i in ids:
                pyexpr_variable_types.setdefault(i, _new_default)
        except pyexpr.PyExprError:
            pass
    # Variables used by pyexpr integer modulo or division must have type int.
    # For example, `n % 2 == 0` with `n::real` becomes `n mod (2::real)`, which Isabelle cannot evaluate, so a correct parity or divisibility step fails.
    # Override an earlier real inference for those variables; genuine real-valued uses can still refer to them through `real n`.
    for p in pyexpr_givens + general_conclusions + general_premises:
        try:
            for i in pyexpr.mod_int_vars(pyexpr.parse_expr(p)):
                if i in pyexpr_variable_types:
                    pyexpr_variable_types[i] = "int"
        except pyexpr.PyExprError:
            pass
    # Applied uninterpreted functions are declared as function-typed fixes rather than scalars. pyexpr.analyze() already keeps these names out of pyexpr_variable_types.
    func_ar = {}
    for p in pyexpr_givens + general_conclusions + general_premises:
        try:
            func_ar.update(pyexpr.func_arities(pyexpr.parse_expr(p)))
        except pyexpr.PyExprError:
            pass
    # Fix names carry the same pv_ prefix as the transpiled terms (see pyexpr._pv) so free variables never collide with an Isabelle constant.
    func_fixes = [(pyexpr._add_prefix_pv(name), '"' + " => ".join(["real"] * (ar + 1)) + '"')
                  for name, ar in sorted(func_ar.items())]
    isabelle_fixes = [(pyexpr._add_prefix_pv(k), v) for k, v in pyexpr_variable_types.items()] + func_fixes
    isabelle_step_conclusions, consts_per = [], []
    for k, s in enumerate(formalized.steps):
        if isinstance(s, state_classes.PyExprStep):
            term, ids, consts, carrier = _transpile_conjunctive(s.pyexpr_conclusion, pyexpr_variable_types)
            isabelle_step_conclusions.append((term, carrier))
            consts_per.append(consts)
        elif k in bridged_by_slot:
            # whitelisted direct claim: its bridged pyexpr form fills this step's s* slot, so later general steps receive it exactly like a general conclusion
            term, ids, consts, carrier = _transpile_conjunctive(bridged_by_slot[k], pyexpr_variable_types)
            isabelle_step_conclusions.append((term, carrier))
            consts_per.append(consts)
        else:
            # direct-domain slot with no bridgeable claim: no general-chain conclusion
            isabelle_step_conclusions.append(None)
            consts_per.append(set())
    # Choose each conjunct's carrier separately so `n % 2 == 0 and x > 3.5` keeps its modulo expression integer-typed. A non-conjunction has the same output shape as py_to_isabelle.
    isabelle_problem_conditions = [_transpile_conjunctive(g, pyexpr_variable_types)[0] for g in pyexpr_givens]
    given_nums, known = set(formalized.problem_nums), set()
    for g in pyexpr_givens:
        try:
            _, ids, cs, _ = pyexpr.py_to_isabelle(g, pyexpr_variable_types)
            given_nums |= cs
            known |= ids
        except pyexpr.PyExprError:
            pass
    isabelle_problem_premises = [(f"g{i}", t) for i, t in enumerate(isabelle_problem_conditions)]
    admitted_problem_sources = tuple(
        state_classes.AdmittedPyExprSource(
            source_kind=state_classes.PremiseSource.PROBLEM,
            name=name, pyexpr_source=source, isabelle_term=term)
        for source, (name, term) in zip(pyexpr_givens, isabelle_problem_premises))
    # Preserve each earlier general conclusion's admitted form-B source next to its form-C assumption name. Direct claims contribute only when the existing direct-to-general bridge produced a pyexpr source for that slot.
    conclusion_sources_by_slot = []
    for j, earlier_step in enumerate(formalized.steps):
        if isinstance(earlier_step, state_classes.PyExprStep):
            conclusion_sources_by_slot.append(earlier_step.pyexpr_conclusion)
        else:
            conclusion_sources_by_slot.append(bridged_by_slot.get(j))
    # Conjunct dumps of the problem GIVENs, used at two levels.
    # Per conjunct: a claim conjunct equal to a given is auto-satisfied in the claim check
    # (it is an assumption; sending it to the prover as a free-variable subgoal would mislabel a step that also proves something new,
    # e.g. "the LCM of 1 through 9 is the answer" alongside a real computation when the givens define answer == lcm(1,...,9)).
    # Per step: a step whose EVERY claim conjunct restates a given is claim_repeats_given,
    # true by assumption but contentless, never rewarded, pattern x
    # (the g*-side analog of claim_repeats_earlier; measured on the 2026-07-17 rollouts, 13.5% of rewarded steps were such restatements).
    given_dumps = set()
    for _g in pyexpr_givens:
        try:
            given_dumps.add(ast.dump(pyexpr.parse_expr(_g)))
        except pyexpr.PyExprError:
            pass

    # Prepare every step before submitting Isabelle checks.
    # Premises for a step depend only on translated givens and earlier translated conclusions, not on the verification results of earlier steps.
    # Premise-consistency checks and step-claim checks can therefore run concurrently.
    # After they finish, the first inconsistent or unknown premise set determines how later steps are scored.
    concl_nums = set()

    # sos triggers by nonlinearity
    givens_nonlin = False
    for _g in pyexpr_givens:
        try:
            if pyexpr.is_nonlinear(pyexpr.parse_expr(_g)):
                givens_nonlin = True
                break
        except pyexpr.PyExprError:
            pass

    # A linear premise set can use the faster complete decision procedures.
    # Nonlinear or unrecognized expressions continue to use ALTERNATION.
    def _src_linear(src):
        try:
            return pyexpr.is_linear_arith(pyexpr.parse_expr(src))
        except pyexpr.PyExprError:
            return False
    givens_linear = all(_src_linear(g) for g in pyexpr_givens)
    # A direct-domain slot counts as linear: a non-bridgeable slot contributes nothing to the general chain, and a bridged slot's equation (pv_name = numeral) is itself linear, so neither may push later steps off the complete linear decision procedures.
    conclusions_linear = [_src_linear(s.pyexpr_conclusion) if isinstance(s, state_classes.PyExprStep)
                          else True for s in formalized.steps]

    # Earlier conclusions join later steps' premises as s*, so a nonlinear one makes those steps sos-eligible exactly like a nonlinear own premise; a direct-domain slot bridges at most a linear numeral equation.
    def _src_nonlinear(src):
        try:
            return pyexpr.is_nonlinear(pyexpr.parse_expr(src))
        except pyexpr.PyExprError:
            return False
    conclusions_nonlinear = [
        _src_nonlinear(s.pyexpr_conclusion) if isinstance(s, state_classes.PyExprStep)
        else False for s in formalized.steps]
    prepared_steps = []
    direct_chains = {}   # chain family -> earlier direct claims, in step order
    # Conjunct dumps of every conclusion admitted to the general chain so far (general conclusions and bridged direct claims): a later step whose claims ALL restate entries here is an s*-chain echo, flagged claim_repeats_earlier and never rewarded (the chain admits translations before verification, so the echo would prove from its possibly-unverified twin).
    earlier_concl_dumps = set()
    for k, source_step in enumerate(formalized.steps):
        if isinstance(source_step, state_classes.DirectDomainStep):
            # Same-family chaining (general s* analog): earlier direct conclusions of the same chain family join this step's assumptions at translation time; ring and field share one family (same signature, locale-compatible statements), group is a different structure and stays separate. The domain pool's consistency probe fails a poisoned chain closed.
            chain = direct_chains.setdefault(source_step.spec.chain_family, [])
            prepared_steps.append(dataclasses.replace(
                source_step,
                previous_conclusions=tuple(chain),
                # the general->direct half of the bridge: every earlier general-session conclusion (general s* terms and earlier bridged direct claims), exactly what a general step in this slot would see; pv_ prefixes keep them collision-free in the domain theorem
                general_conclusions=tuple(
                    pair[0] for pair in isabelle_step_conclusions[:k] if pair is not None)))
            chain.append(source_step.claim)
            # Numerals stated by the direct claim become admissible provenance for later general steps' premises, symmetric with concl_nums accumulating from general conclusions.
            concl_nums |= {Fraction(m) for m in _DIRECT_CLAIM_NUM_RE.findall(source_step.claim)}
            if k in bridged_by_slot:
                # the bridged slot advances cross-step state like a general conclusion: its name becomes known (a later `order_G == 30` must classify as a claim, never as a fresh-name definition) and its numerals join provenance
                try:
                    known |= pyexpr.py_to_isabelle(bridged_by_slot[k], pyexpr_variable_types)[1]
                except pyexpr.PyExprError:
                    pass
                concl_nums |= consts_per[k]
                # the bridged equation fills this slot's s* entry, so a later general step restating it is an echo
                try:
                    earlier_concl_dumps |= {ast.dump(c) for c in conjuncts(bridged_by_slot[k])}
                except pyexpr.PyExprError:
                    pass
            continue
        nl_step = nl_steps[k]
        term, numeral_carrier = isabelle_step_conclusions[k]
        pyexpr_conclusion = source_step.pyexpr_conclusion
        step_pyexpr_premises = list(source_step.pyexpr_premises)
        # A translated step may use numbers from the problem or from the model's own step text. For example, `cos(pi/4)` is admissible when the model wrote it; any other new number is recorded by guard_invented.
        step_text_nums = number_provenance.num_values(nl_step.nl_step_text, words=True)
        allowed_step_nums = step_text_nums | given_nums
        for v in set(step_text_nums):
            allowed_step_nums.add(v * 100)
            if v != 0:
                allowed_step_nums.add(v / 100)
        guard_invented = [str(v) for v in consts_per[k]
                          if v not in allowed_step_nums]

        # Split top-level conjuncts into definitions and claims. A fresh name equated to an expression over known identifiers is a conservative definition and becomes an assumption rather than a proof obligation; `answer` and bare constants are never definitions.
        prop_conj = conjuncts(pyexpr_conclusion)
        isabelle_definitions, pyexpr_definitions = [], []
        isabelle_claims, pyexpr_claims, claim_restates_given = [], [], []
        # Track whether the definitions are affine for premise-consistency checks.
        defs_linear = True

        for c in prop_conj:
            is_def = (isinstance(c, ast.Compare) and len(c.ops) == 1
                      and isinstance(c.ops[0], ast.Eq)
                      and isinstance(c.left, ast.Name)
                      and c.left.id != "answer"
                      and c.left.id not in known)
            if is_def:
                try:
                    rids, _, _ = pyexpr.analyze(c.comparators[0])
                    is_def = bool(rids & known)
                except pyexpr.PyExprError:
                    is_def = False
            ccar = _node_carrier(c, pyexpr_variable_types, "real")
            if is_def:
                isabelle_definitions.append(pyexpr.transpile(c, pyexpr_variable_types, ccar))
                pyexpr_definitions.append(c)
                if not pyexpr.is_linear_arith(c):
                    defs_linear = False
            else:
                isabelle_claims.append(pyexpr.transpile(c, pyexpr_variable_types, ccar))
                pyexpr_claims.append(c)
                claim_restates_given.append(ast.dump(c) in given_dumps)

        # Echo detection over the two sources a claim could restate: the admitted chain (s*) and the problem givens (g*).
        # A step whose every claim conjunct restates one of them proves nothing new and earns nothing:
        # all conjuncts from the givens is claim_repeats_given;
        # an all-echo mix involving chain content is claim_repeats_earlier
        # (the chain admits translations before verification, so that echo may repeat an unverified twin).
        # Definitions do not add proof work;
        # one novel claim conjunct keeps the step on normal verification, where its restated-given conjuncts stay auto-satisfied (claim_restates_given).
        claim_dumps = [ast.dump(c) for c in pyexpr_claims]
        echo_all = bool(pyexpr_claims) and all(
            d in earlier_concl_dumps or d in given_dumps for d in claim_dumps)
        claim_repeats_given = echo_all and all(d in given_dumps for d in claim_dumps)
        claim_repeats_earlier = echo_all and not claim_repeats_given

        # Admit a step premise only when its numbers come from the problem, givens, earlier conclusions, or the model's current step text. admitted_linear records whether every admitted premise is affine. admitted_pyexpr_premises keeps the admitted SOURCES aligned with isabelle_step_premises, so the typed source records below can bind each form-B source to its p{k}_{i} assumption name.
        isabelle_step_premises, premises_nonlinear, admitted_linear = [], False, True
        admitted_pyexpr_premises = []
        prop_dumps = {ast.dump(c) for c in prop_conj}
        allowed_prem_nums = (given_nums | concl_nums | number_provenance.FREE_NUMS
                             | step_text_nums)
        for psrc in step_pyexpr_premises:
            try:
                node = pyexpr.parse_expr(psrc)
                pids, pcs, _ = pyexpr.analyze(node)
            except pyexpr.PyExprError:
                continue
            if any(ast.dump(c) in prop_dumps for c in conjuncts(psrc)):
                continue
            if not all(v in allowed_prem_nums for v in pcs):
                continue
            try:
                isabelle_step_premises.append(_transpile_conjunctive(psrc, pyexpr_variable_types)[0])
            except pyexpr.PyExprError:
                continue
            admitted_pyexpr_premises.append(psrc)
            if pyexpr.is_nonlinear(node):
                premises_nonlinear = True
            if not pyexpr.is_linear_arith(node):
                admitted_linear = False
        # Earlier steps' conclusions become premises of this step. A direct-domain slot participates only through its bridged whitelisted claim; a non-bridgeable direct slot holds None and is skipped (its conclusion stays out of the general chain).
        isabelle_previous_conclusions = [
            (f"s{j}", isabelle_step_conclusions[j][0])
            for j in range(k) if isabelle_step_conclusions[j] is not None]
        isabelle_nonzero_divisors = []
        if isabelle_claims:
            for claim_node in pyexpr_claims:
                claim_carrier = _node_carrier(claim_node, pyexpr_variable_types, "real")
                try:
                    claim_src = ast.unparse(claim_node)
                except AttributeError:
                    claim_src = pyexpr_conclusion
                isabelle_nonzero_divisors.extend(_nonconstant_divisors(
                    claim_src, pyexpr_variable_types, claim_carrier))
        # Step premises may also contain nonconstant denominators that require a nonzero proof.
        for psrc in step_pyexpr_premises:
            try:
                pcar = _node_carrier(pyexpr.parse_expr(psrc), pyexpr_variable_types, "real")
                isabelle_nonzero_divisors.extend(_nonconstant_divisors(psrc, pyexpr_variable_types, pcar))
            except pyexpr.PyExprError:
                pass
        isabelle_nonzero_divisors = list(dict.fromkeys(isabelle_nonzero_divisors))
        step_nonlin = (givens_nonlin or premises_nonlinear
                       or any(pyexpr.is_nonlinear(c) for c in pyexpr_claims)
                       or any(conclusions_nonlinear[:k]))
        # LINEAR_FALSE decides consistency completely when every accumulated premise is affine.
        linear_premises = (givens_linear and all(conclusions_linear[:k])
                       and admitted_linear and defs_linear)
        # The linear claim tactic also requires every claim to be affine.
        linear_step = (linear_premises and bool(pyexpr_claims)
                       and all(pyexpr.is_linear_arith(c) for c in pyexpr_claims))
        # The typed source records: every ADMISSION-PASSED form-B source bound to its exact form-C assumption name and source kind.
        # This is the only door into the general trig system (trigonometry.extract_trig_context),
        # so evidence can never come from a provenance-rejected premise and never re-selects assumptions by string search.
        admitted_sources = list(admitted_problem_sources)
        for j in range(k):
            if isabelle_step_conclusions[j] is None:
                continue
            conclusion_source = conclusion_sources_by_slot[j]
            if conclusion_source is None:
                continue
            admitted_sources.append(state_classes.AdmittedPyExprSource(
                source_kind=state_classes.PremiseSource.PREVIOUS_CONCLUSION,
                name=f"s{j}", pyexpr_source=conclusion_source,
                isabelle_term=isabelle_step_conclusions[j][0]))
        for i, admitted_source in enumerate(admitted_pyexpr_premises):
            admitted_sources.append(state_classes.AdmittedPyExprSource(
                source_kind=state_classes.PremiseSource.STEP,
                name=f"p{k}_{i}", pyexpr_source=admitted_source,
                isabelle_term=isabelle_step_premises[i]))
        for i, definition_node in enumerate(pyexpr_definitions):
            admitted_sources.append(state_classes.AdmittedPyExprSource(
                source_kind=state_classes.PremiseSource.DEFINITION,
                name=f"d{k}_{i}", pyexpr_source=ast.unparse(definition_node),
                isabelle_term=isabelle_definitions[i]))
        admitted_sources = tuple(admitted_sources)
        # Extraction runs only when some admitted source mentions a trig function;
        # an empty context still lets verification try the premise-free identity groups on a trig claim.
        if any(fn in source.pyexpr_source for source in admitted_sources
               for fn in trigonometry.TRIG_FUNCTIONS):
            trig_context = trigonometry.extract_trig_context(
                admitted_sources, pyexpr_variable_types)
        else:
            trig_context = state_classes.TrigContext()
        # Assumption-side sources for the tangent definedness check:
        # every model-authored assumption this step's theorem admits,
        # i.e. its definitions, its admitted step premises, and every earlier chain conclusion (s*, including bridged direct claims).
        # Without them a tangent pinned at a singular angle bypasses the check by riding an assumption instead of the claim
        # (measured: `x == tan(a)` splitting into a definition, and a rejected `x == tan(pi/2)` step still entering the chain).
        # Problem givens are not model-authored and stay out (translation trust).
        # Each text source here already parsed once during admission or transpilation, so the except is unreachable in practice.
        tan_assumption_nodes = list(pyexpr_definitions)
        for assumption_text in admitted_pyexpr_premises + [
                conclusion_sources_by_slot[j] for j in range(k)
                if isabelle_step_conclusions[j] is not None]:
            try:
                tan_assumption_nodes.append(pyexpr.parse_expr(assumption_text))
            except pyexpr.PyExprError:
                pass
        prepared_step = state_classes.IsabelleStep(
            k=k,
            pyexpr_premises=step_pyexpr_premises,
            pyexpr_conclusion=pyexpr_conclusion,
            pyexpr_definitions=pyexpr_definitions,
            pyexpr_claims=pyexpr_claims,
            claim_restates_given=claim_restates_given,
            pyexpr_variable_types=pyexpr_variable_types,
            isabelle_fixes=isabelle_fixes,
            isabelle_problem_premises=isabelle_problem_premises,
            isabelle_previous_conclusions=isabelle_previous_conclusions,
            isabelle_step_premises=[(f"p{k}_{i}", t)
                                    for i, t in enumerate(isabelle_step_premises)],
            isabelle_definitions=[(f"d{k}_{i}", t)
                                  for i, t in enumerate(isabelle_definitions)],
            isabelle_claims=isabelle_claims,
            isabelle_nonzero_divisors=isabelle_nonzero_divisors,
            numeral_carrier=numeral_carrier,
            nonlinear=step_nonlin,
            linear_premises=linear_premises,
            linear_step=linear_step,
            admitted_pyexpr_sources=admitted_sources,
            trig_context=trig_context,
            isabelle_tan_conditions=list(trigonometry.tangent_definedness_conditions(
                pyexpr_claims, pyexpr_variable_types,
                assumption_nodes=tan_assumption_nodes)),
            transcription_missing=formalized.transcription_missing[k],
            guard_invented=guard_invented,
            guard_ok=not guard_invented,
            n_definitions=len(isabelle_definitions),
            n_admitted_premises=len(isabelle_step_premises),
            claim_repeats_earlier=claim_repeats_earlier,
            claim_repeats_given=claim_repeats_given)
        # The consistency probe is built here so the prover-facing code only submits it. Dangerous premise terms leave it None: that step's premises are scored UNKNOWN without a prover call.
        if not tactics.is_dangerous_isabelle(*[t for _, t in prepared_step.isabelle_premises]):
            prepared_step.premise_consistency_theorem = theorem_builders.make_theorem(
                isabelle_fixes, prepared_step.isabelle_premises, "False",
                tactics.LINEAR_FALSE if linear_premises else tactics.ALTERNATION)
        prepared_steps.append(prepared_step)
        # cross-step state advances on TRANSLATIONS, exactly as before
        try:
            known |= pyexpr.py_to_isabelle(pyexpr_conclusion, pyexpr_variable_types)[1]
        except pyexpr.PyExprError:
            pass
        concl_nums |= consts_per[k]
        # this conclusion enters the chain as s{k}, so its conjuncts are echo material for later steps
        earlier_concl_dumps |= {ast.dump(c) for c in prop_conj}
    return state_classes.PreparationOutput(
        steps=prepared_steps, isabelle_fixes=isabelle_fixes, function_arities=func_ar,
        isabelle_problem_conditions=isabelle_problem_conditions,
        isabelle_step_conclusions=isabelle_step_conclusions)
