"""The verification stage of the pipeline: checking every prepared step and creating its result.

verify_response is the response-level entry point: it runs the premise-consistency probes, derives the boundaries, checks every step by its type (an IsabelleStep against the general pool, a DirectDomainStep against its domain pool), creates one StepVerificationResult per step, and builds the per-step pattern.

verify_claims_one_step checks one prepared general step in a fixed order. It first tries self-contained trigonometry and calculus theorems, then the primary with-premises proof, premise-free checks, syntax-specific with-premises tactics, and decimal tolerance. Proven nonzero conditions join the premises only for claim checking.

Owns no pool: every prover call, general and direct-domain alike, goes through the injected `verify` callable (the merged Isa_Step session serves both paths; direct_verify normalizes results itself, so test fakes may return either a typed VerificationOutcome or the legacy two-key dict).
"""
import ast
import re
import threading

import verl.utils.isabelle_utils.pyexpr as pyexpr
import verl.utils.isabelle_utils.stages.direct_verify as direct_verify
import verl.utils.isabelle_utils.stages.preparation as preparation
import verl.utils.isabelle_utils.state_classes as state_classes
import verl.utils.isabelle_utils.tactics as tactics
import verl.utils.isabelle_utils.theorem_builders as theorem_builders
import verl.utils.isabelle_utils.trigonometry as trigonometry


def tolerance_goal(pyexpr_claims, pyexpr_variable_types, carrier):
    """Rewrite decimal equalities using half a written decimal unit.

    Decimal value and precision come from source annotations installed by ``pyexpr.parse_expr()``, not from ``repr(float)``. Therefore ``0.250`` retains three decimal places, and long or scientific literals retain their exact values.
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
        # An integer is exact and does not reduce the decimal side's written precision. Treating it as zero decimal places would give `0 == 0.250` a tolerance of 1/2 and incorrectly accept it.
        # Returning None excludes the integer from the minimum below.
        return None

    for c in pyexpr_claims:
        dec = None
        ccarrier = preparation._node_carrier(c, pyexpr_variable_types, carrier)
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
            parts.append(pyexpr.transpile(c, pyexpr_variable_types, ccarrier))
            continue
        other, val, places = dec
        lhs_t = pyexpr.transpile(other, pyexpr_variable_types, "real")
        tol_den = 2 * 10 ** places
        parts.append(f"(abs ({lhs_t} - (({val.numerator}::real) / "
                     f"({val.denominator}::real))) < "
                     f"(1::real) / ({tol_den}::real))")
        changed = True
    if not changed or not parts:
        return None
    return " & ".join(parts) if len(parts) > 1 else parts[0]


def verify_claims_one_step(proposition, verify, function_arities):
    """Ordered verification of one prepared step's claims; returns {"verified", "tolerance", "danger"}.

    `verify` is the orchestrator's normalized prover callable (engine.PoolVerifier). `function_arities` is response-wide (PreparationOutput.function_arities); a non-empty map enables the SMT attempt for uninterpreted functions. isabelle_fixes comes off the proposition; every IsabelleStep of a response shares the one list.

    Every with-premises attempt carries the equal-strength consistency guard (_guarded_wp): a success counts only when the same theorem shape cannot derive False. Premise-free attempts (the canonical bare goal, eval, limits/derivatives/integrals) need no guard; the unfold attempt is exempt because its premises act only as goal rewrites.
    """
    isabelle_fixes = proposition.isabelle_fixes

    def _canonical_fixes(goal):
        """Minimal, sorted fixes for a premise-free goal: only the variables that actually occur in the goal text.
        Unused universally-quantified fixes are vacuous for provability, but they make otherwise-identical bare theorems (the same arithmetic claim across rollouts) textually different, defeating the theorem cache.
        Sorting canonicalizes the text across responses, so identical bare claims share ONE cache entry (memory + disk) instead of multiple."""
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", goal))
        return sorted((n, t) for (n, t) in isabelle_fixes if n in toks)

    isabelle_premises, isabelle_claims = proposition.isabelle_premises, proposition.isabelle_claims
    # Claims containing huge literals, exponents of at least 1000, factorials, or power towers can keep the initial simp/presburger attempts busy for 60 to 75 seconds, beyond the 15-second watchdog.
    # Use only EVAL_TACTIC for the entire step. Skip synthesized nonzero premises and decimal tolerance because they would run the same expensive tactics again.
    danger = tactics.is_dangerous_isabelle(*isabelle_claims, *[t for _, t in isabelle_premises])
    out = {"verified": None, "tolerance": False, "danger": danger}
    # When consistency is unknown, disable nonzero synthesis, tolerance, and the specialized premise-dependent tactics below.
    # The primary tactic still gets one with-premises attempt;
    # if it fails, verification tries the canonical premise-free goal and premise-free evaluation.
    premise_consistency_unknown = proposition.premise_consistency is state_classes.PremiseConsistency.UNKNOWN
    # SOS is useful only for nonlinear claims and premises. It is disabled for dangerous terms and when premise consistency is unknown.
    nonlin = (not danger) and proposition.nonlinear
    isabelle_proven_nonzero_premises = []
    # Nonzero synthesis runs without its own consistency guard: a proven `d != 0` only ever JOINS the premises of claim attempts, and every with-premises attempt below carries the equal-strength guard over the same joined premise list, so a contradiction cannot be laundered through this step.
    if not danger and not premise_consistency_unknown:
        for d in proposition.isabelle_nonzero_divisors:
            # ALTERNATION proves most d != 0; a denominator that is itself only pinned through a rational premise (pipe, from 1/comb = 1/pipe - 1/leak) needs field_simps, so fall back to FIELD_TACTIC.
            for nztac in (tactics.ALTERNATION, tactics.FIELD_TACTIC):
                if verify(theorem_builders.make_theorem(isabelle_fixes, isabelle_premises,
                                       f"{d} \\<noteq> 0", nztac)).proved:
                    isabelle_proven_nonzero_premises.append((f"nz{len(isabelle_proven_nonzero_premises)}",
                                    f"{d} \\<noteq> 0"))
                    break
    isabelle_premises_with_nonzero = isabelle_premises + isabelle_proven_nonzero_premises
    tac = tactics.SAFE_DANGEROUS if danger else tactics.ALTERNATION

    def _guarded_wp(builder, premises, goal_text, tactic):
        """One with-premises attempt under the equal-strength consistency guard: the attempt counts only when the SAME builder, premise list, and tactic cannot derive False. The response-level probe runs only LINEAR_FALSE or ALTERNATION (preparation), so any stronger tactic or auxiliary-fact theorem here could otherwise prove a claim from a contradiction that probe cannot see; the direct path already probes per claim tactic, and this is the general-path analog. The guard runs only AFTER a success, so its extra prover call is paid only where a reward is at stake; a guard that proves False, times out, or errors voids the success (fail closed)."""
        if not verify(builder(isabelle_fixes, premises, goal_text, tactic)).proved:
            return False
        guard = verify(builder(isabelle_fixes, premises, "False", tactic))
        return guard.outcome is state_classes.ProofOutcome.UNPROVED

    # Tangent definedness check, a MANDATORY pre-check ahead of every claim attempt:
    # HOL totalizes tan (the kernel evaluates tan(pi/2) to 0), so a tangent value is mathematically meaningful ONLY where its cosine is nonzero,
    # and without this cosine-nonzero requirement the special-angle simp set, the generic cascade, and TRIG_TACTIC would all accept the totalized value
    # (measured FP: tan(pi/2) == 0 verified before 2026-07-18).
    # The conditions cover the claim conjuncts and the model-authored assumptions alike
    # (preparation scans definitions, admitted step premises, and earlier chain conclusions:
    # a tangent pinned at a singular angle must not smuggle its totalized value in as an assumption either).
    # Each condition (built from the stored angle term; a None angle is untranspilable and fails closed)
    # is proved premise-free first (the plain tactic, then the special-angle values),
    # then with premises under the equal-strength guard:
    # the plain tactic, then the four quadrant sign-lemma theorems
    # (`cos(2*a) != 0` from interval bounds needs the sign lemmas, which ALTERNATION does not carry and no flat tactic form was measured to reach);
    # an unprovable condition fails the whole step closed, whichever tactic family would have proved the claim.
    for tan_angle, tan_decomp in proposition.isabelle_tan_conditions:
        if tan_angle is None:
            out["verified"] = False
            return out
        tan_condition = f"cos {tan_angle} \\<noteq> 0"
        if any(verify(theorem_builders.make_theorem(
                _canonical_fixes(tan_condition), [], tan_condition,
                condition_tactic)).proved
               for condition_tactic in (tac, trigonometry.SPECIAL_VALUE_TACTIC)):
            continue
        # a literal non-base pi-multiple angle (tan(5*pi/4)) proves its condition through the kernel-checked decomposition;
        # premise-free, so no guard is needed
        if tan_decomp is not None and verify(trigonometry.literal_condition_theorem(
                tan_angle, tan_decomp, tan_condition)).proved:
            continue
        if not danger and not premise_consistency_unknown:
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, tan_condition, tac):
                continue
            sign_builders = [
                lambda fixes, premises, goal, tactic, _lemma=lemma, _positive=positive:
                    trigonometry.cosine_nonzero_theorem(
                        fixes, premises, tan_angle, _lemma, _positive, goal)
                for lemma, positive in trigonometry.QUADRANT_SIGN_LEMMAS]
            if any(_guarded_wp(builder, isabelle_premises_with_nonzero,
                               tan_condition, tac)
                   for builder in sign_builders):
                continue
        out["verified"] = False
        return out

    def _verify_single(goal, node=None):
        # General trig attempts run first (trigonometry.plan_attempts): a bounded, deterministically ordered list per claim.
        # Premise-free identity and special-angle attempts carry no assumptions, so they run even under unknown premise consistency and need no guard.
        # Every premise-dependent attempt renders through ONE function twice (the goal, then False), so its guard twin is byte-identical apart from the goal:
        # only a PROVED positive with an UNPROVED twin rewards, and a twin that proves False, times out, or errors voids that attempt (fail closed).
        trig_attempts = trigonometry.plan_attempts(
            proposition.trig_context, goal, dict(proposition.isabelle_premises),
            claim_node=node)
        for attempt in trig_attempts:
            fixes = trigonometry.attempt_fixes(isabelle_fixes, goal, attempt)
            if not attempt.premise_dependent:
                if verify(trigonometry.render_attempt(fixes, goal, attempt)).proved:
                    return True
                continue
            if premise_consistency_unknown:
                continue
            if not verify(trigonometry.render_attempt(fixes, goal, attempt)).proved:
                continue
            guard = verify(trigonometry.render_attempt(fixes, "False", attempt))
            if guard.outcome is state_classes.ProofOutcome.UNPROVED:
                return True
        # Limits, derivatives, and definite integrals are proved without the accumulated premises, so they also remain available when premise consistency is unknown.
        if theorem_builders.has_limit(goal) and verify(theorem_builders.make_theorem(
                _canonical_fixes(goal), [], goal, theorem_builders.LIMIT_TACTIC)).proved:
            return True
        if theorem_builders.has_deriv(goal) and verify(theorem_builders.make_theorem(
                _canonical_fixes(goal), [], goal, theorem_builders.DERIV_TACTIC)).proved:
            return True
        if node is not None and theorem_builders.has_integral_goal(goal):
            integral_theorem = theorem_builders.integral_recipe(node, goal)
            if integral_theorem is not None and verify(integral_theorem).proved:
                return True
        # Symbolic finite sums are not handled here.
        # pyexpr currently types the range bound as int, while induction needs a nat target or an explicit nonnegative-int induction rule.
        if not premise_consistency_unknown:
            tactic = tactics.LINEAR_CLAIM if proposition.linear_step else tac
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, goal, tactic):
                return True
        else:
            # One attempt with premises stays allowed under unknown consistency: the per-attempt guard establishes with the SAME tactic what the response probe could not, so a success here cannot be an ex-falso route. If it fails, verification continues with premise-free checks only.
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, goal, tac):
                return True
        if verify(theorem_builders.make_theorem(
                _canonical_fixes(goal), [], goal, tac)).proved:
            return True
        if danger:
            return False
        if verify(theorem_builders.make_theorem(
                _canonical_fixes(goal), [], goal, tactics.EVAL_TACTIC)).proved:
            return True
        # The remaining tactics use accumulated premises and are disabled when their consistency is unknown.
        if not premise_consistency_unknown and tactics.has_division(
                goal, *[t for _, t in isabelle_premises_with_nonzero]):
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, goal, tactics.FIELD_TACTIC):
                return True
        # Syntax dispatch reads the goal AND the premises (a linear-looking goal may need a premise's polynomial or trig relation expanded), matching the field dispatch above.
        if not premise_consistency_unknown and tactics.has_trig(
                goal, *[t for _, t in isabelle_premises_with_nonzero]):
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, goal, tactics.TRIG_TACTIC):
                return True
        if not premise_consistency_unknown and tactics.has_poly(
                goal, *[t for _, t in isabelle_premises_with_nonzero]):
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, goal, tactics.ALGEBRA_TACTIC):
                return True
        if not premise_consistency_unknown:
            # The unfold attempt needs no consistency guard: its premises act only as rewrite rules on the goal (no `using assms`), the goal False is not rewritten by variable equations, and eval cannot prove False, so no ex-falso route exists through it.
            u_theorem = theorem_builders.make_theorem_unfold(isabelle_fixes, isabelle_premises_with_nonzero, goal)
            if u_theorem is not None and verify(u_theorem).proved:
                return True
            combined = " ".join([goal] + [t for _, t in isabelle_premises_with_nonzero])
            if ("mod" in combined or "dvd" in combined
                    or proposition.numeral_carrier in ("int", "nat")):
                if _guarded_wp(theorem_builders.make_theorem,
                               isabelle_premises_with_nonzero, goal, "(presburger | arith)"):
                    return True
        if nonlin and not premise_consistency_unknown:
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, goal, tactics.SOS_TACTIC):
                return True
        # SMT for uninterpreted functions fires only when THIS step's goal or premises actually apply one (transpiled as pv_<name>); function_arities is response-wide, and dispatching on it alone made every unrelated step pay the SMT call.
        if function_arities and not premise_consistency_unknown:
            step_texts = [goal] + [t for _, t in isabelle_premises_with_nonzero]
            if any(re.search(r"\bpv_%s\b" % re.escape(name), text)
                   for name in function_arities for text in step_texts):
                if _guarded_wp(theorem_builders.make_theorem_with_logs,
                               isabelle_premises_with_nonzero, goal, tactics.SMT_TACTIC):
                    return True
        if not premise_consistency_unknown and theorem_builders.has_log(
                goal, *[t for _, t in isabelle_premises_with_nonzero]):
            if _guarded_wp(theorem_builders.make_theorem_with_logs,
                           isabelle_premises_with_nonzero, goal, tactics.ALTERNATION):
                return True
        if not premise_consistency_unknown and tactics.has_powr(
                goal, *[t for _, t in isabelle_premises_with_nonzero]):
            if _guarded_wp(theorem_builders.make_theorem,
                           isabelle_premises_with_nonzero, goal, tactics.EXPONENT_TACTIC):
                return True
        return False

    # A claim identical to a GIVEN is auto-satisfied (claim_restates_given, from the preparation stage),
    # including when premise consistency is unknown.
    gflags, cnodes = (proposition.claim_restates_given or [False] * len(isabelle_claims),
                      proposition.pyexpr_claims or [None] * len(isabelle_claims))
    if len(isabelle_claims) > 1:
        out["verified"] = all(gf or _verify_single(g, n)
                              for gf, g, n in zip(gflags, isabelle_claims, cnodes))
    else:
        out["verified"] = gflags[0] or _verify_single(
            isabelle_claims[0], cnodes[0] if cnodes else None)
    if not out["verified"] and not danger and not premise_consistency_unknown:
        # The tolerance fallback proves an approximate goal from premises,
        # so it is disabled when their consistency is unknown.
        tol = tolerance_goal(proposition.pyexpr_claims,
                             proposition.pyexpr_variable_types,
                             proposition.numeral_carrier)
        if tol:
            for tol_tactic in ("(approximation 20)", tactics.ALTERNATION):
                if _guarded_wp(theorem_builders.make_theorem,
                               isabelle_premises + isabelle_proven_nonzero_premises,
                               tol, tol_tactic):
                    out["verified"] = True
                    out["tolerance"] = True
                    break
    return out


_CLAIM_CHECK_EXECUTOR = None
_CLAIM_CHECK_EXECUTOR_LOCK = threading.Lock()


def _claim_check_executor(pool_workers: int):
    """Return the process-wide executor used for step claim checks.

    Its threads mostly wait on server-pool futures, so the executor is sized
    above the number of Isabelle workers and is created once per process.
    """
    global _CLAIM_CHECK_EXECUTOR
    with _CLAIM_CHECK_EXECUTOR_LOCK:
        if _CLAIM_CHECK_EXECUTOR is None:
            from concurrent.futures import ThreadPoolExecutor
            _CLAIM_CHECK_EXECUTOR = ThreadPoolExecutor(
                max_workers=max(32, 8 * int(pool_workers or 4)),
                thread_name_prefix="isa-claim-check")
    return _CLAIM_CHECK_EXECUTOR


def _pmap(fn, items, parallelism):
    """Map `fn` over `items` with at most `parallelism` threads (1 = strictly sequential, preserving step order)."""
    if not items:
        return []
    if parallelism == 1 or len(items) == 1:
        return [fn(p) for p in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(parallelism, len(items))) as ex:
        return list(ex.map(fn, items))


def verify_response(prepared, verify, step_check_parallelism=1, pool_workers=4):
    """Response-level verification entry point: check every prepared step and create its result.

    Runs the premise-consistency probes (general steps only: a DirectDomainStep gets no general-path probe because its whole assumption set, injected chains included, is probed in-locale inside direct_verify), derives the response boundaries, runs the claim checks for every general step not blocked by a proven inconsistency, dispatches each DirectDomainStep to direct_verify.verify with the SAME `verify` callable (one merged pool serves both paths; direct_verify normalizes each result itself), and creates one StepVerificationResult per response step. `verify` is the orchestrator's normalized prover callable (engine.PoolVerifier); when it exposes `submit`, probes are submitted as futures and claim checks run on the process-wide executor, otherwise everything runs synchronously in step order.
    """
    steps = prepared.steps

    # PREMISE CONSISTENCY before claim checking.
    # Classify each accumulated premise set as consistent, inconsistent, or unknown. A completed check that proves False means inconsistent; anything the prover could not decide means unknown. The classification lives on VerificationOutcome so every consumer resolves the same way.
    def _classify_premise_consistency(result):
        return state_classes.VerificationOutcome.from_raw(result).premise_consistency

    consistency_results, pending_consistency_checks = [None] * len(steps), []
    for i, proposition in enumerate(steps):
        if not isinstance(proposition, state_classes.IsabelleStep):
            continue
        if proposition.premise_consistency_theorem is None:
            # dangerous premise terms: preparation built no probe, scored UNKNOWN without a prover call
            consistency_results[i] = state_classes.PremiseConsistency.UNKNOWN
        elif verify.submit is not None:
            pending_consistency_checks.append(
                (i, verify.submit(proposition.premise_consistency_theorem)))
        else:
            # synchronous path: mocks and pools without `submit`
            consistency_results[i] = _classify_premise_consistency(
                verify(proposition.premise_consistency_theorem))
    for i, future in pending_consistency_checks:
        result = state_classes.VerificationOutcome.from_raw(future.result())
        verify.record(result)
        consistency_results[i] = _classify_premise_consistency(result)

    # The first inconsistency blocks that general step and every later general step.
    # The first unknown result enables restricted claim checking from its step on, unless an inconsistency takes precedence.
    boundaries = state_classes.PremiseConsistencyBoundaries(
        inconsistent_from_step=next(
            (i for i, result in enumerate(consistency_results)
             if result is state_classes.PremiseConsistency.INCONSISTENT), None),
        unknown_from_step=next(
            (i for i, result in enumerate(consistency_results)
             if result is state_classes.PremiseConsistency.UNKNOWN), None))

    # Claim checks for every general step not blocked by a proven inconsistency.
    # Unknown consistency allows only the restricted proof attempts in verify_claims_one_step.
    def _check_one(proposition):
        return verify_claims_one_step(proposition, verify, prepared.function_arities)

    steps_to_verify = []
    for i, proposition in enumerate(steps):
        if not isinstance(proposition, state_classes.IsabelleStep) or not proposition.isabelle_claims:
            continue
        if proposition.claim_repeats_earlier or proposition.claim_repeats_given:
            # an echo earns nothing, so no prover time is spent on it: an s*-chain echo could only prove via its injected twin, a given restatement is an assumption repeated verbatim
            continue
        premise_consistency = boundaries.status_for(i)
        if premise_consistency is state_classes.PremiseConsistency.INCONSISTENT:
            continue
        proposition.premise_consistency = premise_consistency
        steps_to_verify.append(proposition)
    if verify.submit is not None and steps_to_verify:
        executor = _claim_check_executor(pool_workers)
        claim_check_results = list(executor.map(_check_one, steps_to_verify))
    else:
        claim_check_results = _pmap(_check_one, steps_to_verify, step_check_parallelism)
    claim_results_by_step = dict(zip(
        (proposition.k for proposition in steps_to_verify), claim_check_results))

    # One StepVerificationResult per response step, created here (preparation produces only verification input).
    results = []
    for i, proposition in enumerate(steps):
        entry = state_classes.StepVerificationResult(step=i)
        results.append(entry)
        premise_consistency = boundaries.status_for(i)
        # Record the general-path consistency status at every response position, including direct-domain steps.
        entry.premise_consistency_inconsistent = (
            premise_consistency is state_classes.PremiseConsistency.INCONSISTENT)
        if isinstance(proposition, state_classes.DirectDomainStep):
            # A direct-domain step checks its own premises and injected conclusions in its locale. General-path consistency boundaries do not block it because its own consistency probe covers the complete assumption set it consumes. direct_verify normalizes each prover result through VerificationOutcome.from_raw.
            try:
                ok, reason = direct_verify.verify(
                    proposition.spec, proposition.premises, proposition.claim,
                    proposition.nl_step_text, verify,
                    previous_conclusions=proposition.previous_conclusions,
                    general_conclusions=proposition.general_conclusions)
            except Exception:  # noqa: BLE001
                ok, reason = False, "domain_error"
            entry.verified = ok
            entry.rewarded = ok
            entry.domain_reason = reason
            continue
        entry.transcription_missing = list(proposition.transcription_missing)
        entry.guard_invented = list(proposition.guard_invented)
        entry.guard_ok = proposition.guard_ok
        entry.n_definitions = proposition.n_definitions
        entry.n_admitted_premises = proposition.n_admitted_premises
        if premise_consistency is state_classes.PremiseConsistency.UNKNOWN:
            entry.premise_consistency_unknown = True
        if not proposition.isabelle_claims:
            entry.neutral = True
            entry.verified = True
            entry.rewarded = False
            continue
        if premise_consistency is state_classes.PremiseConsistency.INCONSISTENT:
            entry.verified = False
            entry.rewarded = False
            continue
        if proposition.claim_repeats_earlier:
            # A claim that only repeats an earlier injected conclusion would prove from its identical premise even when the original step failed. It is not checked or rewarded; consistency symbols retain their higher pattern priority.
            entry.claim_repeats_earlier = True
            entry.verified = False
            entry.rewarded = False
            continue
        if proposition.claim_repeats_given:
            # A claim that only restates problem givens is true by assumption and contentless; rewarding it was a measured farming surface, so it is not checked or rewarded (pattern x, same fate as the s*-chain echo).
            entry.claim_repeats_given = True
            entry.verified = False
            entry.rewarded = False
            continue
        result = claim_results_by_step.get(proposition.k)
        entry.verified = bool(result and result["verified"])
        if result and result["tolerance"]:
            entry.tolerance = True
        # Unknown consistency is not a separate reward condition:
        # claim checking has already restricted which premise-dependent tactics may run.
        entry.rewarded = (entry.verified and entry.guard_ok
                          and not entry.transcription_missing)

    # Per-step symbols: o=rewarded, c=inconsistent premises, u=unknown premise consistency and unverified, m=verified but missing transcription, g=verified but withheld by another guard, x=unverified.
    pattern = "".join(
        "o" if entry.rewarded else
        "c" if entry.premise_consistency_inconsistent else
        "u" if entry.premise_consistency_unknown else
        "m" if entry.verified and entry.transcription_missing else
        "g" if entry.verified else "x"
        for entry in results)
    return state_classes.VerificationOutput(
        steps=results, boundaries=boundaries, pattern=pattern)
