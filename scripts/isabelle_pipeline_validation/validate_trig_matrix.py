"""Real-pool validation case table for the general trigonometry system.

Every case runs through preparation.prepare AND verification.verify_claims_one_step (not the builder alone), against the real Isa_Step pool,
so the admitted-source records, TrigContext extraction, attempt planning, rendering, and the equal-strength False guard are all exercised exactly as in production.
Positive cases must verify;
negative cases (wrong value, wrong sign, quadrant-impossible tangent, conflicting evidence, empty or endpoint bounds, mismatched angles) must not.

Run on a node with the rebuilt Isa_Step heap:
    bash scripts/isabelle_pipeline_validation/with_env.sh python -u scripts/isabelle_pipeline_validation/validate_trig_matrix.py
Exit code 0 = every expectation held.
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl.utils.isabelle_utils import server_pool, state_classes  # noqa: E402
from verl.utils.isabelle_utils.stages import preparation, verification  # noqa: E402


class PoolVerify:
    submit = None

    def __init__(self, pool):
        self.pool = pool

    def __call__(self, theorem):
        return state_classes.VerificationOutcome.from_raw(self.pool.check(theorem))


def _formalized(conclusions, premises=None, declared=None, givens=None):
    n = len(conclusions)
    premises = [list(p) for p in (premises or [[]] * n)]
    return state_classes.FormalizationOutput(
        nl_steps=[state_classes.NaturalLanguageStep(
            nl_premises=[], nl_conclusion="step %d" % k,
            nl_step_text="step %d text" % k) for k in range(n)],
        problem_nums={Fraction(v) for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 24)},
        givens_ok=True, steps_ok=True,
        pyexpr_givens=list(givens or ["answer == n"]),
        pyexpr_variable_types_declared=dict(
            declared or {"n": "int", "answer": "int",
                         "a": "real", "b": "real", "x": "real"}),
        steps=[state_classes.PyExprStep(pyexpr_conclusion=conclusions[k],
                                        pyexpr_premises=premises[k])
               for k in range(n)],
        transcription_missing=[[] for _ in range(n)])


def check_case(verify, conclusions, premises, step_index=0, givens=None):
    prepared = preparation.prepare(
        _formalized(conclusions, premises=premises, givens=givens))
    step = prepared.steps[step_index]
    step.premise_consistency = state_classes.PremiseConsistency.CONSISTENT
    result = verification.verify_claims_one_step(step, verify, prepared.function_arities)
    return bool(result["verified"])


CASES = [
    # --- tan + quadrant value family: all four quadrants, sine and cosine ---
    ("Q1-sin", True, ["sin(a) == 3/5"], [["tan(a) == 3/4", "0 < a", "a < pi / 2"]]),
    ("Q2-sin", True, ["sin(a) == 3/5"], [["tan(a) == -3/4", "pi / 2 < a", "a < pi"]]),
    ("Q3-sin", True, ["sin(a) == -3/5"], [["tan(a) == 3/4", "pi < a", "a < 3 * pi / 2"]]),
    ("Q4-sin", True, ["sin(a) == -3/5"], [["tan(a) == -3/4", "3 * pi / 2 < a", "a < 2 * pi"]]),
    ("Q1-cos", True, ["cos(a) == 4/5"], [["tan(a) == 3/4", "0 < a", "a < pi / 2"]]),
    ("Q3-cos", True, ["cos(a) == -4/5"], [["tan(a) == 3/4", "pi < a", "a < 3 * pi / 2"]]),
    # radical tangent value and a composite claim built from both derived values
    ("Q3-radical-cos", True, ["cos(a) == -1/2"],
     [["tan(a) == sqrt(3)", "pi < a", "a < 3 * pi / 2"]]),
    ("Q3-composite", True, ["cos(a) - sin(a) == (sqrt(3) - 1)/2"],
     [["tan(a) == sqrt(3)", "pi < a", "a < 3 * pi / 2"]]),
    # --- negatives for the value family ---
    ("Q1-wrong-value", False, ["sin(a) == 4/5"], [["tan(a) == 3/4", "0 < a", "a < pi / 2"]]),
    ("Q1-wrong-sign", False, ["sin(a) == -3/5"], [["tan(a) == 3/4", "0 < a", "a < pi / 2"]]),
    ("Q1-tan-sign-contradiction", False, ["sin(a) == -3/5"],
     [["tan(a) == -3/4", "0 < a", "a < pi / 2"]]),
    ("conflicting-tans", False, ["sin(a) == 3/5"],
     [["tan(a) == 3/4", "tan(a) == 5/4", "0 < a", "a < pi / 2"]]),
    ("empty-bounds", False, ["sin(a) == -3/5"],
     [["tan(a) == 3/4", "pi < a", "a < pi / 2"]]),
    ("endpoint-bounds", False, ["sin(a) == 3/5"],
     [["tan(a) == 3/4", "0 <= a", "a < pi / 2"]]),
    ("mismatched-angle", False, ["sin(a) == 3/5"],
     [["tan(b) == 3/4", "0 < a", "a < pi / 2"]]),
    # --- premise-free identity groups ---
    ("identity-pi-symmetry", True, ["sin(pi - a) == sin(a)"], None),
    ("identity-cofunction", True, ["sin(pi / 2 - a) == cos(a)"], None),
    ("identity-periodicity", True, ["sin(a + 2 * pi) == sin(a)"], None),
    ("identity-angle-addition", True,
     ["sin(a + b) == sin(a) * cos(b) + cos(a) * sin(b)"], None),
    ("identity-wrong-sign", False,
     ["sin(a + b) == sin(a) * cos(b) - cos(a) * sin(b)"], None),
    ("identity-wrong-coefficient", False, ["sin(pi - a) == 2 * sin(a)"], None),
    ("special-angle-30", True, ["sin(pi / 6) == 1/2"], None),
    ("special-angle-45", True, ["sin(pi / 4) == sqrt(2) / 2"], None),
    ("special-angle-wrong", False, ["sin(pi / 6) == 1/3"], None),
    ("pythagorean-identity", True, ["sin(a)**2 + cos(a)**2 == 1"], None),
    # --- Pythagorean value family ---
    ("pyth-cos-from-sin-Q2", True, ["cos(a) == -4/5"],
     [["sin(a) == 3/5", "pi / 2 < a", "a < pi"]]),
    ("pyth-wrong-sign", False, ["cos(a) == 4/5"],
     [["sin(a) == 3/5", "pi / 2 < a", "a < pi"]]),
    # --- tan from both values (the definedness check must pass through the cosine premise) ---
    ("tan-from-values", True, ["tan(a) == 3/4"],
     [["sin(a) == 3/5", "cos(a) == 4/5"]]),
    ("tan-from-values-wrong", False, ["tan(a) == 4/3"],
     [["sin(a) == 3/5", "cos(a) == 4/5"]]),
    # --- tangent singularity:
    #     HOL totalizes tan, the mandatory pre-check must refuse the totalized value ---
    ("tan-singularity-pi-half", False, ["tan(pi / 2) == 0"], None),
    ("tan-singularity-3pi-half", False, ["tan(3 * pi / 2) == 0"], None),
    ("tan-special-angle", True, ["tan(pi / 4) == 1"], None),
    # --- product/sum transformations as dedicated groups ---
    ("product-to-sum", True,
     ["sin(a) * sin(b) == (cos(a - b) - cos(a + b)) / 2"], None),
    ("sum-to-product", True,
     ["sin(a) + sin(b) == 2 * sin((a + b) / 2) * cos((a - b) / 2)"], None),
    # --- double angle ---
    ("double-angle-cos", True, ["cos(2 * a) == 2 * cos(a)**2 - 1"], None),
    ("double-angle-sin", True, ["sin(2 * a) == 2 * sin(a) * cos(a)"], None),
    ("double-angle-wrong", False, ["cos(2 * a) == 2 * cos(a)**2 + 1"], None),
    # --- Q2/Q4 cosine symmetry completion ---
    ("Q2-cos", True, ["cos(a) == -4/5"],
     [["tan(a) == -3/4", "pi / 2 < a", "a < pi"]]),
    ("Q4-cos", True, ["cos(a) == 4/5"],
     [["tan(a) == -3/4", "3 * pi / 2 < a", "a < 2 * pi"]]),
    # --- period-shifted quadrant intervals (one full period above/below [0, 2*pi)) ---
    ("period-shift-up-sin", True, ["sin(a) == 3/5"],
     [["tan(a) == -3/4", "5 * pi / 2 < a", "a < 3 * pi"]]),
    ("period-shift-down-cos", True, ["cos(a) == -4/5"],
     [["tan(a) == -3/4", "-3 * pi / 2 < a", "a < -pi"]]),
    ("period-shift-wrong-sign", False, ["sin(a) == -3/5"],
     [["tan(a) == -3/4", "5 * pi / 2 < a", "a < 3 * pi"]]),
    # --- literal non-base angles via kernel-checked decomposition ---
    ("literal-angle-5pi6", True, ["sin(5 * pi / 6) == 1/2"], None),
    ("literal-angle-7pi4", True, ["cos(7 * pi / 4) == sqrt(2) / 2"], None),
    ("literal-angle-13pi6", True, ["sin(13 * pi / 6) == 1/2"], None),
    ("literal-angle-wrong", False, ["sin(5 * pi / 6) == 1/3"], None),
    # --- tangent double angle:
    #     definedness for BOTH the angle and its double must be kernel-proved from the bounds;
    #     without bounds the definedness check fails the claim closed ---
    ("tan-double", True, ["tan(2 * a) == 24/7"],
     [["tan(a) == 3/4", "0 < a", "a < pi / 4"]]),
    ("tan-double-wrong", False, ["tan(2 * a) == 7/24"],
     [["tan(a) == 3/4", "0 < a", "a < pi / 4"]]),
    ("tan-double-no-bounds", False, ["tan(2 * a) == 24/7"],
     [["tan(a) == 3/4"]]),
    # --- inverse trig: conditional rewrites fire only inside the range ---
    ("inverse-sin-literal", True, ["sin(arcsin(1/3)) == 1/3"], None),
    ("inverse-tan-literal", True, ["tan(arctan(5)) == 5"], None),
    ("inverse-range-premises", True, ["arcsin(sin(a)) == a"],
     [["-pi/2 <= a", "a <= pi/2"]]),
    ("inverse-out-of-range", False, ["sin(arcsin(2)) == 2"], None),
    ("inverse-no-range-premise", False, ["arcsin(sin(a)) == a"], None),
    # --- direction-fixed negatives for the transformation groups + the diff double form ---
    ("product-to-sum-wrong", False,
     ["sin(a) * sin(b) == (cos(a - b) + cos(a + b)) / 2"], None),
    ("sum-to-product-wrong", False,
     ["sin(a) + sin(b) == 2 * sin((a + b) / 2) * sin((a - b) / 2)"], None),
    ("double-angle-diff", True, ["cos(2 * a) == cos(a)**2 - sin(a)**2"], None),
    # --- multiple whole-period shifts (nested 2*pi layers) ---
    ("multi-period-literal", True, ["sin(9 * pi / 2) == 1"], None),
    ("multi-period-literal-wrong", False, ["sin(9 * pi / 2) == 0"], None),
    ("multi-period-interval", True, ["sin(a) == 3/5"],
     [["tan(a) == -3/4", "9 * pi / 2 < a", "a < 5 * pi"]]),
    # --- sqrt-valued tangent double angle ---
    ("tan-double-sqrt", True, ["tan(2 * a) == -sqrt(3)"],
     [["tan(a) == sqrt(3)", "pi / 4 < a", "a < pi / 2"]]),
    ("tan-double-sqrt-wrong", False, ["tan(2 * a) == sqrt(3)"],
     [["tan(a) == sqrt(3)", "pi / 4 < a", "a < pi / 2"]]),
    # --- literal tangent at a non-principal angle:
    #     the cosine-nonzero check proves cos(5*pi/4) != 0 through the decomposition,
    #     then the claim itself must still be right ---
    ("literal-tan-nonprincipal", True, ["tan(5 * pi / 4) == 1"], None),
    ("literal-tan-nonprincipal-wrong", False, ["tan(5 * pi / 4) == -1"], None),
    # --- inverse-trig literal special values ---
    ("inverse-literal-arcsin", True, ["arcsin(1/2) == pi / 6"], None),
    ("inverse-literal-arcsin-neg", True, ["arcsin(-1/2) == -pi / 6"], None),
    ("inverse-literal-arctan-one", True, ["arctan(1) == pi / 4"], None),
    ("inverse-literal-arctan-sqrt3", True, ["arctan(sqrt(3)) == pi / 3"], None),
    ("inverse-literal-wrong", False, ["arcsin(1/2) == pi / 3"], None),
    # --- tangent identity rewrites are exempt from the definedness check ---
    ("tan-identity-odd", True, ["tan(-a) == -tan(a)"], None),
    ("tan-identity-period", True, ["tan(a + pi) == tan(a)"], None),
    ("tan-identity-reflection", True, ["tan(pi - a) == -tan(a)"], None),
    ("tan-identity-def", True, ["tan(a) == sin(a) / cos(a)"], None),
    ("tan-identity-wrong", False, ["tan(-a) == tan(a)"], None),
    # --- arccos negative side literals (decomposition-backed value facts) ---
    ("inverse-literal-arccos-neg", True, ["arccos(-1/2) == 2 * pi / 3"], None),
    ("inverse-literal-arccos-neg-wrong", False, ["arccos(-1/2) == pi / 3"], None),
    # --- inverse compositions evaluated to numbers (kernel-checked roots) ---
    ("inverse-composition-cos-arcsin", True, ["cos(arcsin(3/5)) == 4/5"], None),
    ("inverse-composition-sin-arctan", True, ["sin(arctan(3/4)) == 3/5"], None),
    ("inverse-composition-wrong", False, ["cos(arcsin(3/5)) == 3/5"], None),
    # --- arctan premise pins the argument's exact value (tan_arctan is unconditional) ---
    ("arctan-premise-value", True, ["x == sqrt(3) / 3"], [["arctan(x) == pi / 6"]]),
    ("arctan-premise-neg", True, ["x == -1"], [["arctan(x) == -pi / 4"]]),
    ("arctan-premise-wrong", False, ["x == sqrt(3)"], [["arctan(x) == pi / 6"]]),
    # --- quadratic-irrational tangent double angle (exact Q(sqrt n) arithmetic) ---
    ("tan-double-quadratic", True, ["tan(2 * a) == -1"],
     [["tan(a) == 1 + sqrt(2)", "pi / 4 < a", "a < pi / 2"]]),
    ("tan-double-quadratic-wrong", False, ["tan(2 * a) == 1"],
     [["tan(a) == 1 + sqrt(2)", "pi / 4 < a", "a < pi / 2"]]),
    # --- the audited over-broad exemption:
    #     "some variable-angle trig on the other side" also matched value pins,
    #     so a premise pinning the angle at a singularity let HOL prove the totalized 0 == 0.
    #     The structural identity matcher must refuse these ---
    ("exemption-fp-tan-eq-sin", False, ["tan(a) == sin(b)"],
     [["a == pi / 2", "b == 0"]]),
    ("exemption-fp-tan-eq-cos", False, ["tan(a) == cos(a)"],
     [["a == pi / 2"]]),
    # the tan_double formula at a premise-pinned singular double:
    # both sides totalize to 0 (x/0 = 0 in HOL), so the formula shape must carry its conditions too
    ("tan-add-singular-fp", False, ["tan(2 * a) == 2 * tan(a) / (1 - tan(a)**2)"],
     [["a == pi / 4"]]),
    # integer periodicity beyond one pi stays a true exempt identity
    ("tan-identity-period-2pi", True, ["tan(a + 2 * pi) == tan(a)"], None),
    # --- canonical angle keys:
    #     bounds spelled 2*a join a value spelled a*2 (one linear form, one key), which no dump-keyed join ever connected ---
    ("spelling-commuted-angle-join", True, ["sin(a * 2) == 3/5"],
     [["tan(a * 2) == 3/4", "0 < 2 * a", "2 * a < pi / 2"]]),
]

# (name, expect, conclusions, premises, givens):
# cases whose singular pin arrives through a problem GIVEN, so the fresh name on the tangent side classifies as a DEFINITION.
# The audited bypass:
# `x == tan(a) and x == 0` splits into definition d0 `x = tan(a)` plus the tan-free residual claim `x = 0`, which the pre-fix check never conditioned;
# the assumption-side scan must fail it closed at the unprovable cos(pi/2) != 0.
GIVEN_CASES = [
    ("definition-bypass-fp", False, ["x == tan(a) and x == 0"], [[]],
     ["answer == n", "a == pi / 2"]),
]

# (name, expect, conclusions, premises, step index): multi-step chain cases.
CHAIN_CASES = [
    # An earlier FALSE conclusion consumed as an s* assumption:
    # step 0 admits tan(a) == 3/4 into the chain, step 1 sits in Q2 (where the tangent is negative) and claims the tan-consistent value.
    # The positive theorem proves ex falso from the poisoned chain;
    # the equal-strength guard twin must refute it, so the step must NOT verify.
    ("s-star-poisoned-tan", False,
     ["tan(a) == 3/4", "sin(a) == -3/5"],
     [[], ["pi / 2 < a", "a < pi"]], 1),
    # The audited cross-step bypass:
    # step 0's tangent-at-a-singularity claim is REJECTED,
    # but its translation still enters the chain as s0 (admission precedes verification), handing step 1 the HOL-pinned x = 0.
    # Step 1's assumption-side scan must demand cos(pi/2) != 0 and fail closed.
    ("chain-bypass-fp", False, ["x == tan(pi / 2)", "x == 0"], [[], []], 1),
    # An inherited identity rewrite (s0) and a nonzero own tangent premise are both exempt assumption-side,
    # so a legitimate chain still verifies without re-proving conditions.
    ("chain-identity-inherited", True,
     ["tan(-a) == -tan(a)", "sin(a) == 3/5"],
     [[], ["tan(a) == 3/4", "0 < a", "a < pi / 2"]], 1),
    # A nonzero tangent VALUE conclusion (s0) self-certifies its definedness in HOL,
    # so the next step consumes it as evidence without a fresh quadrant proof obligation.
    ("chain-tan-value-inherited", True,
     ["tan(a) == 3/4", "sin(a) == 3/5"],
     [["0 < a", "a < pi / 2"], ["0 < a", "a < pi / 2"]], 1),
]


def main():
    pool = server_pool.IsabelleServerPool(
        num_workers=3, base_dir=f"/tmp/isabelle_pool_trigmx_{os.getpid()}")
    pool.start()
    verify = PoolVerify(pool)
    failures = []
    def run_case(name, expect, verified):
        ok = verified is expect
        print(f"{name:28s} verified={verified!s:5s} expected={expect!s:5s} "
              f"{'OK' if ok else 'EXPECTATION FAILED'}", flush=True)
        if not ok:
            failures.append(name)

    try:
        for name, expect, conclusions, premises in CASES:
            run_case(name, expect, check_case(verify, conclusions, premises))
        for name, expect, conclusions, premises, givens in GIVEN_CASES:
            run_case(name, expect,
                     check_case(verify, conclusions, premises, givens=givens))
        for name, expect, conclusions, premises, index in CHAIN_CASES:
            run_case(name, expect,
                     check_case(verify, conclusions, premises, step_index=index))
    finally:
        pool.shutdown()
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print("ALL EXPECTATIONS HELD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
