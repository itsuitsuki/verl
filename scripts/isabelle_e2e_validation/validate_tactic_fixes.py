"""Real-pool validation for the tactic dispatch and ALTERNATION fixes (2026-07-18).

Positive/negative theorem pairs for each changed proof path, run against the real Isa_Step
pool. Above all this validates that the close-or-fail ALTERNATION syntax (`simp; fail` inside
an `|` chain) parses and behaves: a branch that only rewrites must fail cleanly so a LATER
branch still closes the goal, and a false goal must stay unproved.

Run on a node with the Isa_Step heap:
    bash scripts/isabelle_e2e_validation/with_env.sh python -u scripts/isabelle_e2e_validation/validate_tactic_fixes.py
Exit code 0 = every expectation held.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl.utils.isabelle_utils import (pyexpr, server_pool, state_classes,  # noqa: E402
                                       tactics, theorem_builders)


def build_cases():
    mk, mkl = theorem_builders.make_theorem, theorem_builders.make_theorem_with_logs
    fx = [("pv_x", "real")]
    fi = [("pv_n", "int")]
    cases = []

    # -- ALTERNATION close-or-fail semantics --
    cases.append(("alt-simp-closes", mk([], [], "(2::real) + 2 = 4", tactics.ALTERNATION), "PROVED"))
    # linarith-only goal: under the old chain a progress-making simp could mask it
    cases.append(("alt-linarith-goal", mk(fx, [("g0", "(pv_x) > (3::real)")],
                                          "(pv_x) + 1 > (4::real)", tactics.ALTERNATION), "PROVED"))
    # presburger-only goal (parity), int carrier
    cases.append(("alt-presburger-goal", mk(fi, [], "(pv_n) mod 2 = 0 \\<or> (pv_n) mod 2 = 1",
                                            tactics.ALTERNATION), "PROVED"))
    # the last branch (floor rules) must be reachable
    cases.append(("alt-floor-goal", mk([], [], "floor ((7::real) / 2) = 3", tactics.ALTERNATION), "PROVED"))
    cases.append(("alt-false-goal", mk([], [], "(2::real) + 2 = 5", tactics.ALTERNATION), "UNPROVED"))

    # -- equal-strength guard twin: the SAME theorem shape against False --
    cases.append(("guard-contradiction", mk(fx, [("g0", "(pv_x) = (1::real)"), ("g1", "(pv_x) = (2::real)")],
                                            "False", tactics.ALTERNATION), "PROVED"))
    cases.append(("guard-consistent", mk(fx, [("g0", "(pv_x) = (1::real)")],
                                         "False", tactics.ALTERNATION), "UNPROVED"))

    # -- numeral log facts (literal applications only) --
    cases.append(("log-honest", mkl([], [], "(log (2::real) (8::real)) = (3::real)",
                                    tactics.ALTERNATION), "PROVED"))
    cases.append(("log-wrong", mkl([], [], "(log (2::real) (8::real)) = (4::real)",
                                   tactics.ALTERNATION), "UNPROVED"))
    # the wrong-log-premise ex-falso hole: the guard twin CARRIES the log facts and refutes
    cases.append(("log-guard-refutes", mkl([], [("g0", "(log (2::real) (8::real)) = (4::real)")],
                                           "False", tactics.ALTERNATION), "PROVED"))

    # -- oriented integral --
    def integral_case(src, expect):
        node = pyexpr.parse_expr(src)
        goal = pyexpr.transpile(node, {}, "real")
        theorem = theorem_builders.integral_recipe(node, goal)
        assert theorem is not None, f"recipe did not build for {src}"
        return theorem, expect
    t, e = integral_case("integral(x, x, 2, 5) == 21/2", "PROVED")
    cases.append(("integral-forward", t, e))
    t, e = integral_case("integral(x, x, 5, 2) == -21/2", "PROVED")
    cases.append(("integral-reversed-oriented", t, e))
    t, e = integral_case("integral(x, x, 5, 2) == 0", "UNPROVED")
    cases.append(("integral-reversed-not-zero", t, e))
    # symbolic bound pi: the conditional oriented form must still resolve under simp (0 <= pi)
    t, e = integral_case("integral(sin(x), x, 0, pi) == 2", "PROVED")
    cases.append(("integral-symbolic-pi", t, e))
    return cases


def main():
    cases = build_cases()
    pool = server_pool.IsabelleServerPool(
        num_workers=2, base_dir=f"/tmp/isabelle_pool_tacfix_{os.getpid()}")
    pool.start()
    failures = []
    try:
        for name, theorem, expect in cases:
            outcome = state_classes.VerificationOutcome.from_raw(pool.check(theorem)).outcome.name
            ok = outcome == expect
            print(f"{name:28s} {outcome:9s} expected={expect:9s} {'OK' if ok else 'EXPECTATION FAILED'}",
                  flush=True)
            if not ok:
                failures.append(name)
    finally:
        pool.shutdown()
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print("ALL EXPECTATIONS HELD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
