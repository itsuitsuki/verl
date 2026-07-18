"""Task #36 remainder: ring/field and the direct->general bridge through the REAL 35B judge.

Feeds handcrafted (problem, XML-step response) pairs to the exact production reward entry
(formal_verify._get_isabelle_engine + engine.verify_solution): judge translation -> parse/route ->
HOL-Algebra domain pools (quick_and_dirty=false) / Isa_Step general pool -> reward.
Group was already proven end-to-end through the real judge on dt1 (task #36 log); this closes the
ring/field remainder and exercises translate_steps rule 17 (the `order G` -> `order_G` bridge naming).

Hard criterion: the wrong-claim cases must NOT be rewarded (zero false positives).
Soft criterion: the positive algebra steps should be rewarded; a miss is a false negative and is
reported, not failed, because tactic reach on a given judge emission is a coverage question.

Run on datatech with both judges up:
  bash with_env.sh python -u e2e_36_algebra_judge.py --out /2022533109/zhouchuyan/verl/logs/e2e_val_20260717/case36
"""
import argparse
import json
from pathlib import Path

RING_PROBLEM = (
    "<Question>\nLet R be a commutative ring and let a and b be elements of R. "
    "Show that (-a) * b equals -(a * b), and then compute -(a * b) + a * b. "
    "If the result is the zero element of R, answer 0.\n</Question>\n\n"
    "Let's think step by step and output the final answer in \\boxed{}."
)

FIELD_PROBLEM = (
    "<Question>\nLet F be a field and let x be a nonzero element of F. "
    "Compute x * x^{-1}. If the result is the multiplicative identity of F, answer 1.\n</Question>\n\n"
    "Let's think step by step and output the final answer in \\boxed{}."
)

BRIDGE_PROBLEM = (
    "<Question>\nThe finite group G has exactly 15 elements. Every element of G except "
    "the identity element is colored red. How many red elements are there?\n</Question>\n\n"
    "Let's think step by step and output the final answer in \\boxed{}."
)

CASES = [
    {
        # Both steps are ring claims; step 2 restates step 1's conclusion as a premise, so the
        # same-family direct chain (previous_conclusions) is exercised through the real judge.
        "name": "ring_chain",
        "problem": RING_PROBLEM,
        "ground_truth": "0",
        "must_reward": {0, 1},
        "must_not_reward": set(),
        "response": (
            "<step>\n"
            "<premise>a is an element of the ring R.</premise>\n"
            "<premise>b is an element of the ring R.</premise>\n"
            "<conclusion>In the ring R, (-a) * b = -(a * b).</conclusion>\n"
            "</step>\n"
            "\n"
            "<step>\n"
            "<premise>a is an element of the ring R.</premise>\n"
            "<premise>b is an element of the ring R.</premise>\n"
            "<premise>In the ring R, (-a) * b = -(a * b).</premise>\n"
            "<conclusion>In the ring R, -(a * b) + a * b equals the zero element of R.</conclusion>\n"
            "</step>\n"
            "\n"
            "\\boxed{0}"
        ),
    },
    {
        "name": "field_inverse",
        "problem": FIELD_PROBLEM,
        "ground_truth": "1",
        "must_reward": {0},
        "must_not_reward": set(),
        "response": (
            "<step>\n"
            "<premise>x is an element of the field F.</premise>\n"
            "<premise>x is nonzero in the field F.</premise>\n"
            "<conclusion>In the field F, x * x^{-1} = 1, the multiplicative identity.</conclusion>\n"
            "</step>\n"
            "\n"
            "\\boxed{1}"
        ),
    },
    {
        # Step 0 should emit the whitelisted direct claim `order G = 15`; step 1 is ordinary
        # arithmetic that must receive it as the bridged pv_order_G premise (rule 17 naming).
        "name": "bridge_order",
        "problem": BRIDGE_PROBLEM,
        "ground_truth": "14",
        "must_reward": {1},
        "must_not_reward": set(),
        "response": (
            "<step>\n"
            "<premise>G is a finite group with exactly 15 elements.</premise>\n"
            "<conclusion>The order of the group G is 15.</conclusion>\n"
            "</step>\n"
            "\n"
            "<step>\n"
            "<premise>The order of the group G is 15.</premise>\n"
            "<premise>Exactly one element of G, the identity, is not red.</premise>\n"
            "<conclusion>There are 15 - 1 = 14 red elements.</conclusion>\n"
            "</step>\n"
            "\n"
            "\\boxed{14}"
        ),
    },
    {
        # Rule-17 probe with explicit group vocabulary: Lagrange gives the direct chain
        # (step 0 -> step 1, same group family) and step 1's whitelisted `order G = 15`
        # should bridge into step 2's general arithmetic as pv_order_G.
        "name": "bridge_lagrange",
        "problem": (
            "<Question>\nThe finite group G has a subgroup H with exactly 3 elements, and H "
            "has exactly 5 right cosets in G. Every element of G except the identity receives "
            "a coin. How many coins are handed out?\n</Question>\n\n"
            "Let's think step by step and output the final answer in \\boxed{}."
        ),
        "ground_truth": "14",
        "must_reward": {2},
        "must_not_reward": set(),
        "response": (
            "<step>\n"
            "<premise>G is a finite group.</premise>\n"
            "<premise>H is a subgroup of the group G.</premise>\n"
            "<conclusion>By Lagrange's theorem, the number of right cosets of H times the "
            "number of elements of H equals the order of G.</conclusion>\n"
            "</step>\n"
            "\n"
            "<step>\n"
            "<premise>The number of right cosets of H times the number of elements of H "
            "equals the order of G.</premise>\n"
            "<premise>H has exactly 3 elements.</premise>\n"
            "<premise>H has exactly 5 right cosets in G.</premise>\n"
            "<conclusion>The order of the group G is 15.</conclusion>\n"
            "</step>\n"
            "\n"
            "<step>\n"
            "<premise>The order of the group G is 15.</premise>\n"
            "<premise>Only the identity element receives no coin, so exactly 1 element "
            "receives no coin.</premise>\n"
            "<conclusion>The number of coins handed out is 15 - 1 = 14.</conclusion>\n"
            "</step>\n"
            "\n"
            "\\boxed{14}"
        ),
    },
    {
        # Structure-naming mechanism probe: identical to field_inverse but the problem calls
        # the field R, matching the locale's structure variable. If this proves while
        # field_inverse (field F) stays not_proved, the F-vs-R naming mismatch is confirmed.
        "name": "field_inverse_R",
        "problem": (
            "<Question>\nLet R be a field and let x be a nonzero element of R. "
            "Compute x * x^{-1}. If the result is the multiplicative identity of R, answer 1.\n</Question>\n\n"
            "Let's think step by step and output the final answer in \\boxed{}."
        ),
        "ground_truth": "1",
        "must_reward": {0},
        "must_not_reward": set(),
        "response": (
            "<step>\n"
            "<premise>x is an element of the field R.</premise>\n"
            "<premise>x is nonzero in the field R.</premise>\n"
            "<conclusion>In the field R, x * x^{-1} = 1, the multiplicative identity.</conclusion>\n"
            "</step>\n"
            "\n"
            "\\boxed{1}"
        ),
    },
    {
        # FALSE-POSITIVE probe: (-a) * b = a * b is not a theorem of commutative rings.
        "name": "ring_wrong_claim",
        "problem": RING_PROBLEM,
        "ground_truth": "0",
        "must_reward": set(),
        "must_not_reward": {0},
        "response": (
            "<step>\n"
            "<premise>a is an element of the ring R.</premise>\n"
            "<premise>b is an element of the ring R.</premise>\n"
            "<conclusion>In the ring R, (-a) * b = a * b.</conclusion>\n"
            "</step>\n"
            "\n"
            "\\boxed{0}"
        ),
    },
    {
        # FALSE-POSITIVE probe: x * x^{-1} = 0 is false in every field.
        "name": "field_wrong_claim",
        "problem": FIELD_PROBLEM,
        "ground_truth": "1",
        "must_reward": set(),
        "must_not_reward": {0},
        "response": (
            "<step>\n"
            "<premise>x is an element of the field F.</premise>\n"
            "<premise>x is nonzero in the field F.</premise>\n"
            "<conclusion>In the field F, x * x^{-1} = 0, the zero element.</conclusion>\n"
            "</step>\n"
            "\n"
            "\\boxed{1}"
        ),
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", default="http://127.0.0.1:4873/v1,http://127.0.0.1:4874/v1")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Production api_config (math_step_gdpo_isabelle_combined.sh) except pool_workers,
    # a per-process sizing knob; 2 is enough for these single-response cases.
    api_config = {
        "base_url": args.judges,
        "model": "Qwen3.6-35B-A3B",
        "timeout": 60,
        "api_timeout": 200,
        "isabelle_pool_workers": 2,
        "isabelle_worker_rss_cap_gb": 12,
    }
    from verl.utils.reward_score.formal_verify import _get_isabelle_engine
    engine = _get_isabelle_engine(api_config)

    fp_failures = []
    for k, case in enumerate(CASES):
        rec = engine.verify_solution(
            problem=case["problem"], response=case["response"],
            ground_truth=case["ground_truth"], dataset="e2e36", idx=k, sample=0)
        (outdir / ("%02d_%s.json" % (k, case["name"]))).write_text(
            json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        rewarded = sorted(s["step"] for s in rec.get("steps", []) if s.get("rewarded"))
        reasons = {s["step"]: s.get("domain_reason") for s in rec.get("steps", [])
                   if s.get("domain_reason") is not None}
        print("[%s] pattern=%r rewarded=%s domain_reasons=%s format_ok=%s givens_ok=%s steps_ok=%s"
              % (case["name"], rec.get("pattern"), rewarded, reasons,
                 rec.get("format_ok"), rec.get("givens_ok"), rec.get("steps_ok")), flush=True)
        bad = case["must_not_reward"] & set(rewarded)
        missing = case["must_reward"] - set(rewarded)
        if bad:
            fp_failures.append("%s: rewarded wrong-claim steps %s" % (case["name"], sorted(bad)))
        if missing:
            print("  NOTE false negative (acceptable, coverage question): steps %s not rewarded"
                  % sorted(missing), flush=True)

    print("=" * 60, flush=True)
    if fp_failures:
        for f in fp_failures:
            print("FP-FAIL:", f, flush=True)
        print("RESULT: FALSE POSITIVES PRESENT", flush=True)
        raise SystemExit(1)
    print("RESULT: NO FALSE POSITIVES (see per-case notes for coverage)", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
