"""Regression harness for engine.process_one verification semantics.

Runs process_one with a MOCKED judge (translate) and MOCKED prover (pool) on
scripted scenarios, dumping the full result records to JSON. Usage:

  # 1. capture baseline with the CURRENT (serial) engine:
  python regression_engine_verify.py --engine <path/engine.py> --out base.json
  # 2. after the parallel rewrite, require identical semantics:
  python regression_engine_verify.py --engine <path/new_engine.py> --compare base.json

The mock pool decides verdicts purely from the theorem TEXT (deterministic),
so serial and parallel execution must produce byte-identical records if the
rewrite is semantics-preserving. Scenarios cover: provable steps, unprovable
steps (x), no-premise fallback (r0), eval rescue, definition-only steps,
nz-divisor premises, premise-inconsistency latch, invented-constant guard.
TEMP harness -> delete with ./test."""
import argparse
import importlib.util
import json
import re
import sys


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine_uut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MockPool:
    """Deterministic prover: verdicts derived from theorem text only."""

    def __init__(self, rules):
        self.rules = rules          # scenario-specific flags
        self.calls = []

    def check(self, code):
        self.calls.append(code)
        ok = self._decide(code)
        return {"success": ok, "elapsed": 0.01, "errors": [] if ok else ["mock fail"]}

    def submit(self, code):
        """Synchronous stand-in for IsabelleServerPool.submit (2026-07-11
        review #1) so the regression exercises the engine's submit path;
        determinism is preserved (resolved before return)."""
        from concurrent.futures import Future
        fut = Future()
        try:
            fut.set_result(self.check(code))
        except Exception as e:  # noqa: BLE001 -- mirror dispatcher behavior
            fut.set_exception(e)
        return fut

    def _decide(self, code):
        premise_ids = set(re.findall(r"\bs(\d+)\b", code))
        has_assumes = "assumes" in code
        is_eval = "(eval)" in code
        is_approx = "approximation" in code

        # premise-consistency probe: goal False
        if '"False"' in code.replace("\\<open>", '"').replace("\\<close>", '"') \
                or re.search(r'shows\s+"False"', code):
            latch_at = self.rules.get("inconsistent_from")
            if latch_at is None:
                return False
            # premises include s0..s(k-1); k >= latch  <=>  s(latch-1) present
            return str(latch_at - 1) in premise_ids if latch_at > 0 else True

        # nz-divisor probe: goal "d != 0"
        if "noteq" in code or "\\<noteq>" in code:
            return True                      # divisors provably nonzero

        # claim goals, keyed by a marker number in the goal term
        goal = code.split("shows", 1)[1] if "shows" in code else code
        if "77" in goal:                     # provable WITH premises only
            return has_assumes and not is_eval
        if "88" in goal:                     # unprovable everywhere -> x
            return False
        if "99" in goal:                     # provable ONLY bare (r0 fallback)
            return (not has_assumes) and not is_eval and not is_approx
        if "66" in goal:                     # provable ONLY by eval rescue
            return is_eval
        return False


SCENARIOS = [
    {
        "name": "mixed_paths",
        "problem": "Numbers 77, 88, 99, 66 and 5 appear here. Compute answer.",
        "response": (
            "<step><premise>given</premise><conclusion>a is 77</conclusion></step>"
            "<step><premise>-</premise><conclusion>b is 88</conclusion></step>"
            "<step><premise>-</premise><conclusion>define t</conclusion></step>"
            "<step><premise>-</premise><conclusion>c is 99</conclusion></step>"
            "<step><premise>-</premise><conclusion>d is 66</conclusion></step>"
            " \\boxed{77}"),
        "gt": "77",
        "fixes": [["a", "real"], ["b", "real"], ["c", "real"], ["d", "real"],
                  ["answer", "real"]],
        "givens": ["answer == a"],
        "props": {1: {"prop": "a == 77", "premises": []},
                  2: {"prop": "b == 88", "premises": []},
                  3: {"prop": "t == a + 77", "premises": []},   # def-only (fresh t)
                  4: {"prop": "c == 99", "premises": []},
                  5: {"prop": "d == 66", "premises": []}},
        "rules": {},
    },
    {
        "name": "inconsistency_latch_at_2",
        "problem": "Numbers 77 and 5. Compute answer.",
        "response": (
            "<step><premise>-</premise><conclusion>s0 77</conclusion></step>"
            "<step><premise>-</premise><conclusion>s1 77</conclusion></step>"
            "<step><premise>-</premise><conclusion>s2 77</conclusion></step>"
            "<step><premise>-</premise><conclusion>s3 77</conclusion></step>"
            " \\boxed{77}"),
        "gt": "77",
        "fixes": [["a", "real"], ["b", "real"], ["c", "real"], ["e", "real"],
                  ["answer", "real"]],
        "givens": ["answer == a"],
        "props": {1: {"prop": "a == 77", "premises": []},
                  2: {"prop": "b == 77", "premises": []},
                  3: {"prop": "c == 77", "premises": []},
                  4: {"prop": "e == 77", "premises": []}},
        "rules": {"inconsistent_from": 2},
    },
    {
        "name": "guard_invented_constant",
        "problem": "Numbers 77 and 5 only. Compute answer.",
        "response": (
            "<step><premise>-</premise><conclusion>uses 999 invented</conclusion></step>"
            "<step><premise>-</premise><conclusion>fine 77</conclusion></step>"
            " \\boxed{77}"),
        "gt": "77",
        "fixes": [["a", "real"], ["b", "real"], ["answer", "real"]],
        "givens": ["answer == a"],
        # 999 not in problem/window -> guard_invented on step 1
        "props": {1: {"prop": "a == 77 + 999 - 999", "premises": []},
                  2: {"prop": "b == 77", "premises": []}},
        "rules": {},
    },
    {
        "name": "premises_and_nz",
        "problem": "Numbers 77, 5 and 2. Compute answer.",
        "response": (
            "<step><premise>x is 5</premise><conclusion>a from division 77</conclusion></step>"
            "<step><premise>-</premise><conclusion>b is 77</conclusion></step>"
            " \\boxed{77}"),
        "gt": "77",
        "fixes": [["a", "real"], ["b", "real"], ["x", "real"], ["answer", "real"]],
        "givens": ["x == 5", "answer == a"],
        "props": {1: {"prop": "a == 77 / (x - 2)", "premises": ["x == 5"]},
                  2: {"prop": "b == 77", "premises": []}},
        "rules": {},
    },
]


def run_scenarios(engine):
    out = {}
    for sc in SCENARIOS:
        def mock_translate(prompt_base, parse_fn, validate_fn, *, judge_url,
                           judge_model, max_model_len=12288, soft_prefix=None,
                           _sc=sc, **_kw):   # absorbs api_timeout etc.
            if "VARS line followed by GIVEN" in prompt_base:
                parsed = ([tuple(f) for f in _sc["fixes"]], list(_sc["givens"]))
                return parsed, [{"mock": "givens"}], False
            parsed = {int(k): dict(v) for k, v in _sc["props"].items()}
            return parsed, [{"mock": "steps"}], False

        engine.translate_orig = getattr(engine, "translate")
        engine.translate = mock_translate
        pool = MockPool(sc["rules"])
        config = engine.IsabelleConfig()
        item = {"problem": sc["problem"], "response": sc["response"],
                "ground_truth": sc["gt"], "dataset": "mock", "idx": 0,
                "sample": 0}
        try:
            rec = engine.process_one(0, item, pool, config)
        except Exception as e:  # noqa: BLE001 -- record, don't die
            rec = {"HARNESS_ERROR": repr(e)}
        finally:
            engine.translate = engine.translate_orig
        rec.pop("corrupt_info", None)
        out[sc["name"]] = {"rec": rec, "n_pool_calls": len(pool.calls)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out")
    ap.add_argument("--compare")
    args = ap.parse_args()

    engine = load_engine(args.engine)
    result = run_scenarios(engine)

    for name, r in result.items():
        rec = r["rec"]
        print(f"  {name:28s} pattern={rec.get('pattern','?'):8s} "
              f"n_steps={rec.get('n_steps')} pool_calls={r['n_pool_calls']} "
              f"latch={rec.get('premise_inconsistent_at')}"
              + (f"  ERROR={rec['HARNESS_ERROR']}" if "HARNESS_ERROR" in rec else ""))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True, default=str)
        print(f"BASELINE WRITTEN: {args.out}")
    if args.compare:
        with open(args.compare) as fh:
            base = json.load(fh)
        cur = json.loads(json.dumps(result, default=str))
        # pool-call COUNT may legitimately differ (parallel runs consistency
        # checks past the latch); semantics = the records themselves.
        for v in list(base.values()) + list(cur.values()):
            v.pop("n_pool_calls", None)
            # rec["prof"] is a wall-time profile (2026-07-11 review #6):
            # nondeterministic by nature, absent from older baselines.
            v.get("rec", {}).pop("prof", None)
            # premise_undetermined_at is only present when a probe was
            # undetermined (dangerous/timeout) -- absent in the clean baseline.
            v.get("rec", {}).pop("premise_undetermined_at", None)
            # DOCUMENTED EXCEPTION (2026-07-10): claim cascades are skipped
            # for steps at/after the premise-inconsistency point -- those
            # steps are forced to rewarded=False / pattern 'c' regardless, so
            # only the `verified`/`tolerance` DIAGNOSTIC fields can differ
            # there. Mask them on both sides; `rewarded` and the pattern are
            # still compared strictly.
            for e in (v.get("rec", {}).get("steps") or []):
                if e.get("premises_inconsistent") and not e.get("neutral"):
                    e["verified"] = "MASKED"
                    e.pop("tolerance", None)
        if cur == base:
            print("REGRESSION: IDENTICAL — semantics preserved")
        else:
            for k in base:
                if base[k] != cur.get(k):
                    print(f"REGRESSION MISMATCH in {k}:")
                    print("  base:", json.dumps(base[k], sort_keys=True)[:600])
                    print("  cur :", json.dumps(cur.get(k), sort_keys=True)[:600])
            sys.exit(1)
