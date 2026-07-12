"""CPU-only test for the 2026-07-11 safe-mode reward semantics.

An UNDETERMINED consistency probe (dangerous premise chain, or a
timeout/worker_error) puts that step and every later step into SAFE MODE:
the with-premises proof is forbidden (it could be ex-falso), but an
INDEPENDENT premise-free proof still earns reward; a claim provable only from
the suspect premises earns nothing (symbol 'u'). A PROVEN inconsistency is a
HARD cut (symbol 'c', no rescue).

No Isabelle: a mock pool decides verdicts from the theorem text. Marker
numbers in the goal: 55 = provable premise-free (a tautology; verifies when
the theorem has NO assumes), 77 = provable only WITH premises (verifies when
the theorem HAS assumes). ALTERNATION contains '(eval)', so is_eval cannot
distinguish paths -- has_assumes is the only reliable discriminator."""
import re

import pytest

from verl.utils.isabelle_utils import engine as eng


class _MockPool:
    """A consistency probe for 0-based step k carries exactly k step-conclusion
    premises (s0..s(k-1)), so len({s\\d+}) == k pins the step. probe_by_k maps
    that count to a forced outcome; absent -> 'consistent'."""
    num_workers = 1

    def __init__(self, probe_by_k=None):
        self.probe_by_k = probe_by_k or {}
        self.claim_calls = []      # (has_assumes, code) for non-probe checks

    def submit(self, code):
        from concurrent.futures import Future
        f = Future()
        f.set_result(self.check(code))
        return f

    def check(self, code):
        if not re.search(r'shows\s+"False"', code):
            self.claim_calls.append(("assumes" in code, code))
        ok, extra = self._decide(code)
        r = {"success": ok, "elapsed": 0.01,
             "errors": [] if ok else ["no"],
             "queue_wait": 0.0, "check_time": 0.01}
        r.update(extra)
        return r

    def _decide(self, code):
        has_assumes = "assumes" in code
        if re.search(r'shows\s+"False"', code):        # consistency probe
            k = len(set(re.findall(r"\bs(\d+)\b", code)))
            oc = self.probe_by_k.get(k, "consistent")
            if oc == "inconsistent":
                return True, {}
            if oc == "undetermined":
                return False, {"worker_error": True, "undetermined": True}
            return False, {}
        goal = code.split("shows", 1)[1] if "shows" in code else code
        if "55" in goal:                                # premise-free tautology
            return (not has_assumes), {}
        if "77" in goal:                                # needs premises
            return has_assumes, {}
        return False, {}


def _run(pool, props, monkeypatch):
    fixes = [["a", "real"], ["b", "real"], ["c", "real"], ["d", "real"],
             ["answer", "real"]]
    givens = ["answer == a"]

    def mock_translate(prompt_base, parse_fn, validate_fn, *, judge_url,
                       judge_model, max_model_len=12288, soft_prefix=None,
                       **_kw):
        if "VARS line followed by GIVEN" in prompt_base:
            return ([tuple(f) for f in fixes], list(givens)), [{}], False
        parsed = {int(k): dict(v) for k, v in props.items()}
        return parsed, [{}], False

    monkeypatch.setattr(eng, "translate", mock_translate)
    # embed each prop's marker number in the conclusion text so the
    # transcription guard passes (and stray digits don't trip guard_invented).
    steps_xml = ""
    for k, v in props.items():
        num = "77" if "77" in v["prop"] else "55"
        steps_xml += (f"<step><premise>-</premise>"
                      f"<conclusion>value {num}</conclusion></step>")
    item = {"problem": "Numbers 55 and 77 and 5 appear.",
            "response": steps_xml + " \\boxed{55}",
            "ground_truth": "55", "dataset": "mock", "idx": 0, "sample": 0}
    return eng.process_one(0, item, pool, eng.IsabelleConfig())


def test_undetermined_enters_safe_mode_with_independent_rescue(monkeypatch):
    # probe at step index 1 is undetermined -> steps 1.. are safe mode. A
    # premise-free provable claim (55) still earns reward; one that needs the
    # suspect premises (77) is blocked ('u').
    props = {1: {"prop": "a == 55", "premises": []},   # k0 normal, bare
             2: {"prop": "b == 55", "premises": []},   # k1 safe, bare
             3: {"prop": "c == 55", "premises": []},   # k2 safe, bare
             4: {"prop": "d == 77", "premises": []}}   # k3 safe, needs premises
    rec = _run(_MockPool({1: "undetermined"}), props, monkeypatch)
    assert rec["steps_ok"]
    assert rec["premise_undetermined_at"] == 1
    assert rec["premise_inconsistent_at"] is None
    assert rec["pattern"] == "ooou"


def test_inconsistent_hard_cuts_all_subsequent(monkeypatch):
    # probe at step index 2 proves False -> steps 2.. are HARD cut ('c'), even
    # though k3's claim (55) would be independently provable.
    props = {1: {"prop": "a == 55", "premises": []},
             2: {"prop": "b == 55", "premises": []},
             3: {"prop": "c == 55", "premises": []},   # k2: >= latch -> c
             4: {"prop": "d == 55", "premises": []}}   # k3: bare-provable but c
    rec = _run(_MockPool({2: "inconsistent"}), props, monkeypatch)
    assert rec["premise_inconsistent_at"] == 2
    assert rec["pattern"] == "oocc"


def test_inconsistent_dominates_later_undetermined(monkeypatch):
    # inconsistency at k1 dominates an undetermined at k3: everything from k1 is
    # a hard 'c', the later undetermined boundary is masked.
    props = {1: {"prop": "a == 55", "premises": []},
             2: {"prop": "b == 55", "premises": []},
             3: {"prop": "c == 55", "premises": []},
             4: {"prop": "d == 55", "premises": []}}
    rec = _run(_MockPool({1: "inconsistent", 3: "undetermined"}), props,
               monkeypatch)
    assert rec["premise_inconsistent_at"] == 1
    assert rec["pattern"] == "occc"


def test_safe_mode_issues_no_with_premises_proof(monkeypatch):
    # 2026-07-11 review HIGH: in safe mode the nz loop and tolerance fallback
    # must NOT run with-premises proofs (ALTERNATION's linarith/presburger/auto
    # close any goal ex-falso from inconsistent premises). Assert that once a
    # step is in safe mode, EVERY claim-side prover call it makes is
    # premise-free (no 'assumes' in the theorem).
    props = {1: {"prop": "a == 55", "premises": []},   # k0 normal
             2: {"prop": "b == 77", "premises": []},   # k1 safe, needs-premises
             3: {"prop": "c == 77", "premises": []}}   # k2 safe, needs-premises
    pool = _MockPool({1: "undetermined"})
    rec = _run(pool, props, monkeypatch)
    assert rec["premise_undetermined_at"] == 1
    # k1,k2 need premises and are in safe mode -> blocked, no ex-falso reward
    assert rec["pattern"] == "ouu"
    # after safe mode begins (k1), a with-premises claim proof would only be
    # issued for a NORMAL step. Here only k0 is normal; k0's claim (55) is
    # bare-provable so its first with-premises attempt is allowed. The safe
    # steps (k1,k2) must issue ZERO with-premises claim checks.
    with_prem = [c for ha, c in pool.claim_calls if ha and '"77' in c.split("shows",1)[1]]
    assert with_prem == []      # no ex-falso path for the 77 (needs-premises) claims
