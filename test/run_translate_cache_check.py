"""Standalone (no-pytest) verification of the judge translation cache.
Mirrors tests/utils/test_judge_translate_cache_on_cpu.py but runnable with a
bare python. Replaces judge.call_judge manually. TEMP probe -> delete with ./test."""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.isabelle_utils import judge

_ORIG = judge.call_judge


def reset(tmp):
    os.environ["ISABELLE_TRANSLATE_CACHE_DIR"] = tmp
    judge._TR_CACHE.clear()
    judge._TR_INFLIGHT.clear()
    judge._TR_STATS.update(hits=0, misses=0)


ok_parse = lambda r: r
ok_validate = lambda p: []
fails = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


with tempfile.TemporaryDirectory() as tmp:
    # 1) memory cache collapses identical prompt
    reset(tmp)
    n = {"c": 0}
    judge.call_judge = lambda prompt, thinking, **kw: (n.__setitem__("c", n["c"] + 1) or ("T:" + prompt))
    r1 = judge.translate("PROB", ok_parse, ok_validate, judge_url="u", judge_model="m")
    r2 = judge.translate("PROB", ok_parse, ok_validate, judge_url="u", judge_model="m")
    print("test1 memory-cache")
    check("both parsed T:PROB", r1[0] == "T:PROB" and r2[0] == "T:PROB")
    check("judge called exactly once", n["c"] == 1)
    check("stats hit>=1", judge._TR_STATS["hits"] >= 1)

    # 2) single-flight: 16 concurrent identical -> 1 judge call
    reset(tmp)
    n = {"c": 0}

    def slow(prompt, thinking, **kw):
        n["c"] += 1
        time.sleep(0.3)
        return "T:" + prompt

    judge.call_judge = slow
    out = []
    ts = [threading.Thread(target=lambda: out.append(
        judge.translate("SAME", ok_parse, ok_validate, judge_url="u", judge_model="m")[0]))
        for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("test2 single-flight")
    check("all 16 got T:SAME", all(x == "T:SAME" for x in out) and len(out) == 16)
    check("judge called exactly once for 16 concurrent", n["c"] == 1)

    # 3) failed translation not cached
    reset(tmp)
    n = {"c": 0}
    judge.call_judge = lambda prompt, thinking, **kw: (n.__setitem__("c", n["c"] + 1) or "bad")
    r1 = judge.translate("P", lambda r: None, ok_validate, judge_url="u", judge_model="m")
    r2 = judge.translate("P", lambda r: None, ok_validate, judge_url="u", judge_model="m")
    print("test3 failure-not-cached")
    check("both None", r1[0] is None and r2[0] is None)
    check("judge re-called (>=2, not cached)", n["c"] >= 2)

    # 4) disk survives cold memory
    reset(tmp)
    n = {"c": 0}
    judge.call_judge = lambda prompt, thinking, **kw: (n.__setitem__("c", n["c"] + 1) or ("D:" + prompt))
    judge.translate("DK", ok_parse, ok_validate, judge_url="u", judge_model="m")
    judge._TR_CACHE.clear()          # cold MEMORY only; disk dir kept
    judge._TR_INFLIGHT.clear()
    r = judge.translate("DK", ok_parse, ok_validate, judge_url="u", judge_model="m")
    print("test4 disk-survives-cold-memory")
    check("got D:DK from disk", r[0] == "D:DK")
    check("no new judge call (disk hit)", n["c"] == 1)

judge.call_judge = _ORIG
print("\nRESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
