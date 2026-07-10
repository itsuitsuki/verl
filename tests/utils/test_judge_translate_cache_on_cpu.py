"""CPU-only unit tests for the judge translation cache (judge.py). Mocks
call_judge so NO real judge is needed. Verifies the memory LRU + single-flight
+ disk layers that mirror the FOL shared_state cache: identical prompts (e.g.
the per-problem `givens`, byte-identical across 16 rollouts) collapse to ONE
judge call; failed parses are not cached; disk survives a cold memory cache."""

import threading
import time

from verl.utils.isabelle_utils import judge


def _reset():
    judge._TR_CACHE.clear()
    judge._TR_INFLIGHT.clear()
    judge._TR_STATS.update(hits=0, misses=0)


def _ok_parse(r):
    return r  # non-None -> success


def _ok_validate(_p):
    return []  # no errors -> success


def test_memory_cache_collapses_identical(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "T:" + prompt

    monkeypatch.setattr(judge, "call_judge", fake)
    r1 = judge.translate("PROB", _ok_parse, _ok_validate, judge_url="u", judge_model="m")
    r2 = judge.translate("PROB", _ok_parse, _ok_validate, judge_url="u", judge_model="m")
    assert r1[0] == "T:PROB" and r2[0] == "T:PROB"
    assert calls["n"] == 1               # 2nd call served from cache
    assert judge._TR_STATS["hits"] >= 1


def test_single_flight_concurrent(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def slow(prompt, thinking, **kw):
        calls["n"] += 1
        time.sleep(0.3)                  # let followers pile up on the leader
        return "T:" + prompt

    monkeypatch.setattr(judge, "call_judge", slow)
    out = []

    def work():
        out.append(judge.translate("SAME", _ok_parse, _ok_validate,
                                    judge_url="u", judge_model="m")[0])

    ts = [threading.Thread(target=work) for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(x == "T:SAME" for x in out)
    assert calls["n"] == 1               # 16 concurrent -> judge hit ONCE


def test_failed_translation_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "bad"

    monkeypatch.setattr(judge, "call_judge", fake)
    r1 = judge.translate("P", lambda r: None, _ok_validate, judge_url="u", judge_model="m")
    r2 = judge.translate("P", lambda r: None, _ok_validate, judge_url="u", judge_model="m")
    assert r1[0] is None and r2[0] is None
    assert calls["n"] >= 2               # failures not cached -> judge re-called


def test_disk_cache_survives_cold_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "D:" + prompt

    monkeypatch.setattr(judge, "call_judge", fake)
    judge.translate("DK", _ok_parse, _ok_validate, judge_url="u", judge_model="m")  # -> disk
    _reset()                             # cold memory cache (simulate new process)
    r = judge.translate("DK", _ok_parse, _ok_validate, judge_url="u", judge_model="m")
    assert r[0] == "D:DK"
    assert calls["n"] == 1               # 2nd came from DISK, no new judge call
