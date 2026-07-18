"""CPU-only tests for the judge translation cache.

call_judge is mocked, so no real judge is needed. The tests cover the memory LRU, concurrent identical-request deduplication, and disk cache. Identical prompts, such as one problem's givens across 16 rollouts, use one judge call. Failed translations are not cached, and disk entries survive a cold memory cache.
"""

import threading
import time

from verl.utils.isabelle_utils import translator


def _reset():
    translator._TR_CACHE.clear()
    translator._TR_PENDING.clear()
    translator._TR_STATS.update(hits=0, misses=0)


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

    monkeypatch.setattr(translator, "call_judge", fake)
    r1 = translator.translate("PROB", _ok_parse, _ok_validate, translator_url="u", translator_model="m")
    r2 = translator.translate("PROB", _ok_parse, _ok_validate, translator_url="u", translator_model="m")
    assert r1[0] == "T:PROB" and r2[0] == "T:PROB"
    assert calls["n"] == 1               # 2nd call served from cache
    assert translator._TR_STATS["hits"] >= 1


def test_concurrent_identical_requests_share_one_call(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def slow(prompt, thinking, **kw):
        calls["n"] += 1
        time.sleep(0.3)  # Allow the other callers to observe the pending request.
        return "T:" + prompt

    monkeypatch.setattr(translator, "call_judge", slow)
    out = []

    def work():
        out.append(translator.translate("SAME", _ok_parse, _ok_validate,
                                    translator_url="u", translator_model="m")[0])

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

    monkeypatch.setattr(translator, "call_judge", fake)
    r1 = translator.translate("P", lambda r: None, _ok_validate, translator_url="u", translator_model="m")
    r2 = translator.translate("P", lambda r: None, _ok_validate, translator_url="u", translator_model="m")
    assert r1[0] is None and r2[0] is None
    assert calls["n"] >= 2               # failures not cached -> judge re-called


def test_failed_shared_request_limits_judge_calls(monkeypatch, tmp_path):
    # Sixteen concurrent callers used to run separate three-attempt loops after the first request failed, producing 48 judge calls. The failure is now shared, with at most two shared attempts of MAX_TRIES calls each.
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}
    lock = threading.Lock()

    def failing(prompt, thinking, **kw):
        with lock:
            calls["n"] += 1
        time.sleep(0.05)
        return "unparseable"

    monkeypatch.setattr(translator, "call_judge", failing)
    outs = []

    def work():
        outs.append(translator.translate("FAILPROMPT", lambda r: None, _ok_validate,
                                    translator_url="u", translator_model="m")[0])

    ts = [threading.Thread(target=work) for _ in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(outs) == 16 and all(o is None for o in outs)
    assert calls["n"] <= 2 * translator.MAX_TRIES   # was 16 x MAX_TRIES pre-fix


def test_key_covers_parser_identity(monkeypatch, tmp_path):
    # 2026-07-11 fix: a cache hit must NOT bypass a DIFFERENT parser.
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "T:" + prompt

    monkeypatch.setattr(translator, "call_judge", fake)

    def parse_a(r):
        return r

    def parse_b(r):
        return r + ":B"

    r1 = translator.translate("P", parse_a, _ok_validate, translator_url="u", translator_model="m")
    r2 = translator.translate("P", parse_b, _ok_validate, translator_url="u", translator_model="m")
    assert r1[0] == "T:P" and r2[0] == "T:P:B"   # second parser actually ran
    assert calls["n"] == 2                        # different key -> no reuse


def test_key_covers_function_body(monkeypatch, tmp_path):
    # 2026-07-11 review: __qualname__ alone misses BODY edits -- two
    # same-named parsers with different code must not share a cache entry.
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "T:" + prompt

    monkeypatch.setattr(translator, "call_judge", fake)

    def parse_v1(r):
        return r + ":V1"

    def parse_v2(r):
        return r + ":V2"

    parse_v2.__qualname__ = parse_v1.__qualname__   # force identical names
    r1 = translator.translate("P", parse_v1, _ok_validate, translator_url="u", translator_model="m")
    r2 = translator.translate("P", parse_v2, _ok_validate, translator_url="u", translator_model="m")
    assert r1[0] == "T:P:V1" and r2[0] == "T:P:V2"  # v2 body actually ran
    assert calls["n"] == 2                           # digest split the keys


def test_fn_digest_ignores_line_shifts():
    # 2026-07-11 review round 2: marshal.dumps(code) serialized line-number
    # info, so ANY unrelated line shift in engine.py invalidated the whole
    # translate cache. The digest must depend on behavior only.
    src = "def f(x):\n    return x + 1\n"
    ns_a, ns_b, ns_c = {}, {}, {}
    exec(compile(src, "m", "exec"), ns_a)
    exec(compile("\n" * 50 + src, "m", "exec"), ns_b)      # shifted 50 lines
    exec(compile("def f(x):\n    return x + 2\n", "m", "exec"), ns_c)
    assert translator._fn_digest(ns_a["f"]) == translator._fn_digest(ns_b["f"])
    assert translator._fn_digest(ns_a["f"]) != translator._fn_digest(ns_c["f"])


def test_http_posts_counted_not_cache_markers(monkeypatch, tmp_path):
    # Cache metadata must not count as judge HTTP requests.
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    posts = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "T:ok"},
                                 "finish_reason": "stop"}]}

    def fake_post(url, json=None, timeout=None):
        posts["n"] += 1
        if posts["n"] == 1:
            raise OSError("first endpoint down")     # forces one retry
        return _Resp()

    monkeypatch.setattr(translator.requests, "post", fake_post)
    monkeypatch.setattr(translator.time, "sleep", lambda s: None)
    r = translator.translate("HTTPCOUNT", _ok_parse, _ok_validate,
                        translator_url="u", translator_model="m")
    assert r[0] == "T:ok"
    atts = r[1]
    assert atts[0]["http_posts"] == 2       # 1 failed + 1 successful post
    assert atts[0]["http_wall_time"] >= 0.0
    # cached second call: marker only, no http_posts key
    r2 = translator.translate("HTTPCOUNT", _ok_parse, _ok_validate,
                         translator_url="u", translator_model="m")
    assert r2[1][0].get("cache") == "mem"
    assert "http_posts" not in r2[1][0]


def test_disk_cache_survives_cold_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    _reset()
    calls = {"n": 0}

    def fake(prompt, thinking, **kw):
        calls["n"] += 1
        return "D:" + prompt

    monkeypatch.setattr(translator, "call_judge", fake)
    translator.translate("DK", _ok_parse, _ok_validate, translator_url="u", translator_model="m")  # -> disk
    _reset()                             # cold memory cache (simulate new process)
    r = translator.translate("DK", _ok_parse, _ok_validate, translator_url="u", translator_model="m")
    assert r[0] == "D:DK"
    assert calls["n"] == 1               # 2nd came from DISK, no new judge call
