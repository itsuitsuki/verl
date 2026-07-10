"""Judge LLM interface for Isabelle translation pipeline.

Handles calling the judge model (vLLM endpoint) for NL-to-formal translation,
with retry + feedback loop.

Lifted from scripts/isabelle_poc_math500/pipeline_v3.py.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import OrderedDict

import requests


def call_judge(prompt: str, thinking: bool, *,
               judge_url: str, judge_model: str,
               max_model_len: int = 12288) -> str:
    # judge_url may be a comma-separated list of endpoints (e.g. two vLLM judges
    # on separate GPUs). Pick one per attempt at random -- spreads load evenly
    # across judges with no shared counter (thread-safe), and a retry lands on a
    # different judge for free failover.
    _urls = [u.strip() for u in judge_url.split(",") if u.strip()] or [judge_url]
    want = 8192 if thinking else 6144
    est_in = len(prompt) // 3 + 300
    max_toks = max(1024, min(want, max_model_len - est_in))
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content":
             "You are a precise mathematical formalizer. Follow the output "
             "format exactly. No markdown fences, no commentary."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_toks,
    }
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    last = None
    for _ in range(2):
        try:
            url = random.choice(_urls)
            r = requests.post(f"{url}/chat/completions",
                              json=payload, timeout=240)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            reply = choice["message"]["content"]
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
            reply = re.sub(r"<think>.*", "", reply, flags=re.DOTALL)
            if choice.get("finish_reason") == "length":
                reply += "\n[TRUNCATED]"
            return reply
        except Exception as e:
            last = e
            time.sleep(3)
    raise last


def feedback(prev_reply: str, errors: list) -> str:
    errs = "\n".join(f"- {e[:300]}" for e in errors[:3])
    return (f"\n\nYour previous output was:\n{prev_reply[:2500]}\n\n"
            f"It was rejected:\n{errs}\n"
            "Output a corrected version in the SAME format.")


# --- Judge translation cache (mirrors formal_verify.py's FOL shared_state
# cache: in-memory LRU + single-flight + disk). The givens prompt is byte-
# identical across all 16 rollouts of one problem, so this collapses ~16
# identical judge translations into 1; repeated step transcriptions also hit.
# Only SUCCESSFUL translations are cached (a failed/None parse is retried). ---
_TR_CACHE_VERSION = os.environ.get("ISABELLE_TRANSLATE_CACHE_VERSION", "v1")
_TR_CACHE_MAX = int(os.environ.get("ISABELLE_TRANSLATE_CACHE_MAX", "100000"))
_TR_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_TR_LOCK = threading.Lock()
_TR_INFLIGHT: dict = {}
_TR_STATS = {"hits": 0, "misses": 0}


_FN_DIGESTS: dict = {}


def _fn_digest(fn) -> str:
    """Digest of a function's COMPILED CODE (marshal covers the bytecode,
    consts -- incl. nested code objects -- names and line info). Catches
    body edits that __qualname__ cannot: the same-named validator with
    changed rules would otherwise keep hitting stale cache entries
    (2026-07-11 review). Known limit: closures sharing one code object with
    different captured VALUES share a digest -- in this pipeline the
    captured values derive from the prompt itself, which is already in the
    key. Cached per code object (code objects are compile-time constants,
    never collected)."""
    co = getattr(fn, "__code__", None)
    if co is None:
        return repr(fn)
    d = _FN_DIGESTS.get(id(co))
    if d is None:
        try:
            import marshal
            d = hashlib.sha1(marshal.dumps(co)).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            d = "nodigest"
        _FN_DIGESTS[id(co)] = d
    return d


def _tr_key(prompt_base, judge_model, max_model_len, parse_fn, validate_fn,
            soft_prefix) -> str:
    """Cache identity covers EVERYTHING that determines the parsed result:
    prompt, model, token budget, parser/validator CODE identity (qualname +
    compiled-code digest), and the soft-accept policy. (2026-07-11: the old
    key was only (version, model, prompt) -- a hit silently bypassed the
    CURRENT parser and validator; qualname alone still missed body edits.)"""
    ident = "\0".join((
        _TR_CACHE_VERSION, judge_model, str(max_model_len),
        getattr(parse_fn, "__qualname__", repr(parse_fn)),
        _fn_digest(parse_fn),
        getattr(validate_fn, "__qualname__", repr(validate_fn)),
        _fn_digest(validate_fn),
        repr(soft_prefix), prompt_base,
    ))
    return hashlib.sha1(ident.encode("utf-8")).hexdigest()


def _tr_encode(parsed):
    """Typed, versioned, JSON-only disk schema -- no pickle trust. Knows the
    two real translation shapes; anything else must round-trip plain JSON or
    it simply is not disk-cached."""
    if (isinstance(parsed, (tuple, list)) and len(parsed) == 2
            and isinstance(parsed[0], (list, tuple))
            and isinstance(parsed[1], (list, tuple))
            and all(isinstance(x, (list, tuple)) and len(x) == 2
                    for x in parsed[0])):
        try:
            fixes = [[str(n), str(t)] for n, t in parsed[0]]
        except (TypeError, ValueError):
            return None
        return {"schema": 2, "kind": "givens", "fixes": fixes,
                "givens": [str(g) for g in parsed[1]]}
    if (isinstance(parsed, dict) and parsed
            and all(isinstance(k, int) for k in parsed)):
        try:
            items = [[int(k), str(v["prop"]),
                      [str(x) for x in v.get("premises", [])]]
                     for k, v in parsed.items()]
        except (KeyError, TypeError):
            return None
        return {"schema": 2, "kind": "steps", "items": items}
    try:
        if json.loads(json.dumps(parsed)) == parsed:
            return {"schema": 2, "kind": "json", "value": parsed}
    except (TypeError, ValueError):
        pass
    return None


def _tr_decode(payload):
    if not isinstance(payload, dict) or payload.get("schema") != 2:
        return None
    kind = payload.get("kind")
    try:
        if kind == "givens":
            return ([tuple(x) for x in payload["fixes"]],
                    list(payload["givens"]))
        if kind == "steps":
            return {int(k): {"prop": p, "premises": list(pr)}
                    for k, p, pr in payload["items"]}
        if kind == "json":
            return payload["value"]
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _tr_disk_enabled() -> bool:
    return os.environ.get("ISABELLE_TRANSLATE_DISK_CACHE", "1") not in ("0", "false", "False")


def _tr_disk_path(key: str) -> str:
    base = os.environ.get("ISABELLE_TRANSLATE_CACHE_DIR",
                          "/tmp/verl_isabelle_translate_cache")
    return os.path.join(base, _TR_CACHE_VERSION, key[:2], f"{key}.json")


def _tr_disk_load(key: str):
    """Return (parsed, soft) or None."""
    if not _tr_disk_enabled():
        return None
    try:
        with open(_tr_disk_path(key)) as fh:
            rec = json.load(fh)
        parsed = _tr_decode(rec.get("payload"))
        if parsed is None:
            return None
        return (parsed, bool(rec.get("soft", False)))
    except (OSError, ValueError):
        return None


def _tr_disk_store(key: str, parsed, soft) -> None:
    if not _tr_disk_enabled():
        return
    payload = _tr_encode(parsed)
    if payload is None:
        return
    path = _tr_disk_path(key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"payload": payload, "soft": bool(soft)}, fh)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        pass


def translate_cache_stats() -> dict:
    with _TR_LOCK:
        return dict(_TR_STATS)


MAX_TRIES = 3


def _translate_uncached(prompt_base: str, parse_fn, validate_fn, *,
                        judge_url: str, judge_model: str,
                        max_model_len: int = 12288,
                        soft_prefix: str | None = None) -> tuple:
    """Call judge with retry loop. Returns (parsed, attempts, soft_flag)."""
    attempts, fb = [], None
    last_parsed, last_errors = None, []
    for t in range(MAX_TRIES):
        prompt = prompt_base if fb is None else prompt_base + fb
        t0 = time.time()
        try:
            reply = call_judge(prompt, thinking=(t > 0),
                               judge_url=judge_url, judge_model=judge_model,
                               max_model_len=max_model_len)
        except Exception as e:
            attempts.append({"attempt": t, "fail": "judge_error",
                             "error": str(e)[:200]})
            break
        rec = {"attempt": t, "thinking": t > 0,
               "elapsed": time.time() - t0, "reply_head": reply[:1200]}
        if reply.endswith("[TRUNCATED]"):
            rec["fail"] = "truncated"
            fb = feedback(reply[-1500:], [
                "Your output hit the length limit and was cut off. Output "
                "ONLY the required lines, with shorter source snippets."])
            attempts.append(rec)
            continue
        parsed = parse_fn(reply)
        if parsed is None:
            rec["fail"] = "format"
            fb = feedback(reply, ["Output did not match the required "
                                  "line format."])
            attempts.append(rec)
            continue
        errors = validate_fn(parsed)
        last_parsed, last_errors = parsed, errors
        if errors:
            only_soft = (soft_prefix is not None
                         and all(e.startswith(soft_prefix) for e in errors))
            if only_soft and t >= 1:
                rec["fail"] = "soft_accept"
                rec["errors"] = [e[:200] for e in errors[:3]]
                attempts.append(rec)
                return parsed, attempts, True
            rec["fail"] = "validate"
            rec["errors"] = [e[:200] for e in errors[:3]]
            fb = feedback(reply, errors)
            attempts.append(rec)
            continue
        rec["fail"] = None
        attempts.append(rec)
        return parsed, attempts, False
    if (soft_prefix and last_parsed is not None and last_errors
            and all(e.startswith(soft_prefix) for e in last_errors)):
        return last_parsed, attempts, True
    return None, attempts, False


def translate(prompt_base: str, parse_fn, validate_fn, *,
              judge_url: str, judge_model: str,
              max_model_len: int = 12288,
              soft_prefix: str | None = None) -> tuple:
    """Cached wrapper over _translate_uncached: in-memory LRU + single-flight
    + typed-JSON disk (mirrors the FOL shared_state cache).

    2026-07-11 rework after external review:
    - the key covers parser/validator identity, token budget and soft policy
      (a hit no longer bypasses CURRENT validation logic silently);
    - single-flight failure is SHARED: concurrent callers of a failed flight
      elect ONE replacement leader; after two failed flights the remaining
      followers inherit the failure instead of stampeding the judge
      (16 followers x 3 retries = 48 calls observed pre-fix);
    - flight cleanup is identity-checked (an old owner can no longer pop a
      replacement owner's flight marker -- the ABA race);
    - only SUCCESSFUL translations are cached; failures are never cached."""
    key = _tr_key(prompt_base, judge_model, max_model_len,
                  parse_fn, validate_fn, soft_prefix)
    failed_flights = 0
    while True:
        with _TR_LOCK:
            hit = _TR_CACHE.get(key)
            if hit is not None:
                _TR_CACHE.move_to_end(key)
                _TR_STATS["hits"] += 1
                parsed, soft = hit
                return parsed, [{"cache": "mem"}], soft
            fl = _TR_INFLIGHT.get(key)
            owner = fl is None
            if owner:
                fl = {"event": threading.Event(), "result": None}
                _TR_INFLIGHT[key] = fl

        if not owner:
            # BOUNDED wait: a wedged leader must never deadlock its followers
            # (2026-07-08 step-135 lesson). The leader is itself bounded
            # (api_timeout x retries); on timeout drop the stale marker and
            # re-enter the flow (one waiter becomes the new leader).
            if not fl["event"].wait(900.0):
                with _TR_LOCK:
                    if _TR_INFLIGHT.get(key) is fl:
                        _TR_INFLIGHT.pop(key, None)
                continue
            r = fl["result"]
            if r is not None:
                parsed, soft = r
                return parsed, [{"cache": "flight"}], soft
            failed_flights += 1
            if failed_flights >= 2:
                # two shared flights already failed on this exact prompt --
                # inherit the failure rather than adding more judge load
                return None, [{"cache": "flight-failed"}], False
            continue    # loop: exactly one waiter becomes the retry leader

        out, cval = None, None
        try:
            disk = _tr_disk_load(key)
            if disk is not None:
                with _TR_LOCK:
                    _TR_CACHE[key] = disk
                    while len(_TR_CACHE) > _TR_CACHE_MAX:
                        _TR_CACHE.popitem(last=False)
                    _TR_STATS["hits"] += 1
                cval, out = disk, (disk[0], [{"cache": "disk"}], disk[1])
            else:
                with _TR_LOCK:
                    _TR_STATS["misses"] += 1
                parsed, attempts, soft = _translate_uncached(
                    prompt_base, parse_fn, validate_fn, judge_url=judge_url,
                    judge_model=judge_model, max_model_len=max_model_len,
                    soft_prefix=soft_prefix)
                out = (parsed, attempts, soft)
                if parsed is not None:
                    cval = (parsed, soft)
                    with _TR_LOCK:
                        _TR_CACHE[key] = cval
                        while len(_TR_CACHE) > _TR_CACHE_MAX:
                            _TR_CACHE.popitem(last=False)
                    _tr_disk_store(key, parsed, soft)
        finally:
            with _TR_LOCK:
                fl["result"] = cval
                fl["event"].set()
                if _TR_INFLIGHT.get(key) is fl:   # identity-checked (ABA)
                    _TR_INFLIGHT.pop(key, None)
        return out
