"""LLM translation interface for the Isabelle verification pipeline.

Calls a vLLM endpoint to translate natural language into formal expressions.
Invalid output is returned to the model as feedback for a bounded retry.
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

from verl.utils.isabelle_utils import cache_lock, state_classes


def call_judge(prompt: str, thinking: bool, *,
               judge_url: str, judge_model: str,
               max_model_len: int = 12288,
               timeout: float = 240.0,
               stats: dict | None = None) -> str:
    # A comma-separated URL config denotes multiple vLLM translator endpoints. Choose one for each attempt to distribute concurrent requests. A retry may therefore use a different endpoint when one server is unavailable.
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
            if stats is not None:
                # Count actual HTTP requests. Cache reuse is recorded separately and must not contribute to judge load.
                stats["posts"] = stats.get("posts", 0) + 1
            _http_t0 = time.time()
            try:
                r = requests.post(f"{url}/chat/completions",
                                  json=payload, timeout=timeout)
            finally:
                if stats is not None:
                    stats["wall_time"] = (stats.get("wall_time", 0.0)
                                       + time.time() - _http_t0)
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


# The translation cache combines an in-memory LRU, concurrent identical-request deduplication, and typed JSON disk storage.
# All rollouts of one problem use the same givens prompt, so one caller translates it and the others reuse the result. Repeated step transcriptions can also reuse cached results.
# Failed translations are not cached.
_TR_CACHE_VERSION = os.environ.get("ISABELLE_TRANSLATE_CACHE_VERSION", "v1")
_TR_CACHE_MAX = int(os.environ.get("ISABELLE_TRANSLATE_CACHE_MAX", "100000"))
_TR_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_TR_LOCK = threading.Lock()
_TR_PENDING: dict = {}
_TR_STATS = {"hits": 0, "misses": 0}


_FN_DIGESTS: dict = {}


def _fn_digest(fn) -> str:
    """Return a digest of the function's fields about behavior and compilation.

    The digest includes bytecode, arity, flags, referenced names, and constants, recursively including nested code objects. It excludes source positions and line tables, so moving a function does not invalidate the translation cache. This specifically avoids the old `marshal.dumps(code)` behavior, where shifting an unchanged function by 50 source lines invalidated the entire translation cache. Function names alone are insufficient because a validator can retain its name while its rules change.

    Functions that share one code object but capture different closure values share a digest. This pipeline remains safe because captured values derive from the prompt, which is already part of the cache key.
    """
    co = getattr(fn, "__code__", None)
    if co is None:
        return repr(fn)
    d = _FN_DIGESTS.get(id(co))
    if d is None:
        try:
            import types
            h = hashlib.sha1()

            def _feed(c):
                h.update(c.co_code)
                h.update(str((c.co_argcount, c.co_kwonlyargcount,
                              c.co_flags)).encode())
                for tup in (c.co_names, c.co_varnames, c.co_freevars,
                            c.co_cellvars):
                    h.update(repr(tup).encode())
                for const in c.co_consts:
                    if isinstance(const, types.CodeType):
                        _feed(const)
                    else:
                        h.update(repr(const).encode())

            _feed(co)
            d = h.hexdigest()[:16]
        except Exception:  # noqa: BLE001
            d = "nodigest"
        _FN_DIGESTS[id(co)] = d
    return d


def _tr_key(prompt_base, judge_model, max_model_len, parse_fn, validate_fn,
            soft_prefix) -> str:
    """Return the cache key for every input that determines the parsed result.

    The key includes the prompt, model, token budget, parser and validator identity, compiled-code digests, and the soft-acceptance policy. Including function digests prevents a cache hit from bypassing changed parser or validator behavior.
    """
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
    """Typed, versioned, JSON-only disk schema; no pickle trust. Knows the
    two real translation shapes; anything else must round-trip plain JSON or
    it simply is not disk-cached."""
    if isinstance(parsed, state_classes.PyExprGiven):
        try:
            fixes = [[str(n), str(t)] for n, t in parsed.pyexpr_variable_types]
        except (TypeError, ValueError):
            return None
        # the on-disk key names predate the form naming and stay for compatibility with existing entries
        return {"schema": 2, "kind": "givens", "fixes": fixes,
                "givens": [str(g) for g in parsed.pyexpr_givens]}
    if (isinstance(parsed, dict) and parsed
            and all(isinstance(k, int) for k in parsed)
            and all(isinstance(v, state_classes.PyExprStep)
                    for v in parsed.values())):
        return {"schema": 2, "kind": "steps",
                "items": [[int(k), str(v.pyexpr_conclusion),
                           [str(x) for x in v.pyexpr_premises]]
                          for k, v in parsed.items()]}
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
            return state_classes.PyExprGiven(
                pyexpr_variable_types=[tuple(x) for x in payload["fixes"]],
                pyexpr_givens=list(payload["givens"]))
        if kind == "steps":
            return {int(k): state_classes.PyExprStep(
                        pyexpr_conclusion=p, pyexpr_premises=list(pr))
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
                        api_timeout: float = 240.0,
                        soft_prefix: str | None = None) -> tuple:
    """Call judge with retry loop. Returns (parsed, attempts, soft_flag)."""
    attempts, fb = [], None
    last_parsed, last_errors = None, []
    for t in range(MAX_TRIES):
        prompt = prompt_base if fb is None else prompt_base + fb
        t0 = time.time()
        http = {"posts": 0, "wall_time": 0.0}
        try:
            reply = call_judge(prompt, thinking=(t > 0),
                               judge_url=judge_url, judge_model=judge_model,
                               max_model_len=max_model_len,
                               timeout=api_timeout, stats=http)
        except Exception as e:
            attempts.append({"attempt": t, "fail": "judge_error",
                             "error": str(e)[:200],
                             "http_posts": http["posts"],
                             "http_wall_time": http["wall_time"]})
            break
        rec = {"attempt": t, "thinking": t > 0,
               "elapsed": time.time() - t0, "reply_head": reply[:1200],
               "http_posts": http["posts"], "http_wall_time": http["wall_time"]}
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
              translator_url: str, translator_model: str,
              max_model_len: int = 12288,
              api_timeout: float = 240.0,
              soft_prefix: str | None = None) -> tuple:
    """Translate with memory and disk caches and shared concurrent requests.

    The cache key covers parser/validator identity, token budget, and the soft
    acceptance policy, so a cache hit cannot bypass current validation logic.
    Concurrent callers for the same key share one translation attempt. If that
    attempt fails, one waiting caller retries; after two shared failures the
    remaining callers receive the failure instead of all calling the judge.
    Pending-state cleanup compares object identity, so an old owner cannot
    remove a replacement owner's state after an ABA transition. Only
    successful translations are cached.
    """
    key = _tr_key(prompt_base, translator_model, max_model_len,
                  parse_fn, validate_fn, soft_prefix)
    # Worst-case translation duration: MAX_TRIES attempts x
    # (2 HTTP posts x api_timeout + 3s inter-post wait), plus parse/validation.
    # Every waiting caller, cross-process stale-lock threshold, and
    # cross-process result wait must exceed this bound. The old fixed 900s
    # bound was shorter than the 1449s worst case at the 240s default, which
    # caused callers to duplicate work while a valid request was still running.
    # The additional 180s covers parse/validation, including validators that
    # issue bounded prover checks. Underestimating duplicates work;
    # overestimating only delays recovery after the owner exits.
    owner_budget = MAX_TRIES * (2.0 * api_timeout + 3.0) + 180.0
    failed_shared_attempts = 0
    while True:
        with _TR_LOCK:
            hit = _TR_CACHE.get(key)
            if hit is not None:
                _TR_CACHE.move_to_end(key)
                _TR_STATS["hits"] += 1
                parsed, soft = hit
                return parsed, [{"cache": "mem"}], soft
            pending = _TR_PENDING.get(key)
            owner = pending is None
            if owner:
                pending = {"event": threading.Event(), "result": None}
                _TR_PENDING[key] = pending

        if not owner:
            # A bounded wait prevents a stopped owner from blocking every caller.
            # The bound covers the valid worst case, so expiry means the owner
            # exited or stopped progressing. Remove only the same state object,
            # then loop so one waiting caller assumes responsibility.
            if not pending["event"].wait(owner_budget):
                with _TR_LOCK:
                    if _TR_PENDING.get(key) is pending:
                        _TR_PENDING.pop(key, None)
                continue
            result = pending["result"]
            if result is not None:
                parsed, soft = result
                return parsed, [{"cache": "shared"}], soft
            failed_shared_attempts += 1
            if failed_shared_attempts >= 2:
                # Two shared attempts failed for this exact prompt. Return that
                # failure instead of adding more judge requests.
                return None, [{"cache": "shared-failed"}], False
            continue

        out, cached_value = None, None
        try:
            disk = _tr_disk_load(key)
            if disk is not None:
                with _TR_LOCK:
                    _TR_CACHE[key] = disk
                    while len(_TR_CACHE) > _TR_CACHE_MAX:
                        _TR_CACHE.popitem(last=False)
                    _TR_STATS["hits"] += 1
                cached_value, out = disk, (disk[0], [{"cache": "disk"}],
                                           disk[1])
            else:
                # Deduplicate the first translation across reward-worker
                # processes. One process owns a lock beside the disk entry;
                # the others wait for its stored result. Every wait is bounded,
                # and a missing result falls through to local translation.
                lock = _tr_disk_path(key) + ".lock"
                owns_lock = (
                    not _tr_disk_enabled()
                    or cache_lock.acquire(
                        lock, stale_s=owner_budget + 60.0
                    )
                )
                if not owns_lock:
                    got = cache_lock.wait(
                        lock,
                        lambda: _tr_disk_load(key),
                        deadline_s=owner_budget,
                        poll_s=1.0,
                    )
                    if got is not None:
                        with _TR_LOCK:
                            _TR_CACHE[key] = got
                            while len(_TR_CACHE) > _TR_CACHE_MAX:
                                _TR_CACHE.popitem(last=False)
                            _TR_STATS["hits"] += 1
                        cached_value, out = (
                            got,
                            (got[0], [{"cache": "xproc"}], got[1]),
                        )
                if out is None:
                    with _TR_LOCK:
                        _TR_STATS["misses"] += 1
                    try:
                        parsed, attempts, soft = _translate_uncached(
                            prompt_base, parse_fn, validate_fn,
                            judge_url=translator_url, judge_model=translator_model,
                            max_model_len=max_model_len,
                            api_timeout=api_timeout, soft_prefix=soft_prefix)
                        out = (parsed, attempts, soft)
                        if parsed is not None:
                            cached_value = (parsed, soft)
                            with _TR_LOCK:
                                _TR_CACHE[key] = cached_value
                                while len(_TR_CACHE) > _TR_CACHE_MAX:
                                    _TR_CACHE.popitem(last=False)
                            _tr_disk_store(key, parsed, soft)
                        elif owns_lock and _tr_disk_enabled():
                            cache_lock.mark_failed(lock)
                    except BaseException:
                        if owns_lock and _tr_disk_enabled():
                            cache_lock.mark_failed(lock)
                        raise
                    finally:
                        if owns_lock and _tr_disk_enabled():
                            cache_lock.release(lock)
        finally:
            with _TR_LOCK:
                pending["result"] = cached_value
                pending["event"].set()
                if _TR_PENDING.get(key) is pending:
                    _TR_PENDING.pop(key, None)
        return out
