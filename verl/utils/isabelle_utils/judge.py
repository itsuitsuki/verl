"""Judge LLM interface for Isabelle translation pipeline.

Handles calling the judge model (vLLM endpoint) for NL-to-formal translation,
with retry + feedback loop.

Lifted from scripts/isabelle_poc_math500/pipeline_v3.py.
"""
import hashlib
import os
import pickle
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


def _tr_disk_enabled() -> bool:
    return os.environ.get("ISABELLE_TRANSLATE_DISK_CACHE", "1") not in ("0", "false", "False")


def _tr_disk_path(key: str) -> str:
    base = os.environ.get("ISABELLE_TRANSLATE_CACHE_DIR",
                          "/tmp/verl_isabelle_translate_cache")
    return os.path.join(base, _TR_CACHE_VERSION, key[:2], f"{key}.pkl")


def _tr_disk_load(key: str):
    if not _tr_disk_enabled():
        return None
    try:
        with open(_tr_disk_path(key), "rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError):
        return None


def _tr_disk_store(key: str, value) -> None:
    if not _tr_disk_enabled():
        return
    path = _tr_disk_path(key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except (OSError, pickle.PicklingError):
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
    + disk (mirrors the FOL shared_state cache). Key = (version, judge_model,
    prompt_base). Only SUCCESSFUL translations are cached; a failed/None parse
    is not cached and is retried normally."""
    key = hashlib.sha1(
        "\0".join((_TR_CACHE_VERSION, judge_model, prompt_base)).encode("utf-8")
    ).hexdigest()

    with _TR_LOCK:
        hit = _TR_CACHE.get(key)
        if hit is not None:
            _TR_CACHE.move_to_end(key)
            _TR_STATS["hits"] += 1
            parsed, soft = hit
            return parsed, [{"cache": "mem"}], soft
        inflight = _TR_INFLIGHT.get(key)
        owner = inflight is None
        if owner:
            inflight = {"event": threading.Event(), "result": None}
            _TR_INFLIGHT[key] = inflight

    if not owner:
        inflight["event"].wait()
        r = inflight["result"]
        if r is not None:
            parsed, soft = r
            return parsed, [{"cache": "flight"}], soft
        # leader's translation failed (uncacheable) -> compute once ourselves
        return _translate_uncached(
            prompt_base, parse_fn, validate_fn, judge_url=judge_url,
            judge_model=judge_model, max_model_len=max_model_len,
            soft_prefix=soft_prefix)

    out, cval = None, None
    try:
        disk = _tr_disk_load(key)
        if disk is not None:
            parsed, soft = disk
            with _TR_LOCK:
                _TR_CACHE[key] = disk
                while len(_TR_CACHE) > _TR_CACHE_MAX:
                    _TR_CACHE.popitem(last=False)
                _TR_STATS["hits"] += 1
            cval, out = disk, (parsed, [{"cache": "disk"}], soft)
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
                _tr_disk_store(key, cval)
    finally:
        with _TR_LOCK:
            inflight["result"] = cval
            inflight["event"].set()
            _TR_INFLIGHT.pop(key, None)
    return out
