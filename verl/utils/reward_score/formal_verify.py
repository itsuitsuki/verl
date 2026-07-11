"""
Unified formal-verification step reward (dispatch layer).

This module is the single entry point for all formal-verification step
rewards. It dispatches on ``fol_task_type`` to one of three backends
(the ``fol_*`` config prefix is historical: "fol" here means
"formal-verification step reward", NOT specifically First-Order Logic):

  - ``logic``    -> Z3/SMT  via FOLEngine       (verl.utils.fol_utils)
  - ``math_z3``  -> Z3/SMT  via FOLEngine       (deprecated Z3 math path)
  - ``math``     -> Isabelle via IsabelleEngine (verl.utils.isabelle_utils)

The Z3 path verifies by entailment (premises & NOT(conclusion) UNSAT -> 1.0);
the Isabelle path verifies a whole solution per-step (see isabelle_utils).

Configurable via api_config keys (``fol_*`` prefix kept for backward
compatibility with existing training scripts and past W&B runs):
  - fol_preprocess: "direct" (default) | "structured"   [Z3 only]
  - fol_translation: "implication" (default) | "assertion"  [Z3 only]
  - fol_task_type: "logic" (default) | "math" (Isabelle) | "math_z3" (Z3, deprecated)
  - max_tries: int (default 1), used by declaration/expression repair
  - old_max_tries: int (default 0), used by whole-code correction
  - timeout: float (default 30.0 Z3 / 60.0 Isabelle) -- per-verification deadline
  - fol_cumulative_mode: "current_only" (default) | "step" | "dependency_graph"
  - isabelle_pool_workers: int (Isabelle path only)

Exports:
  - check_step_format_fol
  - compute_step_reward_format_fol
  - compute_step_reward_fol             (Z3 path, per-step)
  - prepare_fol_shared_state            (Z3 path)
  - compute_solution_reward_isabelle    (Isabelle path, whole-solution)
"""

import atexit
import fcntl
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from hashlib import sha1
# import threading
# import time

from verl.utils.fol_utils.common import check_step_format_fol, extract_fol_problem, parse_step_tags
from verl.utils.fol_utils.engine import (
    FOLConfig,
    FOLEngine,
    PreprocessPipeline,
    TaskType,
    TranslationMode,
)

logger = logging.getLogger(__name__)

_PREMISE_PATTERN = re.compile(r"<premise>(.*?)</premise>", re.DOTALL)
_CONCLUSION_PATTERN = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)

_FOL_SHARED_STATE_CACHE_MAX_SIZE = max(1, int(os.environ.get("FOL_SHARED_PREPROCESS_CACHE_SIZE", "512")))
_FOL_SHARED_STATE_DISK_CACHE_VERSION = os.environ.get("FOL_SHARED_PREPROCESS_DISK_CACHE_VERSION", "v2")
_fol_shared_state_cache: OrderedDict[tuple, dict] = OrderedDict()
_fol_shared_state_cache_lock = threading.Lock()
_fol_shared_state_inflight: dict[tuple, dict] = {}
_FOL_VERIFY_CACHE_MAX_SIZE = max(1, int(os.environ.get("FOL_VERIFY_CACHE_SIZE", "4096")))
_FOL_VERIFY_DISK_CACHE_VERSION = os.environ.get("FOL_VERIFY_DISK_CACHE_VERSION", "v2")
_fol_verify_cache: OrderedDict[tuple, float] = OrderedDict()
_fol_verify_cache_lock = threading.Lock()
_fol_verify_inflight: dict[tuple, dict] = {}
_fol_verify_cache_stats = {"hits": 0, "misses": 0}
_fol_verify_cache_stats_lock = threading.Lock()
_fol_verify_cache_step_stats: OrderedDict[int, dict[str, int]] = OrderedDict()
_fol_verify_cache_summary_registered = False


def _normalize_cumulative_mode(mode: str | None) -> str:
    """Normalize cumulative input construction mode."""
    normalized = str(mode or "").strip().lower().replace("-", "_")
    if not normalized:
        return "current_only"
    if normalized in {"current", "current_only", "none", "off", "false", "0"}:
        return "current_only"
    if normalized in {"step", "steps", "all", "all_previous", "all_ancestors"}:
        return "step"
    if normalized in {"dependency", "dependencies", "dependency_graph", "graph"}:
        return "dependency_graph"
    return "current_only"


def _build_fol_config(api_config: dict | None = None) -> FOLConfig:
    """Construct a FOLConfig from reward API config."""
    cfg = dict(api_config or {})
    cumulative_mode = _normalize_cumulative_mode(cfg.get("fol_cumulative_mode"))
    cfg["fol_cumulative_mode"] = cumulative_mode
    cfg["cumulative"] = cumulative_mode != "current_only"

    try:
        preprocess = PreprocessPipeline(cfg.get("fol_preprocess", "direct"))
    except ValueError:
        preprocess = PreprocessPipeline.DIRECT

    try:
        translation = TranslationMode(cfg.get("fol_translation", "implication"))
    except ValueError:
        translation = TranslationMode.IMPLICATION

    try:
        task_type = TaskType(cfg.get("fol_task_type", "logic"))
    except ValueError:
        task_type = TaskType.LOGIC

    return FOLConfig(
        preprocess=preprocess,
        translation=translation,
        task_type=task_type,
        max_tries=int(cfg.get("max_tries", 1)),
        old_max_tries=int(cfg.get("old_max_tries", 0)),
        timeout=float(cfg.get("timeout", 30.0)),
        cumulative=cfg["cumulative"],
        api_config=cfg,
    )


def _as_bool(value, default: bool = False) -> bool:
    """Parse common bool-like config values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _shared_state_disk_cache_enabled(api_config: dict | None) -> bool:
    """Whether to share declaration/preprocess cache across worker processes."""
    cfg = api_config or {}
    if "fol_shared_state_disk_cache" in cfg:
        return _as_bool(cfg.get("fol_shared_state_disk_cache"), True)
    return _as_bool(os.environ.get("FOL_SHARED_PREPROCESS_DISK_CACHE"), True)


def _shared_state_disk_cache_dir(api_config: dict | None) -> str:
    """Return the on-disk cache directory for FOL shared preprocessing."""
    cfg = api_config or {}
    return str(
        cfg.get("fol_shared_state_cache_dir")
        or os.environ.get("FOL_SHARED_PREPROCESS_DISK_CACHE_DIR")
        or "/tmp/verl_fol_shared_preprocess_cache"
    )


def _shared_state_disk_cache_paths(cache_key: tuple, api_config: dict | None) -> tuple[str, str]:
    """Return data and lock paths for one shared-state cache key."""
    key_text = json.dumps(
        [_FOL_SHARED_STATE_DISK_CACHE_VERSION, cache_key],
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    digest = sha1(key_text.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(_shared_state_disk_cache_dir(api_config), digest[:2])
    return os.path.join(cache_dir, f"{digest}.json"), os.path.join(cache_dir, f"{digest}.lock")


def _load_shared_state_from_disk(
    cache_key: tuple,
    fol_config: FOLConfig,
    api_config: dict | None,
) -> dict | None:
    """Load a completed shared state from the cross-process cache."""
    if not _shared_state_disk_cache_enabled(api_config):
        return None
    data_path, _ = _shared_state_disk_cache_paths(cache_key, api_config)
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed_context = data.get("processed_context")
        declarations = data.get("declarations")
        if not isinstance(processed_context, str) or not isinstance(declarations, str) or not declarations:
            return None
        return {
            "config": fol_config,
            "processed_context": processed_context,
            "declarations": declarations,
        }
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Failed to read FOL shared-state disk cache: %s", exc)
        return None


def _store_shared_state_to_disk(
    cache_key: tuple,
    shared_state: dict | None,
    api_config: dict | None,
) -> None:
    """Store a successful shared state in the cross-process cache."""
    if shared_state is None or not _shared_state_disk_cache_enabled(api_config):
        return
    data_path, _ = _shared_state_disk_cache_paths(cache_key, api_config)
    tmp_path = f"{data_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    payload = {
        "version": _FOL_SHARED_STATE_DISK_CACHE_VERSION,
        "processed_context": shared_state["processed_context"],
        "declarations": shared_state["declarations"],
    }
    try:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, data_path)
    except Exception as exc:
        logger.warning("Failed to write FOL shared-state disk cache: %s", exc)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class _SharedStateDiskLock:
    """Per-key file lock used to prevent cross-process declaration stampedes."""

    def __init__(self, cache_key: tuple, api_config: dict | None):
        self.enabled = _shared_state_disk_cache_enabled(api_config)
        self._file = None
        _, self._lock_path = _shared_state_disk_cache_paths(cache_key, api_config)

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
            self._file = open(self._lock_path, "a+", encoding="utf-8")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        except Exception as exc:
            logger.warning("Failed to lock FOL shared-state disk cache: %s", exc)
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
            self._file = None
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is None:
            return False
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
        return False


def _normalize_step_text(text: str) -> str:
    """Normalize natural-language step text for conservative duplicate checks."""
    return " ".join(str(text).strip().lower().split())


def _has_student_premise_conclusion_duplicate(step_text: str) -> bool:
    """Whether actor copied the current conclusion verbatim into a premise."""
    premises = [_normalize_step_text(item) for item in _PREMISE_PATTERN.findall(step_text or "")]
    conclusions = [_normalize_step_text(item) for item in _CONCLUSION_PATTERN.findall(step_text or "")]
    conclusions = [item for item in conclusions if item]
    if not premises or not conclusions:
        return False
    premise_set = {item for item in premises if item}
    return any(conclusion in premise_set for conclusion in conclusions)


def _deduplicate_text_items(items: list[str]) -> list[str]:
    """Remove exact duplicate text items while preserving first-seen order."""
    seen = set()
    deduped = []
    for item in items:
        key = _normalize_step_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item.strip())
    return deduped


def _parse_cumulative_step_history(step_history: list[str]) -> list[dict[str, object]]:
    """Parse step history into premise/conclusion records."""
    parsed_steps = []
    for step in step_history:
        tags = parse_step_tags(step)
        premises = [str(item).strip() for item in tags.get("premises", []) if str(item).strip()]
        conclusion = str(tags.get("conclusion") or "").strip()
        parsed_steps.append({"premises": premises, "conclusion": conclusion})
    return parsed_steps


def _dependency_ancestor_indices(parsed_steps: list[dict[str, object]], current_idx: int) -> set[int]:
    """Find dependency ancestors using exact normalized premise-to-conclusion matches."""
    conclusion_key_to_indices: dict[str, list[int]] = {}
    edges: dict[int, set[int]] = {idx: set() for idx in range(len(parsed_steps))}

    for idx, step in enumerate(parsed_steps):
        for premise in step.get("premises", []):
            premise_key = _normalize_step_text(str(premise))
            if not premise_key:
                continue
            for parent_idx in conclusion_key_to_indices.get(premise_key, []):
                edges[idx].add(parent_idx)

        conclusion_key = _normalize_step_text(str(step.get("conclusion") or ""))
        if conclusion_key:
            conclusion_key_to_indices.setdefault(conclusion_key, []).append(idx)

    ancestors = set()

    def visit(idx: int) -> None:
        for parent_idx in edges.get(idx, set()):
            if parent_idx in ancestors:
                continue
            ancestors.add(parent_idx)
            visit(parent_idx)

    visit(current_idx)
    return ancestors


def _build_source_separated_cumulative_input(step_history: list[str], *, mode: str = "step") -> tuple[str, dict]:
    """Build compact cumulative FOL input with explicit source separation."""
    if not step_history:
        return "", {}

    parsed_steps = _parse_cumulative_step_history(step_history)
    current_idx = len(parsed_steps) - 1
    current_step = parsed_steps[current_idx]
    current_conclusion = str(current_step.get("conclusion") or "").strip()
    current_premises = _deduplicate_text_items([str(item) for item in current_step.get("premises", [])])
    if not current_conclusion:
        return "", {}

    cumulative_mode = _normalize_cumulative_mode(mode)
    if cumulative_mode == "dependency_graph":
        previous_indices = sorted(
            idx for idx in _dependency_ancestor_indices(parsed_steps, current_idx)
            if idx < current_idx
        )
    else:
        previous_indices = list(range(current_idx))

    previous_conclusions = [
        str(parsed_steps[idx].get("conclusion") or "").strip()
        for idx in previous_indices
        if str(parsed_steps[idx].get("conclusion") or "").strip()
    ]
    previous_conclusions = _deduplicate_text_items(previous_conclusions)

    lines = ["Source-Separated Reasoning Input:", "<previous_conclusions>"]
    for conclusion in previous_conclusions:
        lines.append(f"<conclusion>{conclusion}</conclusion>")
    lines.extend(["</previous_conclusions>", "", "<current_premises>"])
    for premise in current_premises:
        lines.append(f"<premise>{premise}</premise>")
    lines.extend([
        "</current_premises>",
        "",
        "<current_conclusion>",
        f"<conclusion>{current_conclusion}</conclusion>",
        "</current_conclusion>",
    ])

    stats = {
        "cumulative_mode": cumulative_mode,
        "previous_conclusions_before_dedup": max(0, len(step_history) - 1),
        "previous_conclusions_selected": len(previous_conclusions),
        "previous_conclusions_after_dedup": len(previous_conclusions),
        "current_premises_before_dedup": len(current_step.get("premises", [])),
        "current_premises_after_dedup": len(current_premises),
    }
    return "\n".join(lines), stats


def prepare_fol_shared_state(
    prompt_text: str,
    *,
    api_config: dict | None = None,
    extra_info: dict | None = None,
) -> dict | None:
    """Precompute response-level FOL state reusable across all steps."""
    context, question, options = extract_fol_problem(prompt_text, extra_info)
    fol_config = _build_fol_config(api_config)
    if not question:
        return None
    if fol_config.task_type not in (TaskType.MATH, TaskType.MATH_Z3) and not context:
        return None
    cache_key = (
        context,
        question,
        options or "",
        fol_config.preprocess.value,
        fol_config.task_type.value,
        (fol_config.api_config or {}).get("model"),
        (fol_config.api_config or {}).get("base_url"),
        (fol_config.api_config or {}).get("temperature"),
        (fol_config.api_config or {}).get("max_tokens"),
        (fol_config.api_config or {}).get("top_p"),
        bool((fol_config.api_config or {}).get("fol_judge_use_outlines", False)),
        fol_config.max_tries,
        fol_config.old_max_tries,
    )

    with _fol_shared_state_cache_lock:
        cached = _fol_shared_state_cache.get(cache_key)
        if cached is not None:
            _fol_shared_state_cache.move_to_end(cache_key)
            return cached
        inflight_state = _fol_shared_state_inflight.get(cache_key)
        owner = inflight_state is None
        if owner:
            inflight_state = {"event": threading.Event(), "result": None, "exc": None}
            _fol_shared_state_inflight[cache_key] = inflight_state

    if not owner:
        inflight_state["event"].wait()
        with _fol_shared_state_cache_lock:
            cached = _fol_shared_state_cache.get(cache_key)
            if cached is not None:
                _fol_shared_state_cache.move_to_end(cache_key)
                return cached
        inflight_exc = inflight_state.get("exc")
        if inflight_exc is not None:
            raise inflight_exc
        return inflight_state.get("result")

    try:
        shared_state = _load_shared_state_from_disk(cache_key, fol_config, fol_config.api_config)
        if shared_state is None:
            with _SharedStateDiskLock(cache_key, fol_config.api_config):
                shared_state = _load_shared_state_from_disk(cache_key, fol_config, fol_config.api_config)
                if shared_state is None:
                    engine = FOLEngine(fol_config)
                    processed_ctx, declarations = engine.preprocess(context, question, options or "")
                    if not declarations:
                        shared_state = None
                    else:
                        shared_state = {
                            "config": fol_config,
                            "processed_context": processed_ctx,
                            "declarations": declarations,
                        }
                        _store_shared_state_to_disk(cache_key, shared_state, fol_config.api_config)
    except BaseException as exc:
        with _fol_shared_state_cache_lock:
            inflight_state["exc"] = exc
            inflight_state["event"].set()
            _fol_shared_state_inflight.pop(cache_key, None)
        raise

    with _fol_shared_state_cache_lock:
        existing = _fol_shared_state_cache.get(cache_key)
        if existing is not None:
            _fol_shared_state_cache.move_to_end(cache_key)
            inflight_state["result"] = existing
            inflight_state["event"].set()
            _fol_shared_state_inflight.pop(cache_key, None)
            return existing
        if shared_state is not None:
            _fol_shared_state_cache[cache_key] = shared_state
            if len(_fol_shared_state_cache) > _FOL_SHARED_STATE_CACHE_MAX_SIZE:
                _fol_shared_state_cache.popitem(last=False)
        inflight_state["result"] = shared_state
        inflight_state["event"].set()
        _fol_shared_state_inflight.pop(cache_key, None)
    return shared_state


def _digest_text(text: str | None) -> str:
    """Return a stable digest for large cache-key strings."""
    if text is None:
        text = ""
    return sha1(text.encode("utf-8")).hexdigest()


def _build_verify_cache_key(
    *,
    shared_state: dict,
    fol_config: FOLConfig,
    step_to_translate: str,
) -> tuple:
    """Build a strict cache key for one FOL verify_step call."""
    api_cfg = fol_config.api_config or {}
    return (
        _digest_text(shared_state["processed_context"]),
        _digest_text(shared_state["declarations"]),
        _digest_text(step_to_translate),
        fol_config.preprocess.value,
        fol_config.translation.value,
        fol_config.task_type.value,
        fol_config.max_tries,
        fol_config.old_max_tries,
        fol_config.timeout,
        fol_config.cumulative,
        api_cfg.get("fol_cumulative_mode"),
        api_cfg.get("model"),
        api_cfg.get("base_url"),
        api_cfg.get("temperature"),
        api_cfg.get("max_tokens"),
        api_cfg.get("top_p"),
        bool(api_cfg.get("fol_judge_use_outlines", False)),
    )


def _verify_disk_cache_enabled(api_config: dict | None) -> bool:
    """Whether to share exact verify rewards across worker processes."""
    cfg = api_config or {}
    if "fol_verify_disk_cache" in cfg:
        return _as_bool(cfg.get("fol_verify_disk_cache"), True)
    return _as_bool(os.environ.get("FOL_VERIFY_DISK_CACHE"), True)


def _verify_disk_cache_dir(api_config: dict | None) -> str:
    """Return the on-disk cache directory for exact FOL verify rewards."""
    cfg = api_config or {}
    return str(
        cfg.get("fol_verify_cache_dir")
        or os.environ.get("FOL_VERIFY_DISK_CACHE_DIR")
        or "/tmp/verl_fol_verify_cache"
    )


def _verify_disk_cache_paths(cache_key: tuple, api_config: dict | None) -> tuple[str, str]:
    """Return data and lock paths for one exact verify cache key."""
    key_text = json.dumps(
        [_FOL_VERIFY_DISK_CACHE_VERSION, cache_key],
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    digest = sha1(key_text.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(_verify_disk_cache_dir(api_config), digest[:2])
    return os.path.join(cache_dir, f"{digest}.json"), os.path.join(cache_dir, f"{digest}.lock")


def _load_verify_reward_from_disk(cache_key: tuple, api_config: dict | None) -> float | None:
    """Load a completed exact verify reward from the cross-process cache."""
    if not _verify_disk_cache_enabled(api_config):
        return None
    data_path, _ = _verify_disk_cache_paths(cache_key, api_config)
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["reward"])
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Failed to read FOL verify disk cache: %s", exc)
        return None


def _store_verify_reward_to_disk(cache_key: tuple, reward: float, api_config: dict | None) -> None:
    """Store a successful exact verify reward in the cross-process cache."""
    if not _verify_disk_cache_enabled(api_config):
        return
    data_path, _ = _verify_disk_cache_paths(cache_key, api_config)
    tmp_path = f"{data_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    payload = {"version": _FOL_VERIFY_DISK_CACHE_VERSION, "reward": float(reward)}
    try:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, data_path)
    except Exception as exc:
        logger.warning("Failed to write FOL verify disk cache: %s", exc)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class _VerifyDiskLock:
    """Per-key file lock used to prevent cross-process exact verify stampedes."""

    def __init__(self, cache_key: tuple, api_config: dict | None):
        self.enabled = _verify_disk_cache_enabled(api_config)
        self._file = None
        _, self._lock_path = _verify_disk_cache_paths(cache_key, api_config)

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
            self._file = open(self._lock_path, "a+", encoding="utf-8")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        except Exception as exc:
            logger.warning("Failed to lock FOL verify disk cache: %s", exc)
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
            self._file = None
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is None:
            return False
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
        return False


def _verify_cache_log_enabled() -> bool:
    """Whether verify-cache summary logging is enabled."""
    return str(os.environ.get("FOL_VERIFY_CACHE_LOG", "0")).strip().lower() not in {"", "0", "false", "no", "off"}


def _print_verify_cache_summary() -> None:
    """Print a one-shot verify-cache summary when the worker process exits."""
    if not _verify_cache_log_enabled():
        return

    with _fol_verify_cache_stats_lock:
        hits = _fol_verify_cache_stats["hits"]
        misses = _fol_verify_cache_stats["misses"]
        step_items = list(_fol_verify_cache_step_stats.items())
    total = hits + misses
    if total <= 0:
        return

    per_step_parts = []
    for step_idx, stats in step_items:
        step_total = stats["hits"] + stats["misses"]
        if step_total <= 0:
            continue
        step_rate = stats["hits"] / step_total
        per_step_parts.append(f"step{step_idx}: {stats['hits']}/{step_total}={step_rate:.1%}")
    suffix = f" | {', '.join(per_step_parts)}" if per_step_parts else ""
    print(
        f"[FOLVerifyCacheSummary][pid={os.getpid()}] "
        f"hits={hits} misses={misses} hit_rate={hits / total:.1%}{suffix}",
        flush=True,
    )


def _register_verify_cache_summary() -> None:
    """Register the summary printer exactly once."""
    global _fol_verify_cache_summary_registered
    with _fol_verify_cache_stats_lock:
        if _fol_verify_cache_summary_registered:
            return
        atexit.register(_print_verify_cache_summary)
        _fol_verify_cache_summary_registered = True


def _log_verify_cache_event(hit: bool, step_index: int) -> None:
    """Accumulate verify-cache hit statistics by logical step index."""
    if not _verify_cache_log_enabled():
        return

    _register_verify_cache_summary()
    with _fol_verify_cache_stats_lock:
        stat_key = "hits" if hit else "misses"
        _fol_verify_cache_stats[stat_key] += 1
        step_stats = _fol_verify_cache_step_stats.setdefault(step_index, {"hits": 0, "misses": 0})
        step_stats[stat_key] += 1


def compute_step_reward_format_fol(
    step_text: str, prompt_text: str, step_history: list[str], **kwargs,
) -> float:
    """Format-check process reward ensuring strict step/premise/conclusion tags."""
    return 1.0 if check_step_format_fol(step_text) else 0.0


def compute_step_reward_fol(
    step_text: str,
    prompt_text: str,
    step_history: list[str],
    *,
    api_config: dict | None = None,
    extra_info: dict | None = None,
    fol_shared_state: dict | None = None,
    return_debug: bool = False,
) -> float | dict:
    """Unified FOL entailment process reward.

    Configurable via api_config:
      fol_preprocess: "direct" | "structured"
      fol_translation: "implication" | "assertion"
      max_tries, timeout, cumulative, fol_format_failed_score
    """
    # _t0 = time.time()
    # _tid = threading.current_thread().name
    # print(f"[FOL][{_tid}] ▶ enter  step={step_text[:60]!r}...", flush=True)

    try:
        debug_info = {
            "cache_hit": False,
            "translation_response": None,
            "correction_attempts": 0,
            "z3_output": None,
            "z3_error": None,
            "judge_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        fol_format_failed_score = float((api_config or {}).get("fol_format_failed_score", 0.0))

        # Format precheck: if the step contains <step> but has bad format
        # (missing <premise>/<conclusion>, mismatched tags, etc.), skip the
        # expensive FOL judge call and return the configured FOL format score.
        if "<step>" in step_text and not check_step_format_fol(step_text):
            debug_info["format_failed_closed"] = True
            debug_info["format_failed_score"] = fol_format_failed_score
            return {"score": fol_format_failed_score, "debug": debug_info} if return_debug else fol_format_failed_score
        if _has_student_premise_conclusion_duplicate(step_text):
            debug_info["student_premise_conclusion_duplicate"] = True
            return {"score": 0.0, "debug": debug_info} if return_debug else 0.0
        shared_state = fol_shared_state or prepare_fol_shared_state(
            prompt_text, api_config=api_config, extra_info=extra_info
        )
        if shared_state is None:
            debug_info["declaration_failed_closed"] = True
            return {"score": 0.0, "debug": debug_info} if return_debug else 0.0

        fol_config = shared_state["config"]
        engine = FOLEngine(fol_config)

        # Handle cumulative mode. Default is current_only; legacy cumulative=True
        # maps to step mode inside _build_fol_config when no explicit mode is set.
        cumulative_mode = _normalize_cumulative_mode(
            (fol_config.api_config or {}).get("fol_cumulative_mode")
        )
        if cumulative_mode != "current_only" and step_history:
            cumulative_input, cumulative_stats = _build_source_separated_cumulative_input(
                step_history,
                mode=cumulative_mode,
            )
            if cumulative_input:
                step_to_translate = cumulative_input
                debug_info.update(cumulative_stats)
            else:
                step_to_translate = step_text
        else:
            step_to_translate = step_text
        step_index = max(0, len(step_history) - 1)

        verify_cache_key = _build_verify_cache_key(
            shared_state=shared_state,
            fol_config=fol_config,
            step_to_translate=step_to_translate,
        )
        owner = False
        inflight_state = None
        with _fol_verify_cache_lock:
            cached_reward = _fol_verify_cache.get(verify_cache_key)
            if cached_reward is not None:
                _fol_verify_cache.move_to_end(verify_cache_key)
                debug_info["cache_hit"] = True
                _log_verify_cache_event(True, step_index)
                return {"score": cached_reward, "debug": debug_info} if return_debug else cached_reward
            inflight_state = _fol_verify_inflight.get(verify_cache_key)
            if inflight_state is None:
                inflight_state = {"event": threading.Event(), "reward": None, "exc": None}
                _fol_verify_inflight[verify_cache_key] = inflight_state
                owner = True

        if not owner:
            inflight_state["event"].wait()
            with _fol_verify_cache_lock:
                cached_reward = _fol_verify_cache.get(verify_cache_key)
                if cached_reward is not None:
                    _fol_verify_cache.move_to_end(verify_cache_key)
                    debug_info["cache_hit"] = True
                    _log_verify_cache_event(True, step_index)
                    return {"score": cached_reward, "debug": debug_info} if return_debug else cached_reward
            inflight_exc = inflight_state.get("exc")
            if inflight_exc is not None:
                raise inflight_exc
            inflight_reward = inflight_state.get("reward")
            if inflight_reward is not None:
                debug_info["cache_hit"] = True
                _log_verify_cache_event(True, step_index)
                return {"score": inflight_reward, "debug": debug_info} if return_debug else inflight_reward
            raise RuntimeError("FOL verify cache in-flight request completed without a result")

        # print(f"[FOL][{_tid}] → verify_step({fol_config.translation.value})...", flush=True)
        # _t2 = time.time()
        try:
            reward = _load_verify_reward_from_disk(verify_cache_key, fol_config.api_config)
            if reward is not None:
                debug_info["cache_hit"] = True
                debug_info["verify_disk_cache_hit"] = True
                _log_verify_cache_event(True, step_index)
            else:
                with _VerifyDiskLock(verify_cache_key, fol_config.api_config):
                    reward = _load_verify_reward_from_disk(verify_cache_key, fol_config.api_config)
                    if reward is not None:
                        debug_info["cache_hit"] = True
                        debug_info["verify_disk_cache_hit"] = True
                        _log_verify_cache_event(True, step_index)
                    else:
                        _log_verify_cache_event(False, step_index)
                        reward = engine.verify_step(
                            shared_state["processed_context"],
                            shared_state["declarations"],
                            step_to_translate,
                            debug_info=debug_info,
                        )
                        reward = float(reward)
                        _store_verify_reward_to_disk(verify_cache_key, reward, fol_config.api_config)
        except BaseException as exc:
            with _fol_verify_cache_lock:
                inflight_state["exc"] = exc
                inflight_state["event"].set()
                _fol_verify_inflight.pop(verify_cache_key, None)
            raise
        with _fol_verify_cache_lock:
            _fol_verify_cache[verify_cache_key] = reward
            if len(_fol_verify_cache) > _FOL_VERIFY_CACHE_MAX_SIZE:
                _fol_verify_cache.popitem(last=False)
            inflight_state["reward"] = reward
            inflight_state["event"].set()
            _fol_verify_inflight.pop(verify_cache_key, None)
        # print(f"[FOL][{_tid}] ◀ done  reward={reward}  verify={time.time()-_t2:.2f}s  total={time.time()-_t0:.2f}s", flush=True)
        return {"score": reward, "debug": debug_info} if return_debug else reward
    except Exception as e:
        # print(f"[FOL][{_tid}] ✗ EXCEPTION after {time.time()-_t0:.2f}s: {e}", flush=True)
        logger.warning("FOL reward computation failed: %s", e)
        if return_debug:
            return {
                "score": 0.0,
                "debug": {
                    "cache_hit": False,
                    "translation_response": None,
                    "correction_attempts": 0,
                    "z3_output": None,
                    "z3_error": str(e),
                    "judge_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            }
        return 0.0


# ---------------------------------------------------------------------------
# Isabelle verification path (task_type == "math")
# ---------------------------------------------------------------------------

_isabelle_engine = None
_isabelle_engine_lock = threading.Lock()


def _get_isabelle_engine(api_config: dict | None = None):
    """Lazy singleton: one IsabelleEngine per process.

    KNOWN LIMITATION (2026-06-14): the singleton is built from whatever
    api_config the first caller passes. Subsequent callers' api_config is
    SILENTLY IGNORED, even if they pass different judge_url / pool_workers
    / session / fol_timeout. This is fine today because:
      * a single training run has exactly one fol_task_type=math reward
        type using Isabelle
      * the Z3 path uses its own FOLEngine, not IsabelleEngine

    It would break if a future training mixed two reward types both
    wanting Isabelle with different configs (e.g. one with HOL-Number_Theory
    and another with HOL-Analysis). In that case, replace this singleton
    with a config-keyed dict of engines.
    """
    global _isabelle_engine
    if _isabelle_engine is not None:
        return _isabelle_engine
    with _isabelle_engine_lock:
        if _isabelle_engine is not None:
            return _isabelle_engine
        from verl.utils.isabelle_utils.engine import IsabelleEngine, IsabelleConfig
        cfg = api_config or {}
        # fol_timeout aligns with Z3 path: per-verification deadline in
        # seconds. Default 60 matches the pool's pre-existing CHECK_DEADLINE.
        timeout = cfg.get("fol_timeout") or cfg.get("timeout") or 60.0
        config = IsabelleConfig(
            judge_url=cfg.get("base_url", "http://127.0.0.1:4873/v1"),
            judge_model=cfg.get("model", "Qwen3.6-35B-A3B"),
            pool_workers=int(cfg.get("isabelle_pool_workers", 32)),
            check_deadline=float(timeout),
            # Judge HTTP deadline; wired (2026-07-11), operational value 240.
            api_timeout=float(cfg.get("api_timeout") or 240.0),
        )
        _isabelle_engine = IsabelleEngine(config)
        return _isabelle_engine


def compute_solution_reward_isabelle(
    problem: str,
    response: str,
    ground_truth: str,
    *,
    api_config: dict | None = None,
    dataset: str = "math",
    idx: int = 0,
    sample: int = 0,
    return_debug: bool = False,
    max_steps: int = 0,
) -> list[float] | tuple[list[float], dict]:
    """Whole-solution Isabelle verification.

    Returns:
        list[float]: per-step rewards (0/1). With return_debug=True, returns
        (rewards, debug_dict) where debug_dict carries judge/verifier
        statistics for metric aggregation (mirrors Z3 path's debug_info).

    Fast-fail mirrors the Z3 path: if XML is malformed or there is no
    \\boxed{} answer, skip the expensive judge + Isabelle call entirely.
    """
    debug = {
        "format_ok": False,
        "givens_ok": False,
        "steps_ok": False,
        "outcome_correct": False,
        "n_steps": 0,
        "verified_steps": 0,
        "rewarded_steps": 0,
        "neutral_steps": 0,
        "translation_attempts_givens": 0,
        "translation_attempts_steps": 0,
        "guard_failed_steps": 0,
        "premise_inconsistent_at": None,
        "format_failed_closed": False,
        "pattern": "",
        # Per-symbol counts from the pattern (o>c>m>g>x priority). g is ALSO
        # kept split as neutral_steps/guard_failed_steps above; these are the
        # combined pattern-symbol tallies for the [Step Rewards] print parity.
        "o_steps": 0,
        "x_steps": 0,
        "c_steps": 0,
        "g_steps": 0,
        "m_steps": 0,
        # t = translation-failed: the XML step existed but never reached
        # Isabelle (givens/steps translation rejected it, e.g. a bare-value
        # conclusion that is not a proposition). These carry NO pattern symbol
        # (engine only tallies steps that reached the prover), so they are
        # counted here as n_steps - len(pattern). Reward is 0 (fail-closed),
        # same as x, but kept SEPARATE so x_rate stays "model reasoning
        # failed" and t_rate is "translator/format could not formalize".
        # Invariant restored: o + x + c + g + m + t == n_steps.
        "t_steps": 0,
        # Per-response wall profile (2026-07-11 review #6): decomposes the
        # reward tail. translate_wall_s = judge translation wall (all calls);
        # prove_queue_s / prove_run_s = summed idle-worker wait vs in-worker
        # check time over this response's prover calls; reward_wall_s = whole
        # verify_solution wall. Cache/restart gauges are process-CUMULATIVE
        # (same value across a batch's responses; the W&B mean IS the value).
        "translate_wall_s": 0.0,
        "prove_calls": 0,
        "prove_queue_s": 0.0,
        "prove_run_s": 0.0,
        "prove_cache_hits": 0,
        "reward_wall_s": 0.0,
        "pool_restarts": 0,
        "thm_cache_hit_rate": 0.0,
        "tr_cache_hit_rate": 0.0,
        # Real judge load vs cache reuse (2026-07-11 review): the legacy
        # judge_calls_* counted cache MARKERS as judge calls. http calls
        # count actual requests.post attempts; the *_hits split says which
        # cache layer answered instead.
        "judge_http_calls": 0,
        "judge_retry_calls": 0,
        "translation_mem_hits": 0,
        "translation_disk_hits": 0,
        "translation_flight_hits": 0,
        "translation_xproc_hits": 0,
        "translation_failures": 0,
        "error": None,
    }
    # Fast-fail: malformed XML or missing boxed answer (matches Z3 path).
    try:
        from verl.utils.isabelle_utils.xml_utils import (
            parse_xml_steps, boxed_answer,
        )
        steps_xml = parse_xml_steps(response)
        if steps_xml is None or boxed_answer(response) is None:
            debug["format_failed_closed"] = True
            return ([0.0], debug) if return_debug else [0.0]
    except Exception:
        # If even the fast-fail check throws, fall through to the engine
        # so the failure shows up in the structured result rather than here.
        pass
    try:
        engine = _get_isabelle_engine(api_config)
        result = engine.verify_solution(
            problem=problem, response=response,
            ground_truth=ground_truth, dataset=dataset,
            idx=idx, sample=sample, max_steps=max_steps,
        )
        debug["format_ok"] = bool(result.get("format_ok"))
        debug["givens_ok"] = bool(result.get("givens_ok"))
        debug["steps_ok"] = bool(result.get("steps_ok"))
        debug["outcome_correct"] = bool(result.get("outcome_correct"))
        debug["n_steps"] = int(result.get("n_steps", 0) or 0)
        debug["pattern"] = str(result.get("pattern") or "")
        debug["premise_inconsistent_at"] = result.get(
            "premise_inconsistent_at")
        # translate_a: single dict OR list of dicts (judge attempts)
        ta = result.get("translate_a")
        if isinstance(ta, list):
            debug["translation_attempts_givens"] = len(ta)
        elif isinstance(ta, dict):
            debug["translation_attempts_givens"] = 1
        # translate_b: list (one entry per chunk) of {"attempts": ...}
        tb = result.get("translate_b") or []
        if isinstance(tb, list):
            debug["translation_attempts_steps"] = sum(
                len(x) if isinstance(x, list) else 1 for x in tb)
        steps = result.get("steps", []) or []
        for s in steps:
            if s.get("verified"):
                debug["verified_steps"] += 1
            if s.get("rewarded"):
                debug["rewarded_steps"] += 1
            if s.get("neutral"):
                debug["neutral_steps"] += 1
            if not s.get("guard_ok", True):
                debug["guard_failed_steps"] += 1
        # Per-symbol counts straight from the pattern string, so every symbol
        # in the [Step Rewards] print (o/x/c/g/m) has a matching metric with
        # the SAME priority resolution as the printed pattern (o>c>m>g>x).
        # rewarded_steps == o_steps by construction; kept for back-compat.
        pat = debug["pattern"]
        debug["o_steps"] = pat.count("o")   # rewarded (earned reward 1)
        debug["x_steps"] = pat.count("x")   # unverified (reached prover, no proof)
        debug["c_steps"] = pat.count("c")   # premises inconsistent (contaminated)
        debug["g_steps"] = pat.count("g")   # verified-but-neutral/guard-failed
        debug["m_steps"] = pat.count("m")   # verified-but-transcription-missing
        # t = translation-failed: XML steps that never reached the prover.
        # The pattern only has a symbol per prover-reached step, so any
        # shortfall against n_steps is a translation/format failure. Clamp at
        # 0 so a definition-only-neutral quirk can never make this negative.
        debug["t_steps"] = max(0, debug["n_steps"] - len(pat))
        # Wall profile from the engine (review #6). Guarded: an old cached
        # rec or an early-return path may lack keys; defaults stay 0.
        prof = result.get("prof") or {}
        debug["translate_wall_s"] = float(prof.get("translate_s") or 0.0)
        debug["prove_calls"] = int(prof.get("prove_calls") or 0)
        debug["prove_queue_s"] = float(prof.get("prove_queue_s") or 0.0)
        debug["prove_run_s"] = float(prof.get("prove_run_s") or 0.0)
        debug["prove_cache_hits"] = int(prof.get("prove_cache_hits") or 0)
        debug["reward_wall_s"] = float(prof.get("reward_wall_s") or 0.0)
        try:
            debug["pool_restarts"] = int(engine.pool.restart_count)
            _ch = int(engine.pool.cache_hits)
            _cm = int(engine.pool.cache_misses)
            debug["thm_cache_hit_rate"] = _ch / max(1, _ch + _cm)
        except Exception:  # noqa: BLE001 -- gauges must never fail scoring
            pass
        try:
            from verl.utils.isabelle_utils.judge import translate_cache_stats
            _ts = translate_cache_stats()
            debug["tr_cache_hit_rate"] = _ts["hits"] / max(
                1, _ts["hits"] + _ts["misses"])
        except Exception:  # noqa: BLE001
            pass

        # Split real judge HTTP load from cache reuse (review round 2).
        def _tr_attempt_entries(x):
            # translate_a is a list (or single dict); translate_b is one
            # chunk's list or a list of per-chunk lists.
            if isinstance(x, dict):
                return [x]
            out = []
            for e in (x or []):
                if isinstance(e, list):
                    out.extend(a for a in e if isinstance(a, dict))
                elif isinstance(e, dict):
                    out.append(e)
            return out

        for a in (_tr_attempt_entries(result.get("translate_a"))
                  + _tr_attempt_entries(result.get("translate_b"))):
            marker = a.get("cache")
            if marker == "mem":
                debug["translation_mem_hits"] += 1
            elif marker == "disk":
                debug["translation_disk_hits"] += 1
            elif marker == "flight":
                debug["translation_flight_hits"] += 1
            elif marker == "xproc":
                debug["translation_xproc_hits"] += 1
            posts = int(a.get("http_posts") or 0)
            debug["judge_http_calls"] += posts
            debug["judge_retry_calls"] += max(0, posts - 1)
        # response-level FINAL translation failures (not per-retry noise)
        debug["translation_failures"] = int(
            (1 if debug["format_ok"] and not debug["givens_ok"] else 0)
            + (1 if debug["givens_ok"] and not debug["steps_ok"] else 0))
        if not result.get("steps_ok") or not steps:
            n = debug["n_steps"] or 1
            rewards = [0.0] * n
        else:
            rewards = [1.0 if s.get("rewarded") else 0.0 for s in steps]
        return (rewards, debug) if return_debug else rewards
    except Exception as e:
        logger.warning("Isabelle reward computation failed: %s", e)
        debug["error"] = str(e)[:200]
        return ([0.0], debug) if return_debug else [0.0]
