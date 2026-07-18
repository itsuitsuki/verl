"""
Step Reward Manager for Step-GDPO and non-parametric process rewards.
TODO: support parametric process reward models

Computes both outcome reward (answer correctness) and step-level process rewards
(format / random / fol). Process rewards are passed as per-step (position, score)
lists via reward_extra_info so the advantage estimator can reconstruct token-level
tensors and apply big-pool normalization.
"""

import asyncio
import hashlib
import inspect
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.experimental.reward_loop.reward_manager.format_penalty import (
    assess_response, place_extra_penalty, verifier_response)
from verl.utils.reward_score import default_compute_score
from verl.utils.step_splitter import (char_end_to_token_pos, default_split_fn,
                                      get_step_token_positions)


def _compute_step_reward_random(step_text: str, prompt_text: str, step_history: list[str], **kwargs) -> float:
    """Random baseline process reward."""
    random_seed = kwargs.get("random_seed")
    if random_seed is not None:
        payload = "\x1f".join([str(random_seed), prompt_text, *step_history, step_text])
        digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).digest()
        return float(digest[0] & 1)
    return float(random.randint(0, 1))


def _as_bool(value) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    elif isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@register("step")
class StepRewardManager(RewardManagerBase):
    """
    Step Reward Manager for Step-GDPO and non-parametric process rewards.
    TODO: support parametric process reward models

    Computes:
    - outcome_reward: scalar correctness score (placed at last valid response token)
    - process_reward: per-step scores via configurable step_reward_types

    The step-level process rewards are serialized as lists of (position, score) tuples
    in reward_extra_info, keyed by "{type}_step_reward". The advantage estimator
    (step_gdpo) reads these to build token-level tensors and perform big-pool normalization.
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
        split_fn: Optional[Callable[[str], list[str]]] = None,
        step_reward_type: Optional[str | list[str]] = None,
        step_reward_fns: Optional[dict] = None,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Pluggable step splitter
        self.split_fn = split_fn or default_split_fn

        # API configuration for LLM-based step rewards (FOL, self_eval, etc.)
        # Priority: config.reward.api_config > env vars > defaults
        self.api_config = {
            "model": os.environ.get("SELF_EVAL_MODEL", os.environ.get("FOL_MODEL")),
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_BASE_URL"),
            "temperature": 0.6,
            "max_tokens": 1024,
        }
        cfg_override = config.get("reward", {}).get("api_config",
                       config.get("reward", {}).get("fol_api_config", {}))
        if cfg_override:
            self.api_config.update({k: v for k, v in cfg_override.items() if v is not None})

        # FOL configuration: correct_loop max retries, timeout, pipeline settings
        # Priority: reward config > algorithm config > defaults
        reward_cfg = config.get("reward", {})
        algo_cfg = config.get("algorithm", {})
        max_tries = reward_cfg.get("fol_max_tries", algo_cfg.get("fol_max_tries", None))
        if max_tries is not None:
            self.api_config["max_tries"] = int(max_tries)
        old_max_tries = reward_cfg.get("fol_old_max_tries", algo_cfg.get("fol_old_max_tries", None))
        if old_max_tries is not None:
            self.api_config["old_max_tries"] = int(old_max_tries)
        # Per-verification deadline in seconds (Z3 subprocess timeout / Isabelle
        # verify_timeout). New alias `verify_timeout` preferred; `fol_timeout`
        # kept for backward compatibility with existing scripts.
        z3_timeout = reward_cfg.get("verify_timeout", algo_cfg.get("verify_timeout",
                     reward_cfg.get("fol_timeout", algo_cfg.get("fol_timeout", None))))
        if z3_timeout is not None:
            self.api_config["timeout"] = int(z3_timeout)
        api_timeout = reward_cfg.get("api_timeout", algo_cfg.get("api_timeout", None))
        if api_timeout is not None:
            self.api_config["api_timeout"] = int(api_timeout)
        # New alias `verify_cumulative_mode` preferred; `fol_cumulative_mode`
        # kept for backward compatibility.
        fol_cumulative_mode = reward_cfg.get("verify_cumulative_mode", algo_cfg.get("verify_cumulative_mode",
                              reward_cfg.get("fol_cumulative_mode", algo_cfg.get("fol_cumulative_mode", None))))
        if fol_cumulative_mode is not None:
            self.api_config["fol_cumulative_mode"] = str(fol_cumulative_mode)
        # New alias `verify_task_type` preferred; `fol_task_type` kept for
        # backward compatibility ("fol" prefix = formal-verification, not FOL).
        fol_task_type = reward_cfg.get("verify_task_type", algo_cfg.get("verify_task_type",
                        reward_cfg.get("fol_task_type", algo_cfg.get("fol_task_type", None))))
        if fol_task_type is not None:
            self.api_config["fol_task_type"] = str(fol_task_type)
        isabelle_pool_workers = reward_cfg.get("isabelle_pool_workers", algo_cfg.get("isabelle_pool_workers", None))
        if isabelle_pool_workers is not None:
            self.api_config["isabelle_pool_workers"] = int(isabelle_pool_workers)
        isabelle_rss_cap = reward_cfg.get("isabelle_worker_rss_cap_gb",
                                          algo_cfg.get("isabelle_worker_rss_cap_gb", None))
        if isabelle_rss_cap is not None:
            self.api_config["isabelle_worker_rss_cap_gb"] = float(isabelle_rss_cap)
        print(f"FOL config 'fol_cumulative_mode' is set to: {self.api_config.get('fol_cumulative_mode', 'current_only')}")

        # FOL pipeline / translation mode
        fol_preprocess = reward_cfg.get("fol_preprocess", algo_cfg.get("fol_preprocess", None))
        if fol_preprocess is not None:
            self.api_config["fol_preprocess"] = str(fol_preprocess)
        fol_translation = reward_cfg.get("fol_translation", algo_cfg.get("fol_translation", None))
        if fol_translation is not None:
            self.api_config["fol_translation"] = str(fol_translation)
        fol_judge_use_outlines = reward_cfg.get(
            "fol_judge_use_outlines",
            algo_cfg.get("fol_judge_use_outlines", False),
        )
        self.api_config["fol_judge_use_outlines"] = bool(fol_judge_use_outlines)
        self.validate_with_step_reward = _as_bool(
            reward_cfg.get("validate_with_step_reward", algo_cfg.get("validate_with_step_reward", True))
        )

        # Step reward type: explicit parameter > reward config > algorithm config > default "random"
        if step_reward_type is not None: # explicit parameter only exists @ unit tests
            if isinstance(step_reward_type, str):
                self.step_reward_types = [step_reward_type]
            else:
                self.step_reward_types = list(step_reward_type)
        else: # all training scripts follow this branch
            reward_cfg = config.get("reward", {})
            algo_cfg = config.get("algorithm", {})
            
            srt = reward_cfg.get("step_reward_type", None)
            if srt is None:
                srt = algo_cfg.get("step_reward_type", None)
            if srt is None:
                raise ValueError("step_reward_type is not specified")
                
            if isinstance(srt, str):
                self.step_reward_types = [srt]
            else:
                self.step_reward_types = list(srt)
                
        # Initialize pluggable reward functions registry
        self.step_reward_fns = {
            "random": _compute_step_reward_random
        }
        
        # Built-in extra reward types if requested (Lazy loading)
        if any(rt in ["fol", "fol_old", "format"] for rt in self.step_reward_types):
            try:
                from verl.utils.reward_score.formal_verify import (
                    compute_step_reward_fol, compute_step_reward_format_fol)
                if "format" not in self.step_reward_fns:
                    self.step_reward_fns["format"] = compute_step_reward_format_fol
                if "fol" not in self.step_reward_fns:
                    self.step_reward_fns["fol"] = compute_step_reward_fol
            except ImportError as e:
                logging.getLogger(__name__).warning("Failed to lazily load built-in FOL reward functions: %s", e)
            try:
                from verl.utils.reward_score.fol_old import \
                    compute_step_reward_fol as compute_step_reward_fol_old
                if "fol_old" not in self.step_reward_fns:
                    self.step_reward_fns["fol_old"] = compute_step_reward_fol_old
            except ImportError:
                pass

        if "self_eval" in self.step_reward_types:
            from verl.utils.reward_score.self_eval import \
                compute_step_reward_self_eval
            if "self_eval" not in self.step_reward_fns:
                self.step_reward_fns["self_eval"] = compute_step_reward_self_eval

        # Override with any user-provided step_reward_fns
        if step_reward_fns:
            self.step_reward_fns.update(step_reward_fns)

        # Resolve use_xml_steps: reward config > algorithm config > False
        reward_cfg = config.get("reward", {})
        algo_cfg = config.get("algorithm", {})
        use_xml_cfg = reward_cfg.get("use_xml_steps", None)
        if use_xml_cfg is None:
            use_xml_cfg = algo_cfg.get("use_xml_steps", None)
        self.use_xml = bool(use_xml_cfg) if use_xml_cfg is not None else False

        # --- Anti-reward-hacking penalty config ---
        # When the model exploits process reward by inflating steps, breaking format,
        # or hitting max_response_length, zero out all process rewards for that response.
        # Configurable via algorithm or reward config.
        self.penalty_max_steps = int(
            reward_cfg.get("penalty_max_steps", algo_cfg.get("penalty_max_steps", 0))
        )  # 0 = disabled; e.g. 12 to penalize responses with >12 steps
        self.penalty_on_truncated = bool(
            reward_cfg.get("penalty_on_truncated", algo_cfg.get("penalty_on_truncated", False))
        )  # penalize responses that hit max_response_length
        self.penalty_on_multi_boxed = bool(
            reward_cfg.get("penalty_on_multi_boxed", algo_cfg.get("penalty_on_multi_boxed", False))
        )  # penalize responses with multiple \boxed{}
        self.penalty_on_bad_format = bool(
            reward_cfg.get("penalty_on_bad_format", algo_cfg.get("penalty_on_bad_format", False))
        )  # penalize each badly formatted <step> (per step; see format_penalty.py)
        self.penalty_score = float(
            reward_cfg.get("penalty_score", algo_cfg.get("penalty_score", 0.0))
        )  # score to assign when penalized (default 0.0, can be negative)
        random_reward_seed = reward_cfg.get("random_reward_seed", algo_cfg.get("random_reward_seed", None))
        self.random_reward_seed = int(random_reward_seed) if random_reward_seed is not None else None

        # LLM-backed step rewards are token-budget-bound long before 64 threads
        # are useful. Keep the default conservative, and let scripts opt in to
        # more parallelism via config/env when the upstream API budget allows it.
        max_workers = reward_cfg.get(
            "step_reward_max_workers",
            algo_cfg.get("step_reward_max_workers", os.environ.get("VERL_STEP_REWARD_MAX_WORKERS")),
        )
        if max_workers is None:
            uses_llm_step_reward = any(rt in {"fol", "self_eval"} for rt in self.step_reward_types)
            max_workers = 4 if uses_llm_step_reward else min(16, os.cpu_count() or 4)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))

    def _get_step_token_positions(self, response_text: str, valid_response_ids, valid_response_length: int):
        """Map character-level step boundaries to token positions.

        Delegates to the shared ``get_step_token_positions`` utility.

        Returns:
            List of (step_text, token_end_pos) where token_end_pos is the
            index of the last token in this step (within response_ids).
        """
        return get_step_token_positions(
            response_text=response_text,
            valid_response_length=valid_response_length,
            tokenizer=self.tokenizer,
            use_xml=self.use_xml,
            split_fn=self.split_fn,
            response_ids=valid_response_ids,
        )

    def _init_reward_extra_info(self, score) -> dict:
        """Initialize a stable reward_extra_info schema for this manager."""
        reward_extra_info = {
            "acc": score,
            "num_steps": 0,
            "process_reward_penalized": False,
            "penalty_reason": "",
            "validation_skipped_step_reward": False,
        }
        for reward_type in self.step_reward_types:
            reward_extra_info[f"{reward_type}_step_reward"] = []
        if "fol" in self.step_reward_types:
            reward_extra_info.update(
                {
                    "fol_debug": [],
                    "fol_judge_prompt_tokens": 0,
                    "fol_judge_completion_tokens": 0,
                    "fol_judge_total_tokens": 0,
                    "fol_judge_calls": 0,
                    "fol_judge_completion_tokens_per_call": 0.0,
                    "fol_cache_hit_rate": 0.0,
                    "fol_verifier_steps": 0,
                    "fol_entailed_steps": 0,
                    "fol_not_entailed_steps": 0,
                    "fol_invalid_translation_steps": 0,
                    "fol_invalid_expression_steps": 0,
                    "fol_expression_repair_steps": 0,
                    "fol_autofilled_quantifier_steps": 0,
                    "fol_autofilled_free_identifier_steps": 0,
                    "fol_autofilled_symbolic_constant_steps": 0,
                    "fol_sort_mismatch_steps": 0,
                    "fol_leakage_steps": 0,
                    "fol_student_duplicate_steps": 0,
                    "fol_declaration_failed_steps": 0,
                    "fol_format_failed_steps": 0,
                }
            )
            # Isabelle backend keys must be prefilled too: hard-penalized /
            # validation-skipped responses return before the Isabelle branch,
            # and a key-set mismatch across a batch turns the whole batch's
            # isabelle/* metrics into NaN (missing keys collate as None).
            # Zeros are also semantically right for penalized responses
            # (bad format => format_ok=0, nothing verified).
            if (self.api_config or {}).get("fol_task_type") == "math":
                reward_extra_info.update(
                    {
                        "isabelle_format_ok": 0,
                        "isabelle_givens_ok": 0,
                        "isabelle_steps_ok": 0,
                        "isabelle_outcome_correct": 0,
                        "isabelle_n_steps": 0,
                        "isabelle_verified_steps": 0,
                        "isabelle_rewarded_steps": 0,
                        "isabelle_neutral_steps": 0,
                        "isabelle_guard_failed_steps": 0,
                        "isabelle_judge_calls_givens": 0,
                        "isabelle_judge_calls_steps": 0,
                        "isabelle_judge_calls_total": 0,
                        "isabelle_o_steps": 0,
                        "isabelle_x_steps": 0,
                        "isabelle_c_steps": 0,
                        "isabelle_u_steps": 0,
                        "isabelle_g_steps": 0,
                        "isabelle_m_steps": 0,
                        "isabelle_t_steps": 0,
                        "isabelle_judge_http_wall_time": 0.0,
                        "isabelle_translate_validate_wall_time": 0.0,
                        "isabelle_prove_calls": 0,
                        "isabelle_prove_queue_time": 0.0,
                        "isabelle_prove_run_time": 0.0,
                        "isabelle_prove_cache_hits": 0,
                        "isabelle_reward_wall_time": 0.0,
                        "isabelle_pool_restarts": 0,
                        "isabelle_thm_cache_hit_rate": 0.0,
                        "isabelle_tr_cache_hit_rate": 0.0,
                        "isabelle_judge_http_calls": 0,
                        "isabelle_judge_retry_calls": 0,
                        "isabelle_translation_mem_hits": 0,
                        "isabelle_translation_disk_hits": 0,
                        "isabelle_translation_shared_hits": 0,
                        "isabelle_translation_xproc_hits": 0,
                        "isabelle_translation_failures": 0,
                        "isabelle_pattern": "",
                        "isabelle_error": "",
                    }
                )
        return reward_extra_info

    async def run_single(self, data: DataProto) -> dict:
        """Compute outcome + process rewards for a single data item."""
        assert len(data) == 1, "StepRewardManager only supports single data item"
        data_item = data[0]

        # Extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        # Extract metadata
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # Decode response
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # 1. Compute outcome reward
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        async def _grade_outcome(solution_str):
            """One outcome-grader call; also used to regrade on the first-boxed prefix when the response contains several \\boxed answers."""
            if self.is_async_reward_score:
                return await self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                )
            return await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        result = await _grade_outcome(response_str)
        if isinstance(result, dict):
            score = result["score"]
        else:
            score = result

        reward_extra_info = self._init_reward_extra_info(score)
        if isinstance(result, dict):
            reward_extra_info.update(result)

        # Validation reports plain benchmark accuracy: the format rules below (including the
        # boxed contract) shape TRAINING rewards only, so returning here keeps validation
        # curves comparable with baselines and published numbers.
        if _as_bool(data_item.non_tensor_batch.get("__validate__", False)) and not self.validate_with_step_reward:
            reward_extra_info["validation_skipped_step_reward"] = True
            return {"reward_score": score, "reward_extra_info": reward_extra_info}

        # 2. Compute step-level process rewards
        # 2.1 Splitting | get token positions (end pos) of each step for assigning rewards
        step_positions = self._get_step_token_positions(response_str, valid_response_ids, valid_response_length)
        reward_extra_info["num_steps"] = len(step_positions)

        # --- Anti-reward-hacking precheck ---
        # The unified rules live in format_penalty.assess_response (shared with the tree
        # manager): truncation and the boxed contract are whole-response, a badly formatted
        # step is penalized ALONE while the valid steps still verify, and the step-count cap
        # penalizes only the suffix.
        decision = assess_response(
            response_str,
            step_positions,
            use_xml=self.use_xml,
            valid_response_length=int(valid_response_length),
            response_length=int(response_length),
            penalty_score=self.penalty_score,
            penalty_max_steps=self.penalty_max_steps,
            penalty_on_truncated=self.penalty_on_truncated,
            penalty_on_multi_boxed=self.penalty_on_multi_boxed,
            penalty_on_bad_format=self.penalty_on_bad_format,
        )
        if decision.reasons:
            reward_extra_info["penalty_reason"] = "|".join(decision.reasons)
        if decision.penalized:
            reward_extra_info["process_reward_penalized"] = True
        if decision.num_steps_override is not None:
            reward_extra_info["num_steps"] = decision.num_steps_override
        if decision.outcome_override is not None:
            # A response without any \boxed answer forfeits the outcome.
            score = float(decision.outcome_override)
            reward_extra_info["acc"] = score
        if decision.first_boxed_end is not None:
            # Several \boxed answers: the FIRST one is the committed answer. The initial
            # grade read the full text (whose LAST boxed is the exploitable one), so regrade
            # on the prefix that ends with the first boxed.
            regrade = await _grade_outcome(response_str[: decision.first_boxed_end])
            if isinstance(regrade, dict):
                score = regrade["score"]
                reward_extra_info.update(regrade)
            else:
                score = regrade
            reward_extra_info["acc"] = score
        if decision.skip_verifier:
            for reward_type in self.step_reward_types:
                reward_extra_info[f"{reward_type}_step_reward"] = list(decision.process_rewards)
            return {"reward_score": score, "reward_extra_info": reward_extra_info}
        penalty_step_indices = decision.penalty_step_indices
        # The whole-response Isabelle call must not see the invalid steps (each is already
        # penalized by index above) nor anything past the committed first \boxed, but it must
        # keep the answer text even when the unclosed tail block swallowed the \boxed line;
        # the per-step FOL loop skips the same indices instead.
        response_for_verifier = verifier_response(response_str, decision)
        # Reasoning tags outside every step block and each \boxed after the committed
        # first one cost one penalty_score at their own token position (mapped here, where
        # the tokenizer lives); appended per reward type below.
        extra_penalty_char_ends = list(decision.extra_boxed_char_ends)
        if decision.stray_tag_char_end is not None:
            extra_penalty_char_ends.append(decision.stray_tag_char_end)
        extra_penalty_positions = [
            char_end_to_token_pos(valid_response_ids, self.tokenizer,
                                  char_end, int(valid_response_length))
            for char_end in extra_penalty_char_ends
        ]

        # 2.2 Extract prompt text for reward functions that need it
        raw_prompt = data_item.non_tensor_batch.get("raw_prompt", [])
        if raw_prompt is not None and len(raw_prompt) > 0:
            # raw_prompt is a list of message dicts; take the last user message content
            prompt_text = raw_prompt[-1]["content"] if isinstance(raw_prompt[-1], dict) else str(raw_prompt[-1])
        else:
            prompt_text = ""

        # 2.3 Compute process rewards for each step_reward_type (parallel)
        for reward_type in self.step_reward_types:
            reward_fn = self.step_reward_fns.get(reward_type)
            if reward_fn is None:
                raise ValueError(f"Unknown step reward type: {reward_type}")

            fol_shared_state = None
            if reward_type == "fol" and (self.api_config or {}).get("fol_task_type") == "math":
                from verl.utils.reward_score.formal_verify import \
                    compute_solution_reward_isabelle
                loop = asyncio.get_event_loop()
                isabelle_rewards, isabelle_debug = await loop.run_in_executor(
                    self._executor,
                    lambda: compute_solution_reward_isabelle(
                        problem=prompt_text,
                        response=response_for_verifier,
                        ground_truth=(extra_info or {}).get(
                            "math_final_answer",
                            (extra_info or {}).get("answer", "")),
                        api_config=self.api_config,
                        return_debug=True,
                        # Steps beyond penalty_max_steps get penalty_score and
                        # Their per-step results are not used (see mapping below).
                        # don't waste judge/prover time computing them.
                        max_steps=self.penalty_max_steps,
                    ),
                )
                step_rewards = []
                ri = 0
                for i, (_step_text, token_end_pos) in enumerate(step_positions):
                    if i in penalty_step_indices:
                        step_rewards.append((int(token_end_pos), self.penalty_score))
                    elif ri < len(isabelle_rewards):
                        step_rewards.append((int(token_end_pos), float(isabelle_rewards[ri])))
                        ri += 1
                    else:
                        step_rewards.append((int(token_end_pos), 0.0))
                for extra_pos in extra_penalty_positions:
                    place_extra_penalty(step_rewards, extra_pos, self.penalty_score,
                                        max(0, int(valid_response_length) - 1))
                step_rewards.sort(key=lambda item: item[0])
                reward_extra_info[f"{reward_type}_step_reward"] = step_rewards
                # Per-response Isabelle metrics (mirror Z3 fol_judge/* pattern).
                # Reward manager aggregates across batch; single-value scalars
                # are averaged downstream.
                d = isabelle_debug or {}
                reward_extra_info["isabelle_format_ok"] = int(bool(d.get("format_ok")))
                reward_extra_info["isabelle_givens_ok"] = int(bool(d.get("givens_ok")))
                reward_extra_info["isabelle_steps_ok"] = int(bool(d.get("steps_ok")))
                reward_extra_info["isabelle_outcome_correct"] = int(bool(d.get("outcome_correct")))
                reward_extra_info["isabelle_n_steps"] = int(d.get("n_steps") or 0)
                reward_extra_info["isabelle_verified_steps"] = int(d.get("verified_steps") or 0)
                reward_extra_info["isabelle_rewarded_steps"] = int(d.get("rewarded_steps") or 0)
                reward_extra_info["isabelle_neutral_steps"] = int(d.get("neutral_steps") or 0)
                reward_extra_info["isabelle_guard_failed_steps"] = int(d.get("guard_failed_steps") or 0)
                # Per-symbol pattern counts (o/x/c/g/m), same priority as the
                # printed pattern. g is also kept split above (neutral/guard).
                reward_extra_info["isabelle_o_steps"] = int(d.get("o_steps") or 0)
                reward_extra_info["isabelle_x_steps"] = int(d.get("x_steps") or 0)
                reward_extra_info["isabelle_c_steps"] = int(d.get("c_steps") or 0)
                reward_extra_info["isabelle_u_steps"] = int(d.get("u_steps") or 0)
                reward_extra_info["isabelle_g_steps"] = int(d.get("g_steps") or 0)
                reward_extra_info["isabelle_m_steps"] = int(d.get("m_steps") or 0)
                reward_extra_info["isabelle_t_steps"] = int(d.get("t_steps") or 0)
                reward_extra_info["isabelle_judge_calls_givens"] = int(d.get("translation_attempts_givens") or 0)
                reward_extra_info["isabelle_judge_calls_steps"] = int(d.get("translation_attempts_steps") or 0)
                reward_extra_info["isabelle_judge_calls_total"] = int(
                    (d.get("translation_attempts_givens") or 0)
                    + (d.get("translation_attempts_steps") or 0))
                # Wall profile + cache/restart gauges (2026-07-11 review #6).
                reward_extra_info["isabelle_judge_http_wall_time"] = float(
                    d.get("judge_http_wall_time") or 0.0)
                reward_extra_info["isabelle_translate_validate_wall_time"] = float(
                    d.get("translate_validate_wall_time") or 0.0)
                reward_extra_info["isabelle_prove_calls"] = int(d.get("prove_calls") or 0)
                reward_extra_info["isabelle_prove_queue_time"] = float(d.get("prove_queue_time") or 0.0)
                reward_extra_info["isabelle_prove_run_time"] = float(d.get("prove_run_time") or 0.0)
                reward_extra_info["isabelle_prove_cache_hits"] = int(d.get("prove_cache_hits") or 0)
                reward_extra_info["isabelle_reward_wall_time"] = float(d.get("reward_wall_time") or 0.0)
                reward_extra_info["isabelle_pool_restarts"] = int(d.get("pool_restarts") or 0)
                reward_extra_info["isabelle_thm_cache_hit_rate"] = float(d.get("thm_cache_hit_rate") or 0.0)
                reward_extra_info["isabelle_tr_cache_hit_rate"] = float(d.get("tr_cache_hit_rate") or 0.0)
                # Real HTTP judge load vs per-layer cache reuse (2026-07-11).
                reward_extra_info["isabelle_judge_http_calls"] = int(d.get("judge_http_calls") or 0)
                reward_extra_info["isabelle_judge_retry_calls"] = int(d.get("judge_retry_calls") or 0)
                reward_extra_info["isabelle_translation_mem_hits"] = int(d.get("translation_mem_hits") or 0)
                reward_extra_info["isabelle_translation_disk_hits"] = int(d.get("translation_disk_hits") or 0)
                reward_extra_info["isabelle_translation_shared_hits"] = int(d.get("translation_shared_hits") or 0)
                reward_extra_info["isabelle_translation_xproc_hits"] = int(d.get("translation_xproc_hits") or 0)
                reward_extra_info["isabelle_translation_failures"] = int(d.get("translation_failures") or 0)
                # Per-step verification result symbols (o=rewarded, x=unverified, c=premises-inconsistent, m=transcription-missing, g=guard-failed) for the [Step Rewards] sample print.
                reward_extra_info["isabelle_pattern"] = str(d.get("pattern") or "")
                # ALWAYS set the key (empty string when no error): batch
                # collation asserts every response has the same
                # reward_extra_info keys, so a conditional key crashes the
                # step whenever only some responses error.
                reward_extra_info["isabelle_error"] = str(d.get("error") or "")
                continue

            if reward_type == "fol":
                from verl.utils.reward_score.formal_verify import (
                    _has_student_premise_conclusion_duplicate,
                    prepare_fol_shared_state,
                )
                from verl.utils.step_splitter import check_step_format

                # Some responses have only malformed XML steps or copied
                # premise/conclusion duplicates. compute_step_reward_fol will
                # fail them before using declarations, so skip the expensive
                # response-level declaration call unless at least one step can
                # actually reach the FOL judge.
                needs_fol_shared_state = False
                for i, (step_text, _token_end_pos) in enumerate(step_positions):
                    if i in penalty_step_indices:
                        continue
                    if "<step>" in step_text and not check_step_format(step_text):
                        continue
                    if _has_student_premise_conclusion_duplicate(step_text):
                        continue
                    needs_fol_shared_state = True
                    break

                if needs_fol_shared_state:
                    loop = asyncio.get_event_loop()
                    fol_shared_state = await loop.run_in_executor(
                        self._executor,
                        lambda: prepare_fol_shared_state(
                            prompt_text,
                            api_config=self.api_config,
                            extra_info=extra_info,
                        ),
                    )

            # Pre-build all step histories so calls can run in parallel. A penalized step's
            # text must not leak into later steps' cumulative context either (`step` /
            # `dependency_graph` modes): the history keeps the current step and only the
            # non-penalized earlier steps.
            call_args = []
            for i, (step_text, token_end_pos) in enumerate(step_positions):
                if i in penalty_step_indices:
                    continue
                history = [s for j, (s, _) in enumerate(step_positions[: i + 1])
                           if j not in penalty_step_indices]
                call_args.append((step_text, prompt_text, history, token_end_pos))

            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    self._executor,
                    lambda args=args: reward_fn(
                        args[0], args[1], args[2],
                        api_config=self.api_config, extra_info=extra_info,
                        **({"random_seed": self.random_reward_seed} if reward_type == "random" else {}),
                        **({"return_debug": True} if reward_type == "fol" else {}),
                        **({"fol_shared_state": fol_shared_state} if reward_type == "fol" else {}),
                    ),
                )
                for args in call_args
            ]
            scores = await asyncio.gather(*futures)

            step_rewards = []
            fol_debug = []
            fol_judge_prompt_tokens = 0
            fol_judge_completion_tokens = 0
            fol_judge_total_tokens = 0
            fol_judge_calls = 0
            fol_cache_hits = 0
            fol_verifier_steps = 0
            fol_entailed_steps = 0
            fol_not_entailed_steps = 0
            fol_invalid_translation_steps = 0
            fol_invalid_expression_steps = 0
            fol_expression_repair_steps = 0
            fol_autofilled_quantifier_steps = 0
            fol_autofilled_free_identifier_steps = 0
            fol_autofilled_symbolic_constant_steps = 0
            fol_sort_mismatch_steps = 0
            fol_leakage_steps = 0
            fol_student_duplicate_steps = 0
            fol_declaration_failed_steps = 0
            fol_format_failed_steps = 0
            for args, score_item in zip(call_args, scores):
                if reward_type == "fol" and isinstance(score_item, dict):
                    score_value = float(score_item.get("score", 0.0))
                    step_debug = score_item.get("debug", {})
                    fol_debug.append(step_debug)
                    fol_verifier_steps += 1
                    if isinstance(step_debug, dict):
                        if step_debug.get("cache_hit"):
                            fol_cache_hits += 1
                        judge_usage = step_debug.get("judge_usage", {})
                        if isinstance(judge_usage, dict):
                            fol_judge_prompt_tokens += int(judge_usage.get("prompt_tokens", 0) or 0)
                            fol_judge_completion_tokens += int(judge_usage.get("completion_tokens", 0) or 0)
                            fol_judge_total_tokens += int(judge_usage.get("total_tokens", 0) or 0)
                            fol_judge_calls += int(judge_usage.get("calls", 0) or 0)
                        z3_output = str(step_debug.get("z3_output", "") or "")
                        if "SUCCESS_ENTAILED" in z3_output:
                            fol_entailed_steps += 1
                        if "FAILED_NOT_ENTAILED" in z3_output:
                            fol_not_entailed_steps += 1
                        if step_debug.get("translation_failed_closed") or "FAILED_INVALID_TRANSLATION" in z3_output:
                            fol_invalid_translation_steps += 1
                        if (
                            step_debug.get("invalid_expression_syntax")
                            or step_debug.get("invalid_expression_syntax_initial")
                            or "FAILED_INVALID_EXPRESSION" in z3_output
                        ):
                            fol_invalid_expression_steps += 1
                        if int(step_debug.get("expression_correction_attempts", 0) or 0) > 0:
                            fol_expression_repair_steps += 1
                        if step_debug.get("autofilled_quantifier_variables"):
                            fol_autofilled_quantifier_steps += 1
                        if step_debug.get("autofilled_free_identifiers"):
                            fol_autofilled_free_identifier_steps += 1
                        if step_debug.get("autofilled_symbolic_constants"):
                            fol_autofilled_symbolic_constant_steps += 1
                        if (
                            step_debug.get("translation_sort_mismatches")
                            or step_debug.get("invalid_translation_reason") == "z3_sort_mismatch"
                        ):
                            fol_sort_mismatch_steps += 1
                        if step_debug.get("conclusion_leakage_detected") or "FAILED_LEAKED_CONCLUSION" in z3_output:
                            fol_leakage_steps += 1
                        if step_debug.get("student_premise_conclusion_duplicate"):
                            fol_student_duplicate_steps += 1
                        if step_debug.get("declaration_failed_closed"):
                            fol_declaration_failed_steps += 1
                        if step_debug.get("format_failed_closed"):
                            fol_format_failed_steps += 1
                else:
                    score_value = float(score_item)
                step_rewards.append((int(args[3]), score_value))
            for i in sorted(penalty_step_indices):
                if i < len(step_positions):
                    _, token_end_pos = step_positions[i]
                    step_rewards.append((int(token_end_pos), self.penalty_score))
            for extra_pos in extra_penalty_positions:
                place_extra_penalty(step_rewards, extra_pos, self.penalty_score,
                                    max(0, int(valid_response_length) - 1))
            step_rewards.sort(key=lambda item: item[0])

            key = f"{reward_type}_step_reward"
            reward_extra_info[key] = step_rewards
            if reward_type == "fol" and fol_debug:
                reward_extra_info["fol_debug"] = fol_debug
                reward_extra_info["fol_judge_prompt_tokens"] = fol_judge_prompt_tokens
                reward_extra_info["fol_judge_completion_tokens"] = fol_judge_completion_tokens
                reward_extra_info["fol_judge_total_tokens"] = fol_judge_total_tokens
                reward_extra_info["fol_judge_calls"] = fol_judge_calls
                reward_extra_info["fol_judge_completion_tokens_per_call"] = (
                    float(fol_judge_completion_tokens) / fol_judge_calls if fol_judge_calls > 0 else 0.0
                )
                reward_extra_info["fol_cache_hit_rate"] = (
                    float(fol_cache_hits) / fol_verifier_steps if fol_verifier_steps > 0 else 0.0
                )
                reward_extra_info["fol_verifier_steps"] = fol_verifier_steps
                reward_extra_info["fol_entailed_steps"] = fol_entailed_steps
                reward_extra_info["fol_not_entailed_steps"] = fol_not_entailed_steps
                reward_extra_info["fol_invalid_translation_steps"] = fol_invalid_translation_steps
                reward_extra_info["fol_invalid_expression_steps"] = fol_invalid_expression_steps
                reward_extra_info["fol_expression_repair_steps"] = fol_expression_repair_steps
                reward_extra_info["fol_autofilled_quantifier_steps"] = fol_autofilled_quantifier_steps
                reward_extra_info["fol_autofilled_free_identifier_steps"] = fol_autofilled_free_identifier_steps
                reward_extra_info["fol_autofilled_symbolic_constant_steps"] = fol_autofilled_symbolic_constant_steps
                reward_extra_info["fol_sort_mismatch_steps"] = fol_sort_mismatch_steps
                reward_extra_info["fol_leakage_steps"] = fol_leakage_steps
                reward_extra_info["fol_student_duplicate_steps"] = fol_student_duplicate_steps
                reward_extra_info["fol_declaration_failed_steps"] = fol_declaration_failed_steps
                reward_extra_info["fol_format_failed_steps"] = fol_format_failed_steps

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
