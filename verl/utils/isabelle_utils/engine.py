"""Isabelle/HOL step-level verification for mathematical reasoning.

A response step passes through three representation forms.

A. Natural language

`problem` is the original problem text.

Each parsed response step is represented by `NaturalLanguageStep`:

* `nl_premises`: the contents of the step's `<premise>` tags;
* `nl_conclusion`: the contents of its first `<conclusion>` tag;
* `nl_step_text`: the complete text inside its `<step>` tags.


A->B: Formalization Stage

B. PyExpr (skippable if direct-domain, such as group, ring, and field statements)


`VARS`: type declarations
`GIVEN`: original problem conditions translated to pyexpr, including exactly one condition that names the requested quantity through `answer`.
e.g.
    VARS: price real, count int, total real, answer real
    GIVEN: price == 5
    GIVEN: count == 3
    GIVEN: answer == total
stored into `PyExprGiven` as PyExpr layer information.
    PyExprGiven(pyexpr_variable_types=[("price", "real"), ("count", "int"), ("total", "real"), ("answer", "real"),],
        pyexpr_givens=["price == 5", "count == 3", "answer == total",],
    )

For each general response step:
`pyexpr_premises`: pyexpr translated from `nl_premises`;
`pyexpr_conclusion`: pyexpr translated `nl_conclusion`; usually a conjunction
`pyexpr_definitions`: definitions of fresh intermediate variables, from the def part of conjuncts in `pyexpr_conclusion`;
`pyexpr_claims`: real conclusion claim, from remaining conjuncts that must be proved.


B->C: Preparation Stage

C. Isabelle

`isabelle_fixes`: var/func type declarations (e.g.  fixes x :: real and n :: int), held & keep updated from pyexpr, across steps in one whole resp
* var names have `pv_` prefix to avoid collisions with Isabelle constants. 

`IsabelleStep` holds the current step's pyexpr and Isabelle representation.

`isabelle_premises` (a concated list in `IsabelleStep`):
    1. pyexpr givens (original conditions)
        translator 输出的 GIVEN 转成 Isabelle 后的内容
        name: g0, g1, ...
    2. complete conclusion list of `pyexpr_conclusion` of prev steps, that can be presented by a general step (prev step conclusions)

        - 此前 general step 的完整 `pyexpr_conclusion`；
        - 可安全桥接的 direct-domain conclusion，例如
            `order G = 15` → `order_G == 15` → Isabelle term。
        Whitelisted arithmetic-valued direct conclusions can be bridged into later general steps; non-bridgeable abstract-algebra conclusions remain excluded.
        name: s0, s1, ...
    3. pyexpr_premises transpiled (translated to Isabelle) (curr step premises)
        name: p{k}_{i}
    4. new definitions recognized from current step pyexpr_conclusion (curr step definitions from conclusion)
        name: d{k}_{i}
    (5. In the final verification/theorem proof, the real premises called will include provable nonzero conditions for division, tried respectively from `isabelle_nonzero_divisors`)

`DirectDomainStep` works samely as `IsabelleStep` but with another list of premises:
(general_conclusions, previous same-domain direct conclusions, current direct step's own premises)

`isabelle_claims`: pyexpr_claims transpiled (translated to Isabelle) (curr step conclusion claim)
"""
import atexit
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass

import verl.utils.isabelle_utils.server_pool as server_pool
import verl.utils.isabelle_utils.stages.formalization as formalization
import verl.utils.isabelle_utils.stages.preparation as preparation
import verl.utils.isabelle_utils.stages.verification as verification
import verl.utils.isabelle_utils.state_classes as state_classes
import verl.utils.step_splitter as step_splitter
from verl.utils.reward_score.math_dapo import (last_boxed_only_string,
                                               remove_boxed)


def _boxed_answer(response):
    """Content of the last \\boxed{...} in `response`, or None.
    Reuses the canonical math_dapo extractor (last_boxed_only_string + remove_boxed)
    so the isabelle path shares ONE boxed parser with the math outcome reward instead of a private copy."""
    s = last_boxed_only_string(response)
    return remove_boxed(s) if s is not None else None


def corrupt_steps(nl_steps, ground_truth):
    """Negative debugging for the verifier itself: deliberately break a correct solution so it can be checked to FAIL.

    Picks a middle step and increments one non-trivial number in its conclusion (not 0/1/2, not the ground-truth answer),
    e.g. turning "= 48" into "= 49". If the verifier still rewards the corrupted solution it is not actually checking anything.

    Returns the edit {step, old, new}, or None when no eligible number exists.
    NEVER use this in training."""
    order = sorted(range(len(nl_steps)), key=lambda i: abs(i - (len(nl_steps) - 1) / 2))
    for k in order:
        con = nl_steps[k].nl_conclusion
        cands = [m for m in re.finditer(r"(?<![\w.])(\d+)(?![\w.])", con)
                 if m.group(1) not in {"0", "1", "2"} and m.group(1) != ground_truth.strip()]
        if not cands:
            continue
        m = cands[len(cands) // 2]
        old = m.group(1)
        new = str(int(old) + 1)
        new_con = con[: m.start()] + new + con[m.end():]
        nl_steps[k].nl_step_text = nl_steps[k].nl_step_text.replace(con, new_con)
        nl_steps[k].nl_conclusion = new_con
        return {"step": k, "old": old, "new": new}
    return None


@dataclass
class IsabelleConfig:
    translator_url: str = "http://127.0.0.1:4873/v1"
    translator_model: str = "Qwen3.6-35B-A3B"
    max_model_len: int = 12288
    pool_workers: int = 32
    
    # A use_theories call that exceeds this deadline fails instead of blocking reward computation. The value corresponds to the Z3 path's fol_timeout setting.
    verify_timeout: float = 60.0
    # The reward API's api_timeout config controls each translator HTTP request
    api_timeout: float = 240.0
    # Each worker's Poly/ML process tree may use at most this many GB before the pool restarts it.
    # The value is a Hydra config setting, not an environment variable; 12 GB was validated under the 300 GB cgroup.
    each_worker_proc_tree_mem_max_gb: float = 12.0
    # Steps per translator call. A solution with at most this many steps is translated in a single call; a longer one is split into chunks of this size, with later chunks shown the earlier props as context. Sized to what one prompt handles well at max_model_len; a Hydra config knob, not a hardcoded constant.
    translate_chunk_steps: int = 20
    # Parallelism of per-step prover checks within one response: how many steps' consistency probes and claim proofs run against the pool at the same time (1 = strictly sequential). Hydra config knob, was the ISABELLE_STEP_CHECK_PAR env var.
    step_check_parallelism: int = 4
    # A prover process burning ~100% CPU for longer than this many seconds is treated as a runaway (native tactic that ignored the cooperative timeout) and reaped. Hydra config knob, was the ISABELLE_RUNAWAY_CPU_S env var.
    runaway_cpu_s: float = 90.0


# Response Processing
class PoolVerifier:
    """Normalized prover access for one response, built by process_one and handed to the pipeline stages as `verify`; the only implementation that parameter ever receives.

    A call routes the theorem through pool.submit's process-wide FIFO when available (real pools) and falls back to pool.check for pools without it (mocks, older callers).
    The returned value is the single normalization boundary: everything past it sees a classified VerificationOutcome,
    whether the pool (or a test mock) returned a typed result or a legacy dict.
    Every call also lands in the response's prove_* profile counters, locked because premise checks and claim checks run concurrently.
    `submit` stays exposed for the orchestrator's future-based consistency scheduling, which records those results itself on collection.
    """

    def __init__(self, pool, profile):
        self.pool = pool
        self.profile = profile
        self.submit = getattr(pool, "submit", None)
        self._profile_lock = threading.Lock()

    def record(self, result):
        with self._profile_lock:
            self.profile["prove_calls"] += 1
            self.profile["prove_queue_time"] += result.queue_wait
            self.profile["prove_run_time"] += result.check_time
            if result.cache_hit:
                self.profile["prove_cache_hits"] += 1

    def __call__(self, theorem):
        result = state_classes.VerificationOutcome.from_raw(
            self.submit(theorem).result() if self.submit is not None
            else self.pool.check(theorem))
        self.record(result)
        return result


def process_one_response(response_id, item, pool, config, outdir=None, corrupt=False,
                max_steps: int = 0):
    problem = item["problem"]
    parsed_steps = step_splitter.parse_xml_steps(item["response"])
    nl_steps = None if parsed_steps is None else [state_classes.NaturalLanguageStep(nl_premises=d["premises"], nl_conclusion=d["conclusion"], nl_step_text=d.get("block_text", "")) for d in parsed_steps]
    # Verify only the first max_steps steps (0 = no limit). The step reward manager penalizes steps beyond penalty_max_steps and DISCARDS their results, so translating/verifying them is pure waste -- and step-inflated responses (15-20 steps) are exactly the reward-time stragglers. Truncated steps are never translated, never enter any translator prompt, and are never referenced.
    if max_steps > 0 and nl_steps is not None and len(nl_steps) > max_steps:
        nl_steps = nl_steps[:max_steps]
    box = _boxed_answer(item["response"])
    record = state_classes.ResponseVerificationResult(
        response_id=response_id, dataset=item["dataset"], idx=item["idx"],
        sample=item["sample"], format_ok=nl_steps is not None,
        boxed=box,
        outcome_correct=box is not None and box == item["ground_truth"].strip())
    # Per-response wall profile
    profile_results = {"translator_http_time": 0.0, # translator_http_time measures only requests.post wall time and therefore does not overlap prover queue or run time. 
                       "translate_validate_time": 0.0, # translate_validate_time is retained as the end-to-end cached translate+parse+validate wall and may include prover-backed validation.
            "prove_calls": 0, "prove_queue_time": 0.0,
            "prove_run_time": 0.0, "prove_cache_hits": 0}
    record.prof = profile_results
    verify = PoolVerifier(pool, profile_results)
    if nl_steps is not None and corrupt:
        record.corrupt_info = corrupt_steps(nl_steps, item["ground_truth"])
        if record.corrupt_info is None:
            record.format_ok = False
            return record.to_dict()
    if nl_steps is None:
        # No per-response print -> format_ok=0 in metrics
        return record.to_dict()
    record.n_steps = len(nl_steps)

    # Form A (natural language) to form B (constrained pyexpr).
    formalized = formalization.formalize(problem, nl_steps, config, verify, profile_results)

    record.translation_record_from_problem = formalized.translation_record_from_problem
    if not formalized.givens_ok:
        # No per-response print -> givens_ok=0 in metrics.
        return record.to_dict()
    record.givens_ok = True
    pyexpr_givens, record.translation_record_from_steps = formalized.pyexpr_givens, formalized.translation_record_from_steps
    if not formalized.steps_ok:
        # No per-response print -> steps_ok=0 in metrics.
        return record.to_dict()
    record.steps_ok = True

    # Form B (constrained pyexpr) to form C (Isabelle).
    prepared = preparation.prepare(formalized)
    isabelle_fixes = prepared.isabelle_fixes

    if outdir is not None:
        (outdir / f"{response_id:03d}_debug.json").write_text(json.dumps(
            {"isabelle_fixes": isabelle_fixes, "pyexpr_givens": pyexpr_givens,
             "pyexpr_conclusions": [s.pyexpr_conclusion if isinstance(s, state_classes.PyExprStep)
                                    else None for s in formalized.steps],
             "pyexpr_premises": [list(s.pyexpr_premises) if isinstance(s, state_classes.PyExprStep)
                                 else None for s in formalized.steps],
             "isabelle_problem_conditions": prepared.isabelle_problem_conditions,
             "isabelle_step_conclusions": [pair[0] if pair else None
                                           for pair in prepared.isabelle_step_conclusions],
             "nl_steps": [asdict(nl_step) for nl_step in nl_steps]}, indent=2))

    # Form C (Isabelle) to the final result for every general and direct-domain step.
    verified = verification.verify_response(
        prepared, verify,
        step_check_parallelism=max(1, int(config.step_check_parallelism)),
        pool_workers=getattr(pool, "num_workers", 4))
    record.steps.extend(verified.steps)
    record.premise_consistency_inconsistent_at = verified.boundaries.inconsistent_from_step
    if verified.boundaries.unknown_from_step is not None:
        record.premise_consistency_unknown_at = verified.boundaries.unknown_from_step
    record.pattern = verified.pattern
    # No per-response print: the pattern + outcome converge into the trainer's single [Step Rewards] sample line (ray_trainer.py) once per step.
    # The record leaves the engine as plain data (to_dict serializes the typed step results too).
    return record.to_dict()

class IsabelleEngine:
    """Isabelle/HOL verification engine for math step-level rewards.

    Wraps process_one: given a problem + policy response + ground truth,
    returns per-step verification results.
    """

    def __init__(self, config: IsabelleConfig | None = None):
        self.config = config or IsabelleConfig()
        # base_dir MUST be per-process: multiple RewardLoopWorker processes each build their own engine,
        # and IsabelleWorker.start() executes rmtree to its master_dir.
        # A shared path lets a later process wipe an earlier process's live worker dirs (ENOENT on the next theory write).
        # verify_timeout is a POOL arg (not a module-global mutation), so two engines cannot clobber each other's timeout.
        self.pool = server_pool.IsabelleServerPool(
            num_workers=self.config.pool_workers,
            base_dir=f"/tmp/isabelle_pool_engine_{os.getpid()}",
            each_worker_proc_tree_mem_max_gb=self.config.each_worker_proc_tree_mem_max_gb,
            verify_timeout=self.config.verify_timeout,
            runaway_cpu_seconds=self.config.runaway_cpu_s)
        self.pool.start()
        atexit.register(self._safe_shutdown)

    def _safe_shutdown(self):
        try:
            self.pool.shutdown()
        except Exception:
            pass

    def verify_solution(self, problem: str, response: str,
                        ground_truth: str, dataset: str = "math",
                        idx: int = 0, sample: int = 0,
                        max_steps: int = 0) -> dict:
        """Verify a complete solution. Returns the same dict as process_one."""
        item = {
            "problem": problem,
            "response": response,
            "ground_truth": ground_truth,
            "dataset": dataset,
            "idx": idx,
            "sample": sample,
        }
        _time_start = time.time()
        out = process_one_response(idx, item, self.pool, self.config, max_steps=max_steps)
        # Total reward wall for this response (includes translate + prove + queue waits + CPU prep); the gap vs the parts is scheduling overhead.
        if isinstance(out.get("prof"), dict):
            out["prof"]["reward_wall_time"] = time.time() - _time_start
        return out

    def shutdown(self):
        self.pool.shutdown()
