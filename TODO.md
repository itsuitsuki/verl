# FOL / Tree-GAE Follow-Up Notes

This note summarizes the current debugging state for the LogiQA2K 1.5B FOL reward runs, so another person can continue without reconstructing the full chat history.

## Current Priority TODO, 2026-05-05

Treat this section as the authoritative next-step list. Older sections below are historical context unless they agree with this section.

### Active / Latest Runs

- GSM8K Qwen2.5-7B FOL Step-GDPO, `train_gsm8k7b_fol_gpu34_v1.log`
  - Completed on `013_2` GPU3/4 with judge load balancer `http://127.0.0.1:4874/v1`.
  - Final val/test acc `90.447%` at step `934`; best val/test acc `91.054%` at step `200`; initial val/test acc `83.776%` at step `0`.
  - Final train score `100.000%`; recent-10 train score mean about `94.531%`.
  - Late-run bottleneck was mostly 7B actor generation/update rather than judge: recent averages were generation `~41s`, actor update `~46s`, and FOL reward compute `~16s`.
- ReClor Qwen2.5-7B FOL Step-GDPO, `train_reclor7b_fol_gpu1_v1.log`
  - Not currently active. The run reached about step `206/579` and then stopped when `017_2` became unavailable / lost the judge route.
  - Latest val acc `74.600%` at step 200; best val acc `75.000%` at step 100; initial val acc `71.000%` at step 0.
  - Latest train score before crash `73.438%`; recent-10 train score mean about `77.891%`.
  - Restart only after a clean two-GPU slot and stable `:4874` judge route are available.
- LogiQA Qwen2.5-7B FOL Step-GDPO, `train_logiqa7b_fol_gpu34_v2.log`
  - Not currently active. The two-GPU attempt reached only step `0` validation, with initial val acc `49.002%`, then failed before training updates.
  - The previous one-GPU attempt, `train_logiqa7b_fol_gpu3_v1.log`, OOMed during actor / rollout weight update.
  - The two-GPU v2 attempt also needs a config change before rerun: one launch hit CUDA/cumem OOM, and the lower-memory launch failed vLLM startup with no available KV cache blocks.

### Completed Baselines To Use In Tables

- GSPO outcome-only, `train_gspo_outcome_only_logiqa_full_prompt1_gpu4_v1.log`
  - Completed. Final val acc `0.3687`; best val acc `0.4885` at step 1200.
  - This is not a separate reward baseline from DAPO; it is DAPO/outcome-only reward with GSPO policy loss, sequence-level mean aggregation, and KL disabled.
- DAPO outcome-only, `train_dapo_outcome_only_logiqa_full_prompt1_gpu4_v1.log`
  - Completed. Final val acc `0.3902`; best val acc `0.4531` at step 750.
- A800 format Step-GDPO, `train_format_step_gdpo_a800_gpu0_v1.log`
  - Completed. Final val acc `0.3625`; best val acc `0.4470` at step 600.
- A800 format Tree-GAE, `train_format_tree_gae_a800_gpu1_v1.log`
  - Completed. Final val acc `0.2888`; best val acc `0.4040` at step 550.
- A800 outcome-only Tree-GAE, `train_outcome_tree_gae_a800_gpu2_v1.log`
  - Completed. Final val acc `0.3118`; best val acc `0.3994` at step 400.
- ReClor 1.5B completed baselines:
  - FOL Step-GDPO v4, `train_fol_step_gdpo_reclor_gpu2_v4.log`: best `58.600%` at step 450, final `58.000%`.
  - FOL Step-GDPO KL-off, `train_reclor_fol_kloff_gpu2_v1.log`: stopped after step 404; best `56.400%` at step 200, latest `54.800%` at step 400.
  - Format Step-GDPO, `train_format_step_gdpo_reclor_gpu2_v1.log`: best/final `59.000%` at step 579.
  - Self-eval Step-GDPO, `train_self_eval_step_gdpo_reclor_gpu2_v1.log`: best `58.400%` at step 450, final `57.800%`.
  - GSPO outcome-only, `train_gspo_outcome_only_reclor_gpu4_v1.log`: best `59.600%` at step 400, final `55.000%`.
  - DAPO outcome-only, `train_dapo_outcome_only_reclor_gpu3_v1.log`: best `59.400%` at step 550, final `58.800%`.
- GSM8K 1.5B completed baselines:
  - Qwen2.5-1.5B-Instruct base, `test_base_qwen25_15b_gsm8k_0172_gpu2.log`: test `40.788%`.
  - FOL Step-GDPO, `train_fol_step_gdpo_gsm8k_gpu4_v1.log`: best `75.739%` at step 700/900, final `74.223%`.
  - Format Step-GDPO, `train_format_step_gdpo_gsm8k_0172_gpu0_v1.log`: best `77.331%` at step 700, final `74.905%`.
  - Self-eval Step-GDPO, `train_self_eval_step_gdpo_gsm8k_gpu3_v1.log`: best/final `79.682%` at step 934.
  - GSPO outcome-only, `train_gspo_outcome_only_gsm8k_0172_gpu1_v1.log`: best `77.938%` at step 800, final `75.284%`.
  - DAPO outcome-only, `train_dapo_outcome_only_gsm8k_0172_gpu3_v1.log`: best `77.180%` at step 400, final `73.086%`.

### Next Experiments

1. Finish the Qwen2.5-7B FOL Step-GDPO main experiments on LogiQA and ReClor, including train + test/eval.
   - LogiQA: run as a two-GPU FOL job, but do not repeat the failed v1/v2 launch configs.
   - Before relaunching LogiQA 7B, adjust the rollout memory / tensor-parallel / offload settings so the actor update and vLLM KV cache both fit.
   - Use `CUDA_VISIBLE_DEVICES=3,4` on `013_2` if those GPUs remain free, with the existing `:4874` judge load balancer.
   - ReClor: resume or restart `train_reclor7b_fol_gpu1_v1.log` only after a stable two-GPU slot is available.
   - If using `017_2`, restore the reverse judge forward to `127.0.0.1:4874` before launching.
   - Test/evaluate best and final checkpoints. For ReClor, use labeled validation as the reportable held-out split unless a labeled test source is added.
2. Rerun LogiQA Qwen2.5-1.5B FOL Step-GDPO after the small rewarding change.
   - Treat this as the refreshed 1.5B main FOL run; compare against the previous v4 numbers and report best/final validation plus held-out test.
   - Keep judge / FOL settings fixed unless the rewarding change explicitly requires a config delta.
3. Run pure GRPO / outcome-only baselines with and without the reasoning prompt.
   - With prompt: use the same prompt-v2 data format as the FOL/format/self-eval runs.
   - Without prompt: remove the reasoning system prompt while keeping dataset splits, answer extraction, rollout count, and optimizer settings comparable.
   - Run train + held-out test, and use this as the prompt-dependence ablation for the paper tables.
4. Evaluate LogiQA-trained checkpoints out of domain.
   - Evaluate best/final LogiQA checkpoints on ReClor and GSM8K.
   - Label these as transfer / out-of-domain results, not in-domain test scores.
   - For ReClor, use labeled validation unless a labeled test source is added; for GSM8K, use the standard test parquet.
5. Rerun GSM8K Qwen2.5-Math-1.5B-Instruct FOL only with XML-compatible prompting / constraints.
   - The prior run stopped early because the math model emitted natural-language CoT without XML steps, so FOL process reward was skipped.
6. Use two experimental protocols in the paper.
   - In-domain protocol: train on dataset A and evaluate on A for controlled reward/method comparisons.
   - Universal-verifier protocol: train one mixed checkpoint, then evaluate it on multiple datasets.
   - Candidate logic mix: LogiQA + ReClor + AR-LSAT, tested on LogiQA/ReClor/AR-LSAT/FOLIO/FLDx2.
   - Candidate math mix: GSM8K + AQUA-RAT, tested on GSM8K/AQUA-RAT and selected math test sets.
7. Finish probe work before scaling the mixed-training runs.
   - FOLIO full and FLDx2 full: report strict/all FOL judge metrics separately.
   - ProcessBench GSM8K: use it as diagnostic F1 for step correctness, not as a training reward until quality is known.
8. Dataset expansion.
   - AR-LSAT preprocessing is ready for A/B/C/D/E labels; run a small FOL probe before training.
   - Add AQUA-RAT only through math mode, with A-E option handling.
   - Treat ARC as low-priority or non-FOL because most signal is world knowledge rather than Z3-verifiable logic.
9. Tree search remains lower priority.
   - Current evidence says tree collapse is not fixed by simply making the tree deeper.
   - If revisited, use outcome gating or a nontrivial-step constraint rather than another deeper-only run.
10. Reporting.
   - Keep README tables updated with best and final/current metrics to three decimals.
   - Separate validation from held-out test; for ReClor, the official test split is unlabeled, so use labeled validation unless a new labeled test source is added.
11. Related-work alignment for the paper.
   - Logic-RL (`arXiv:2502.14768`): cite as rule-based RLVR / strict-format logic training, not as a PRM baseline. Use our format and rule-reward ablations plus cross-dataset generalization to compare against this line.
   - AURORA (`arXiv:2502.11520`): cite as automated LLM-as-a-judge process-label generation plus universal PRM training. If we compare directly, use mixed-train multi-test evaluation and ProcessBench-style diagnostics rather than only A-on-A training.
   - ThinkPRM / Process Reward Models That Think (`arXiv:2504.16828`): cite as generative PRM with verification CoT. Our contrast is executable Z3 verification and fail-closed semantics; probes on ProcessBench, FOLIO, and FLDx2 are the minimum evidence for this comparison.
   - For PRM-style related work in general, keep both views: in-domain A->A tables for controlled training comparisons, and single-checkpoint mixed-train multi-test tables for "universal verifier / PRM" claims.

## Current Findings

- Latest active-run snapshot, 2026-04-30 17:36 UTC:
  - `train_fol_step_gdpo_gpu2_v4_2.log`
    - Experiment: `qwen1.5b_step_gdpo_fol_gpu2_v4`
    - GPU2, judge load balancer `http://127.0.0.1:4874/v1`
    - Progress `89/1844`, ETA about `20:38:14`, roughly `42.33s/it`.
    - Latest/best val acc so far: `0.33487` at step 50.
    - Latest train score: `0.53125`; recent-10 mean about `0.325`.
    - Bottleneck: `gen ~= 20.15s`, `update_actor ~= 12.73s`, FOL reward compute mean/max `8.21/18.21s`.
    - FOL judge pressure is still high: total tokens mean/max `6011/14979`; invalid translation rate about `8.46%`.
  - `train_fol_tree_gae_gpu3_v4.log`
    - Experiment: `qwen1.5b_tree_gae_fol_gpu3_v4`
    - GPU3, judge load balancer `http://127.0.0.1:4874/v1`
    - Progress `43/1844`, ETA about `22:10:27`, roughly `44.32s/it`.
    - Initial val acc: `0.27496`.
    - Latest train score: `0.28125`; recent-10 mean about `0.14375`.
    - Bottleneck: `tree_expansion ~= 20.65s`, mostly `tree/ext_prm_eval ~= 20.57s`, plus `update_actor ~= 13.81s`.
    - Latest printed tree finish reason was `{'stop': 60}`; no obvious length/repetition explosion in that sample.
  - `train_self_eval_step_gdpo_gpu4_v1.log`
    - Experiment: `qwen1.5b_step_gdpo_self_eval_gpu4_v1`
    - GPU4, self-eval endpoint `http://127.0.0.1:8199/v1`
    - Progress `3/1844`, ETA about `17:05:39`, roughly `33.43s/it`.
    - Initial val acc: `0.29186`.
    - Latest train score: `0.28125`; early recent mean about `0.3229`.
    - Bottleneck: `gen ~= 12.03s`, `update_actor ~= 12.72s`, reward compute mean/max `2.63/7.85s`.
- Latest judge-server finding, 2026-04-30:
  - Both 35B judge services now run with prefix caching enabled:
    - `4872`: GPU0/1, `--max-model-len 12288 --gpu-memory-utilization 0.95 --max-num-seqs 256 --enable-prefix-caching --max-cudagraph-capture-size 256`
    - `4873`: GPU5/6, same settings.
  - Load balancer `4874` is alive and balancing both backends.
  - Latest LB snapshot: `4872` had `9455` requests and `2` transient disconnect failures; `4873` had `9570` requests and `0` failures.
  - vLLM logs show recent `200 OK` traffic and prefix cache hit rate around `80%`; the two `4872` failures look like transient connection drops, not OOM or a dead judge.
  - This server-side prefix-cache fix is the main reason FOL became much faster than the earlier 60h+ ETA runs. Keep it in every future FOL judge launch.
- Latest completed / external baseline results:
  - DAPO outcome-only, `train_dapo_outcome_only_logiqa_full_prompt1_gpu4_v1.log`
    - Completed `1844/1844`.
    - Final val acc `0.39017` at step 1844.
    - Best val acc `0.45315` at step 750.
  - A800 format Step-GDPO, `train_format_step_gdpo_a800_gpu0_v1.log`
    - Last synced progress `913/1844`.
    - Latest val acc `0.44086` at step 900.
    - Best val acc `0.44700` at step 600.
  - A800 format Tree-GAE, `train_format_tree_gae_a800_gpu1_v1.log`
    - Last synced progress `1320/1844`.
    - Latest val acc `0.30261` at step 1300.
    - Best val acc `0.40399` at step 550.
  - A800 outcome-only Tree-GAE, `train_outcome_tree_gae_a800_gpu2_v1.log`
    - Last synced progress `1331/1844`.
    - Latest val acc `0.30876` at step 1300.
    - Best val acc `0.39939` at step 400.
- Latest FOL speed / format finding, 2026-04-30:
  - Current full-data FOL Step-GDPO v4 prints boxed-only samples such as `\boxed{{B}}`.
  - In XML mode, the shared splitter previously fell back to delimiter splitting when no `<step>` tags existed, so a boxed-only response became one fake step and still triggered expensive FOL declaration + translation judge calls.
  - A real printed sample recheck showed this failure mode can spend roughly `25.8s` in declaration/preprocess plus `12.4s` in verify/translation before returning `0.0`.
  - Current working-tree fix:
    - Step-GDPO and Tree-GAE now treat FOL/FOL-old runs with `use_xml_steps=true`, `penalty_on_bad_format=true`, and zero `<step>` tags as `bad_format(no_xml_step)`;
    - they assign `penalty_score` at the final valid response token;
    - they skip all external PRM / FOL judge calls for that response.
  - This is controlled by the existing `penalty_on_bad_format` and `penalty_score` config, not hardcoded.
- Latest FOL judge overflow finding, 2026-04-30:
  - Full LogiQA Step-GDPO FOL run with short response cap still crashed around step 9 because a declaration-repair judge call exceeded the 8192-token judge context.
  - The offending request was not the actor prompt. It was the FOL declaration repair path: long declaration system prompt + repeated bad declaration JSON + many duplicate-identifier validation errors.
  - The immediate code fix is in `verl/utils/fol_utils/engine.py`:
    - compact duplicate declaration payloads before repair;
    - locally accept the compacted payload if it already renders as valid Z3 declarations;
    - cap declaration repair validation errors;
    - use compact JSON and a short repair-specific system prompt instead of the full declaration prompt.
  - This should reduce fatal context-overflow risk even when the judge is not restarted with a larger max context.
- FOL step-GDPO v3 is the most plausible "multi-step and reasonably correct" run so far.
  - Best observed val: step 350, acc `0.38249`, val num_steps `7.11`.
  - It keeps multi-step generations, unlike tree runs.
  - Recent FOL judge metrics improved versus older v2/v2.1 runs, but leakage and sort/declaration failures still exist.
- FOL Tree-GAE shallow v3 achieved the highest current FOL-tree val acc but via short reasoning.
  - Best observed val: step 450, acc `0.43318`.
  - By the end it collapsed to very short paths, roughly one reasoning step / terminal answer.
  - Step 500 dropped to acc `0.37942`.
- FOL Tree-GAE deeper `(M,N,L,T)=(4,1,3,1)` did not solve the "multi-step and correct" target.
  - Step 50 val: acc `0.33487`, val num_steps `3.00`.
  - Step 100 val: acc `0.37174`, val num_steps `1.14`.
  - Step 250 val: acc `0.31490`, val num_steps `1.00`.
  - It became short as training progressed and was much slower due to FOL judge load.
  - The run was stopped after step 250/290-era evidence because it was worse than the shallow tree run and no longer represented multi-step search.
- Short-tree collapse is not FOL-specific.
  - Old outcome tree quickly collapsed to `num_steps/mean ~= 2`.
  - Old self-eval tree also collapsed to `num_steps/mean ~= 2`.
  - Old format tree resisted longer, but still shortened to about `2.4-2.7` steps late in training.
- Small actor capacity is likely a real factor.
  - 1.5B learns reward shortcuts and direct-answer strategies.
  - Stable "long chain + XML-valid + FOL-verifiable + correct" behavior may require a stronger actor.

## Current Runs To Watch

- Current FOL Step-GDPO v4:
  - Log: `train_fol_step_gdpo_gpu2_v4_2.log`
  - Experiment: `qwen1.5b_step_gdpo_fol_gpu2_v4`
  - Judge endpoint: load balancer `http://127.0.0.1:4874/v1`
  - GPU: `CUDA_VISIBLE_DEVICES=2`
  - Keep `+algorithm.fol_cumulative_mode=current_only` and `+algorithm.validate_with_step_reward=false`.
  - Current env: `REWARD_NUM_WORKERS=128`, `STEP_REWARD_MAX_WORKERS=32`, `FOL_OPENAI_MAX_INFLIGHT=512`.
  - Watch step 100/150/200 val acc, val num_steps, invalid translation rate, FOL token max, and whether no-step responses skip judge as expected.
- Current FOL Tree-GAE v4:
  - Log: `train_fol_tree_gae_gpu3_v4.log`
  - Experiment: `qwen1.5b_tree_gae_fol_gpu3_v4`
  - Judge endpoint: load balancer `http://127.0.0.1:4874/v1`
  - GPU: `CUDA_VISIBLE_DEVICES=3`
  - Keep `+algorithm.fol_cumulative_mode=current_only`, `+algorithm.validate_with_step_reward=false`, `+trainer.tree_defer_initial_ext_prm=true`, and `+trainer.tree_overlap_ext_prm=true`.
  - Current env: `REWARD_NUM_WORKERS=64`, `STEP_REWARD_MAX_WORKERS=32`, `FOL_OPENAI_MAX_INFLIGHT=512`.
  - Watch step 50/100 val acc, tree path length, leaf finish reasons, and whether it collapses to one-step direct answers.
- Current self-eval Step-GDPO:
  - Log: `train_self_eval_step_gdpo_gpu4_v1.log`
  - Experiment: `qwen1.5b_step_gdpo_self_eval_gpu4_v1`
  - GPU: `CUDA_VISIBLE_DEVICES=4`
  - Endpoint: `http://127.0.0.1:8199/v1`
  - Watch step 50/100 val acc and val num_steps. This is a cheap comparison point against FOL Step-GDPO.

## Engineering Changes Already Made / Pending Commit

- `verl/experimental/reward_loop/reward_manager/step.py`
  - No-XML-step hard penalty for FOL/FOL-old XML step rewards when bad-format penalty is enabled.
  - Skips response-level FOL declaration when every step would fail before reaching the judge.
- `verl/experimental/reward_loop/reward_manager/tree.py`
  - Same no-XML-step hard penalty for Tree-GAE FOL/FOL-old external PRMs.
- `scripts/benchmark_judge_latency.py`
  - Lightweight OpenAI-compatible judge latency benchmark for direct endpoint / load-balancer checks.
- `verl/utils/tree_structure.py`
  - Path-level TreeManager penalties for truncated / repeated / multi-boxed / bad-format branch paths are now consumed only when `prm_name == "fol"`.
  - This prevents FOL-specific fail-closed penalties from contaminating `format` or `self_eval` external PRM runs.
- `bash_scripts/fol_tree_gae_localjudge_boost_deeper.sh`
  - Adds a deeper FOL Tree-GAE run script.
  - Uses `(M,N,L,T)=(4,1,3,1)`:
    - `actor_rollout_ref.rollout.n=4`
    - `trainer.tree_top_n=1`
    - `trainer.tree_rounds=3`
    - `trainer.tree_branches=1`
  - Theoretical leaves per prompt: `4 * (1 + 1 * 3 * 1) = 16`, matching step-GDPO `n=16`.

## Experiment Directions

### Prompt-v2 Follow-Up

- `verl/prompts/logical_reasoning.txt` was revised to keep FOL/Z3-friendly atomic reasoning while replacing unsafe boxed examples such as `\boxed{animal}` and `\boxed{No}` with option-letter-safe examples.
- `verl/prompts/logical_reasoning_2.txt` adds a shorter prompt variant closer to the earlier premise-selection style.
- Existing `data/logiqa2k/*.parquet` files embed the old system prompt, so prompt changes require regenerating parquet data before rerunning baselines.
- For ReClor-style datasets, check both sides before training:
  - raw labels are often `0..3`;
  - the prompt should display options as A/B/C/D;
  - `reward_model.ground_truth` and reward extraction must agree on either option letters or raw indices. Prefer option letters for MCQ rewards.

### Baseline Matrix To Run

The useful experiment set is now broader than only FOL. Run it in stages so the full matrix does not waste GPU time before the prompt effect is isolated.

Stage 0: regenerate prompt-v2 data.

- Create a new data directory, e.g. `data/logiqa2k_prompt_v2/`, using `verl/prompts/logical_reasoning.txt`.
- Optionally create `data/logiqa2k_prompt_v2_short/` using `logical_reasoning_2.txt`.
- Do not overwrite the old parquet unless the old prompt is no longer needed for comparison.

Stage 1: new-prompt 1.5B sanity baselines.

- Actor: Qwen2.5-1.5B.
- Methods: step-GDPO first; tree-GAE only after step-GDPO shows nonzero/healthy behavior.
- Rewards to run:
  - outcome-only;
  - format;
  - self-eval;
  - FOL;
  - random, as a sanity/control run.
- Goal: determine whether the prompt alone fixes answer-format collapse or changes the short-reasoning tendency.

Stage 2: stronger-actor baselines.

- Actor families to compare:
  - Qwen2.5-1.5B and Qwen2.5-7B;
  - 3.5/4B and 9B class models if available locally.
- Rewards:
  - outcome-only, format, self-eval, FOL;
  - random only once per actor family as a control, not necessarily every size.
- Primary metrics:
  - val acc;
  - val num_steps;
  - XML bad-format / missing-boxed / multi-boxed rates;
  - FOL invalid translation, leakage, entailed rate, and judge latency.

Stage 3: tree reruns after step baselines.

- Only rerun Tree-GAE when the corresponding step-GDPO run is healthy.
- For each promising actor/reward pair, compare shallow tree vs deeper tree.
- Stop early if the tree collapses to one-step answer paths again without improving acc.
- For future FOL tree attempts, do not repeat the stopped deeper `(4,1,3,1)` setup unchanged.
  - It preserved theoretical 16 leaves but made the search effectively chain-like and still collapsed to direct boxed leaves.
  - Prefer configurations with real branching, e.g. `rollout.n=4, tree_rounds=2, tree_top_n=2, tree_branches=2`, or `rollout.n=4, tree_rounds=1, tree_top_n=2, tree_branches=3`.

- Let FOL step-GDPO v3 finish first.
  - It is the main candidate for "more steps and decent accuracy".
  - Compare best checkpoint by both acc and val num_steps, not acc alone.
- Do not prioritize running FOL Tree-GAE deeper to completion.
  - It is slow and has already shortened by step 100.
  - If kept running, record whether later acc improves despite short paths.
- Run `format tree deeper` as a cheap diagnostic if GPU is available.
  - Goal: test whether deeper Tree-GAE itself can preserve tree depth when the external PRM is cheap and deterministic.
  - If format deeper also shortens, the problem is mostly Tree-GAE/reward structure rather than FOL judge.
- Consider self-eval tree only after the format deeper diagnostic.
  - Self-eval is more expensive and old self-eval tree also shortens.
  - The old self-eval tree script used penalty config that could previously contaminate non-FOL PRMs; after the FOL-only penalty fix, reruns are cleaner.
- Try a stronger actor.
  - 3B/7B is the most likely route if the goal is "multi-step + correct".
  - 1.5B appears to optimize short answer strategies more readily than long verifiable reasoning.

## Reward / Objective Ideas

- Separate two targets explicitly:
  - Outcome accuracy.
  - Process quality: enough useful steps, XML validity, and FOL-verifiable steps.
- Do not reward length blindly.
  - Format step50 had many steps but poor acc, so "more steps" alone is not useful.
- Consider a mild process-shape objective only when correctness is nonzero.
  - Example: only reward extra valid steps for samples with correct final answer, or add a small bonus for 2-5 valid reasoning steps.
  - Avoid forcing very long traces; many LogiQA examples may not need them.
- Consider a tree anti-collapse rule.
  - Penalize branches that immediately emit only boxed answer after one shallow step.
  - Keep this separate from FOL judge failure penalties.
- Prefer outcome gating over a hard length penalty as the first tree anti-collapse change.
  - Example: if `valid_step_count < 2`, set `outcome_reward_weight = 0` or heavily downweight it for that path.
  - This prevents the model from getting full outcome credit for `one weak <step> + boxed`, while avoiding a blanket `-1` for all short but otherwise valid outputs.
- Consider separating tree reasoning expansion from final answer generation.
  - Tree expansion should search over reasoning steps.
  - The final boxed answer can be generated or scored as a separate finalization stage.
  - This would stop Tree-GAE from using forked nodes mainly as direct answer-letter probes.
- Add lightweight nontrivial-step checks only if outcome gating is insufficient.
  - A reasoning step should not simply restate the definition and jump to the answer.
  - The conclusion should not be a bare option choice except in a designated final reasoning/finalization step.
  - Avoid rewarding length alone; the goal is useful multi-step reasoning, not verbose traces.
- Consider changing Tree-GAE weights.
  - Current FOL tree uses `[0.8, 0.2]`.
  - Higher FOL/process weight may preserve process, but only after FOL success/leakage rates are good enough.

## Dataset Coverage

### Logic Datasets: Z3 Logic Mode (`fol_task_type: "logic"`)

逻辑侧使用 entity-predicate schema + FOL prompt，Z3 作为 SAT/SMT solver 天然适配约束满足类推理。

| 数据集 | 题型 | Z3 覆盖率 | 说明 |
|--------|------|-----------|------|
| LogiQA | 逻辑推理选择题 | ~90% | 当前主力训练/评估数据集 |
| FOLIO | FOL 自然语言推理 | ~90% | 天然 FOL 表达，Z3 直接适配 |
| AR-LSAT | 分析推理（逻辑游戏） | ~90-95% | 排序/分组/条件约束 = CSP，Z3 的甜区 |
| ReClor | 阅读理解式逻辑推理 | ~85-90% | 偏自然语言推理，个别题需 pragmatic reasoning |
| SCONE | 状态追踪（实体位置/属性变化） | ~85-90% | 约束满足 + 状态更新，Z3 适配 |

不适合 FOL verification 的数据集：
- **ARC (AI2 Reasoning Challenge)**：小学科学常识推理，需要世界知识和因果推理，Z3 无法编码物理直觉和常识（覆盖率 ~20-30%）。

逻辑侧覆盖率整体较高，不需要 Isabelle 等外部工具补充。

### Math Datasets: Z3 Math Mode (`fol_task_type: "math"`)

已实现 `fol_task_type="math"` 模式，使用纯 Int/Real 算术 schema + 数学专用 prompt。

**各数学数据集 Z3 step-level 覆盖率估计：**

| 数据集 | 难度 | Z3 覆盖率 | 建议 |
|--------|------|-----------|------|
| GSM8K | 小学算术 | ~95% | ✅ 主力对标数据集，FoVer 也报了此数据集 |
| AQUA-RAT | 数学应用题（多选） | ~80-85% | ✅ 类似 GSM8K 但稍难，含 rationale，多选格式 A-E |
| MATH-500 | 高中竞赛混合 | ~50% | ⚠️ 可跑，需标注覆盖率限制；Prealgebra/Algebra 好，Precalculus 废 |
| AMC 10/12 | 高中竞赛 | ~40-50% | ⚠️ 算术/代数/数论可以，组合计数和几何不行 |
| Minerva | STEM 混合 | ~30-40% | ⚠️ 含微积分/物理/化学，Z3 无 sin/cos/积分/微分 |
| AIME 2024 | 竞赛进阶 | ~20-30% | ❌ 大量组合/三角/复数/数列，信号太稀疏 |
| Olympiad Bench | 奥赛级 | ~15-20% | ❌ 需归纳/构造性证明/高等技巧，基本不可用 |

覆盖率随难度递增骤降。GSM8K 是 Z3 的甜区；AIME 及以上没有 Isabelle 不应跑 step-level PRM。

Z3 的根本限制：
- **无归纳证明**：组合数学和数论中需要数学归纳法的证明不可表达
- **无高阶逻辑**：无法表达 "对所有函数 f..."
- **无三角函数/微积分**：Precalculus 类别几乎完全不覆盖
- **非线性算术不完备**：Z3 用 nlsat 启发式，复杂多项式可能 timeout 返回 UNKNOWN → 0.0

覆盖不了的步骤 fail-closed（返回 0.0），不产生错误正面奖励，但降低了 process reward 的有效信号密度。

### TODO: Isabelle/HOL 集成路线

参考 FoVer（2505.15960）的 `sorry` trick 实现 step-level 定理证明验证：

1. **评估可行性**
   - Isabelle 每步验证 ~3s（FoVer 用 40 并行进程在 CPU 上跑）
   - 在线 RL 场景下：假设 8 步/proof × 16 rollouts = ~6 min/batch 纯验证延迟
   - 需要评估是否可以异步验证（rollout 先用 outcome reward，Isabelle reward 延迟回传）

2. **架构方案（优先级从高到低）**
   - **方案 A：离线标注 + PRM 蒸馏**：用 Isabelle 离线标注数学步骤数据 → 训练 PRM → PRM score 作为在线 RL 的 process reward signal（不需要 Isabelle 在线跑）
   - **方案 B：异步在线验证**：Isabelle 验证结果延迟回传到下一个 RL epoch
   - **方案 C：在线同步验证**：直接在 reward loop 中调用 Isabelle（延迟最大，但信号最准确）

3. **Autoformalization 挑战**
   - NL → Isabelle thy 的翻译比 NL → Z3 更难
   - FoVer 用 Qwen 2.5 7B few-shot 做 autoformalization
   - 需要 Isabelle 服务器基础设施（Java/ML 运行时）
   - 验证 GSM8K/MATH 上的 autoformalization 成功率

4. **与当前系统的集成点**
   - `engine.py` 的 `TaskType` 枚举可扩展 `MATH_ISABELLE`
   - 验证语义不同：Z3 用 UNSAT entailment check，Isabelle 用 `sorry` trick step isolation
   - 需要新的 `IsabelleEngine` 类替代 `FOLEngine.verify_step()`

## FOL Judge / Pipeline Ideas

- Immediate FOL speed TODO:
  1. Run a short smoke test after the no-step penalty change and confirm boxed-only outputs produce `bad_format(no_xml_step)` with one `penalty_score` at the final response token and zero FOL judge usage.
  2. Add a temporary one-batch dump option for FOL runs to save all 64 generated responses, step positions, FOL debug, and per-sample reward timing. Current console logs print only one sample per step.
  3. Add declaration/preprocess judge usage and timing metrics. Current `fol_judge/calls` mostly reflects verify/translation calls and undercounts true judge pressure.
  4. Done in working tree: add cross-process declaration/preprocess cache keyed by prompt + FOL config.
     - Successful declarations are cached under `/tmp/verl_fol_shared_preprocess_cache` by default.
     - `FOL_SHARED_PREPROCESS_DISK_CACHE=0` disables it.
     - `FOL_SHARED_PREPROCESS_DISK_CACHE_DIR=/path/to/cache` changes the cache directory.
     - `FOL_SHARED_PREPROCESS_DISK_CACHE_VERSION=...` can be bumped to invalidate old cache entries after prompt/pipeline changes.
     - A 64-record / 4-prompt / 16-process smoke test reduced actual fake preprocess calls to 4.
  5. Done in working tree: add cross-process exact verify reward cache.
     - Exact verify rewards are cached under `/tmp/verl_fol_verify_cache` by default.
     - `FOL_VERIFY_DISK_CACHE=0` disables it.
     - `FOL_VERIFY_DISK_CACHE_DIR=/path/to/cache` changes the cache directory.
     - `FOL_VERIFY_DISK_CACHE_VERSION=...` can be bumped to invalidate old cache entries after prompt/pipeline changes.
     - A 64-record / 4-exact-key / 16-process smoke test reduced actual fake `verify_step` calls to 4, with 60 disk-cache hits.
  6. Recheck whether `+reward.api_config.max_tokens=512` or `768` is safe for FOL judge quality after no-step waste is removed.
- Continue reducing invalid FOL translation.
  - v3 improved over v2/v2.1 substantially.
  - Remaining failure modes include unknown identifiers, sort mismatch, declaration failures, leakage, and Z3 runtime errors.
- Be careful with cumulative verification.
  - It gives stronger prefix checking but makes deeper paths more expensive and more failure-prone.
  - It also increases context length pressure for step-GDPO.
- Keep old whole-code correction off unless deliberately ablated.
  - New expression-level repair is safer than whole-program correction.
- Keep path-level penalties FOL-only.
  - `format` should be `1/0` via `check_step_format_fol`.
  - `self_eval` should use its own judge score, not FOL fail-closed penalties.

## Current Best Reference Points

| Run | Log | Status | Best Val Acc | Best Step | Val Num Steps At Best | Final / Latest Val Acc | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| FOL step-GDPO v3 | `train_fol_step_gdpo_gpu2_v3.log` | running | `0.38249` | 350 | `7.11` | `0.38249` latest known | Best current multi-step FOL run. |
| FOL Tree-GAE shallow v3 | `train_fol_tree_gae_gpu3_judge56_v3.log` | done | `0.43318` | 450 | `1.00` | `0.37942` at step 500 | Highest FOL-tree acc but short-answer strategy. |
| FOL Tree-GAE deeper v1 | `train_fol_tree_gae_gpu4_judge56_deeper_v1.log` | stopped | `0.37174` | 100 | `1.14` | `0.31490` at step 250 | Worse than shallow; collapsed to val num_steps `1.00`. |
| FOL step-GDPO old | `train_fol_step_gdpo_gpu2.log` | done | `0.39017` | 350 | `5.45` | `0.35791` at step 500 | Old aux FOL reward metric scale is buggy; compare acc/steps only. |
| FOL step-GDPO v2 | `train_fol_step_gdpo_gpu2_v2.log` | failed/stopped | `0.34869` | 100 | `3.97` | `0.34869` at step 100 | Hit fatal errors later. |
| FOL step-GDPO v2.1 | `train_fol_step_gdpo_gpu2_v2_1.log` | stopped | `0.34562` | 150 | `2.45` | `0.34562` at step 150 | Shorter than v3. |
| format Tree-GAE old | `train_format_tree_gae_gpu3.log` | done | `0.43318` | 450 | `2.53` | `0.40707` at step 500 | Cheap format reward; longer than FOL tree but still shortens. |
| outcome Tree-GAE old | `train_outcome_tree_gae_gpu3.log` | done | `0.35638` | 500 | `0.00` | `0.35638` at step 500 | Outcome-only tree collapse / no step metric. |
| self-eval step-GDPO old | `train_self_eval_step_gdpo_gpu3.log` | done | `0.41475` | 350 | `1.00` | `0.40553` at step 500 | Strong acc, short answers. |
| self-eval Tree-GAE old | `train_self_eval_tree_gae_gpu5_new.log` | done | `0.36098` | 150 | not recorded in latest table | `0.25960` at step 500 | Poor late behavior; old penalty contamination may affect interpretation. |

Auxiliary reward means from older runs can exceed `[0, 1]` due to an older aggregation/reporting bug. Use val acc, val num_steps, format validity, and FOL debug rates for cross-run comparison.

- Step-GDPO self-eval 8:2:
  - Log: `train_self_eval_step_gdpo_gpu3.log`
  - Best val acc: `0.41475` at step 350.
  - But val num_steps collapsed to `1.0`.
- FOL Tree-GAE shallow v3:
  - Log: `train_fol_tree_gae_gpu3_judge56_v3.log`
  - Best val acc: `0.43318` at step 450.
  - But it is a short-answer strategy.
- FOL step-GDPO v3:
  - Log: `train_fol_step_gdpo_gpu2_v3.log`
  - Best observed val acc so far: `0.38249` at step 350.
  - Keeps val num_steps around `7`.

## Useful Local Commands

Summarize current FOL runs:

```bash
python - <<'PY'
import re
from pathlib import Path
files = [
    ("step", "train_fol_step_gdpo_gpu2_v3.log"),
    ("tree_shallow", "train_fol_tree_gae_gpu3_judge56_v3.log"),
    ("tree_deeper", "train_fol_tree_gae_gpu4_judge56_deeper_v1.log"),
]
keys = [
    "training/global_step",
    "critic/score/mean",
    "step_gdpo/fol_step_reward/mean",
    "fol_judge/invalid_translation_rate/mean",
    "fol_judge/leakage_rate/mean",
    "fol_judge/entailed_rate/mean",
    "tree/avg_leaves_per_tree",
    "tree/pass_rate",
    "num_steps/mean",
    "response_length/mean",
    "timing_s/step",
]
step_re = re.compile(r"step:(\d+) -")
def get(line, key):
    m = re.search(re.escape(key) + r":([-+0-9.eE]+)", line)
    return None if not m else float(m.group(1))
for name, file in files:
    p = Path(file)
    print("\n", name, file)
    if not p.exists():
        print("missing")
        continue
    step_lines = []
    val_lines = []
    for line in p.open(errors="ignore"):
        if step_re.search(line):
            step_lines.append(line)
        if "val-core/logiqa/acc/mean@1" in line:
            val_lines.append(line)
    if step_lines:
        line = step_lines[-1]
        print("latest", step_re.search(line).group(1))
        for key in keys:
            value = get(line, key)
            if value is not None:
                print(key, value)
    for line in val_lines[-5:]:
        print("val", step_re.search(line).group(1) if step_re.search(line) else "?", get(line, "val-core/logiqa/acc/mean@1"), get(line, "val-aux/logiqa/num_steps/mean@1"))
PY
```
