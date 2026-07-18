# verl: Volcano Engine Reinforcement Learning for LLMs (Forked)

This is a customized fork of [verl](https://github.com/volcengine/verl) tailored for logical reasoning tasks, process reward-based reinforcement learning methods (Step-GDPO, parallel generation or tree search), and specialized dataset preprocessing pipelines & prompting for LogiQA datasets.

For the original verl library's detailed documentation and features, please refer to [README-bytedance.md](README-bytedance.md).

---

## LogiQA Dataset Preprocessing & Prompting

The LogiQA dataset preprocessing allows injecting custom reasoning instructions (e.g., `p1 & p2 -> i1` for step-wise logical inferences) via flexible prompt file configurations.

### Version, Format, # of samples, and Output Directory

You can customize the LogiQA dataset loading and preprocessing by configuring a few parameters in `logiqa.py`:

- `--version`: Specifies LogiQA dataset version (`1` for `lucasmccabe/logiqa` or `2` for `baber/logiqa2`). Default is `1`.
- `--num_samples`: The number of training samples to keep. Use `-1` for all samples. Default is `2000`.
- `--local_save_dir`: The directory to save the output `.parquet` files. Default is `./data/logiqa2k`.
- `--format`: Prompt formatting style. Default is `flat`.
  - `flat`: Regular plain text format (`Context: ...\n\nQuestion: ...\n\nOptions: ...`).
  - `xml`: XML tag format (`<Context>...\n</Context>\n<Question>...`).

**Example:**

Version 1, 2000 samples, XML text format (the used version)

```bash
python examples/data_preprocess/logiqa.py \
    --version 1 \
    --num_samples 2000 \
    --local_save_dir ./data/logiqa2k \
    --format xml \
    --system_prompt_file logical_reasoning.txt
```

### Prompt-v2 Parquet Generation

The `verl-fol-2` experiment scripts expect prompt-v2 parquet files by default. Generate them before launching new prompt-v2 baselines:

```bash
bash bash_scripts/prepare_logiqa2k_prompt_v2.sh
```

This writes:

```text
data/logiqa2k_prompt_v2/train.parquet
data/logiqa2k_prompt_v2/validation.parquet
data/logiqa2k_prompt_v2/test.parquet
```

The default prompt file is `verl/prompts/logical_reasoning.txt`. It is the main FOL/Z3-aware prompt with MCQ-safe A/B/C/D examples.

To generate the shorter prompt variant:

```bash
PROMPT_FILE=logical_reasoning_2.txt \
DATA_DIR=./data/logiqa2k_prompt_v2_short \
bash bash_scripts/prepare_logiqa2k_prompt_v2.sh
```

Equivalent explicit command:

```bash
python examples/data_preprocess/logiqa.py \
    --version 1 \
    --num_samples -1 \
    --format xml \
    --local_save_dir ./data/logiqa2k_prompt_v2 \
    --system_prompt_file logical_reasoning.txt
```

Prompt changes do not affect existing parquet files. Regenerate parquet whenever `logical_reasoning.txt` or `logical_reasoning_2.txt` changes.

For ReClor, use the same prompt-v2 format and A/B/C/D answer labels:

```bash
bash bash_scripts/prepare_reclor_prompt_v2.sh
```

This writes labeled splits only:

```text
data/reclor_prompt_v2/train.parquet
data/reclor_prompt_v2/validation.parquet
```

The official ReClor test split is unlabeled, so the script skips it by default.
To save it for generation-only inspection:

```bash
bash bash_scripts/prepare_reclor_prompt_v2.sh --save_unlabeled_test
```

For AR-LSAT, use the same prompt-v2 format and A/B/C/D/E answer labels:

```bash
bash bash_scripts/prepare_ar_lsat_prompt_v2.sh
```

This writes all labeled splits from `olegbask/AR-LSAT`:

```text
data/ar_lsat_prompt_v2/train.parquet
data/ar_lsat_prompt_v2/validation.parquet
data/ar_lsat_prompt_v2/test.parquet
```

The new experiment scripts use:

```bash
trainer.project_name='verl-fol-2'
DATA_NAME=${DATA_NAME:-logiqa2k_prompt_v2}
```

You can still run against another dataset directory by overriding:

```bash
DATA_DIR=./data/logiqa2k_prompt_v2 bash bash_scripts/fol_step_gdpo_localjudge_boost.sh
```

### Local Judge Boost Setup

The `fol_*_localjudge_boost.sh` scripts assume an OpenAI-compatible FOL judge is
already running. The recommended setup is two tensor-parallel Qwen3.6-35B-A3B
judge servers behind the local load balancer:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model /root/run/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --port 4872 \
    --tensor-parallel-size 2 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --max-cudagraph-capture-size 256

CUDA_VISIBLE_DEVICES=5,6 python3 -m vllm.entrypoints.openai.api_server \
    --model /root/run/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --port 4873 \
    --tensor-parallel-size 2 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --max-cudagraph-capture-size 256

python3 scripts/openai_lb.py \
    --host 127.0.0.1 \
    --port 4874 \
    --backend http://127.0.0.1:4872 \
    --backend http://127.0.0.1:4873 \
    --timeout 700
```

If only one large-memory GPU is available, start a single-card judge instead.
This is slower and has a lower concurrency cap, but is enough for probes and
small FOL runs:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model /root/run/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --port 4872 \
    --tensor-parallel-size 1 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --max-cudagraph-capture-size 128
```

If the single-card server OOMs during initialization, first lower
`--max-num-seqs` to `64`, then lower `--gpu-memory-utilization` to `0.85`.

Then point FOL training at the load balancer:

```bash
OPENAI_BASE_URL=http://127.0.0.1:4874/v1 \
OPENAI_API_KEY=EMPTY \
FOL_MODEL=Qwen3.6-35B-A3B \
FOL_OPENAI_MAX_INFLIGHT=512 \
bash bash_scripts/fol_step_gdpo_localjudge_boost.sh
```

For scripts that start their own local judge vLLM, the same server-side defaults
are enabled in the script: prefix caching, `VLLM_MAX_NUM_SEQS=256`, and
`VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=256`. Override those environment variables only
when the judge server has memory pressure or needs a different concurrency cap.

### Experiment Alignment Invariants

Use this section when reproducing runs on another cluster. The items below are
the baseline constants for comparable FOL Step-GDPO runs. Do not silently change
them across clusters.

Core task and reward:

```text
DATA_DIR=data/logiqa2k_prompt_v2
algorithm.adv_estimator=step_gdpo
algorithm.step_reward_type=fol
algorithm.use_xml_steps=true
algorithm.step_reward_weights=[0.8,0.2]
algorithm.penalty_max_steps=12
algorithm.penalty_on_truncated=true
algorithm.penalty_on_multi_boxed=true
algorithm.penalty_on_bad_format=true
algorithm.penalty_score=-1.0
algorithm.fol_max_tries=1
algorithm.fol_timeout=10
algorithm.fol_cumulative_mode=current_only
algorithm.fol_judge_use_outlines=true
algorithm.validate_with_step_reward=false
reward.api_config.api_context_shrink_min_tokens=16
reward.api_config.api_context_shrink_retries=6
```

Training and rollout invariants for the main `n=16` LogiQA 7B run:

```text
actor_rollout_ref.model.path=/root/run/models/Qwen2.5-7B-Instruct
data.train_batch_size=4
data.max_prompt_length=2048
data.max_response_length=1536
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.n=16
actor_rollout_ref.rollout.temperature=0.8
actor_rollout_ref.rollout.top_p=0.95
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.02
actor_rollout_ref.actor.kl_loss_type=low_var_kl
algorithm.use_kl_in_reward=false
data.seed=42
actor_rollout_ref.actor.data_loader_seed=42
critic.data_loader_seed=42
```

Resource knobs may change across clusters, but record them in the run note if
they differ:

```text
CUDA_VISIBLE_DEVICES
trainer.n_gpus_per_node
OPENAI_BASE_URL
REWARD_NUM_WORKERS
STEP_REWARD_MAX_WORKERS
FOL_OPENAI_MAX_INFLIGHT
actor_rollout_ref.rollout.gpu_memory_utilization
actor_rollout_ref.rollout.max_model_len
actor_rollout_ref.rollout.max_num_seqs
actor_rollout_ref.rollout.max_num_batched_tokens
actor_rollout_ref.actor.fsdp_config.optimizer_offload
```

Changing `actor_rollout_ref.rollout.n` changes the experiment. If a memory
fallback uses `n=8`, the experiment name and log name must include `n8`, for
example `qwen7b_step_gdpo_fol_logiqa_gpu67_n8_v1`.

For H200 migration, the clean two-GPU layout is one H200 for the single-card
judge and one H200 for training. Keep the invariants above unchanged unless the
run name explicitly marks the change. For H100, do not assume the current
Qwen3.6-35B-A3B judge is stable as a single-card server; prefer tensor-parallel
judge service or a smaller judge.

## LogiQA Prompt-v2 Baseline Snapshot

Snapshot from local logs on 2026-05-02. Accuracy is `val-core/logiqa/acc/mean@1`;
percentages are reported to three decimal places.
For active runs, the "final/current" column reports the latest available validation
instead of a completed final checkpoint.

| Method | Log | Train GPU | Judge / Eval GPU | Best Val Acc | Final / Current Val Acc | Notes |
|---|---|---:|---:|---:|---:|---|
| GSPO outcome-only | `train_gspo_outcome_only_logiqa_full_prompt1_gpu4_v1.log` | GPU4 | none | `48.848% @1200` | `36.866% @1844` | DAPO/outcome reward with GSPO policy loss, `seq-mean-token-mean`, KL off |
| FOL Step-GDPO v4 | `train_fol_step_gdpo_gpu2_v4_2.log` | GPU2 | GPU0/1 + GPU5/6 via `:4874` LB | `47.926% @550` | `42.704% @1844` | Completed; final checkpoint also evaluated on held-out test |
| DAPO outcome-only | `train_dapo_outcome_only_logiqa_full_prompt1_gpu4_v1.log` | GPU4 | none | `45.315% @750` | `39.017% @1844` | Outcome-only GRPO/DAPO baseline |
| Self-eval Step-GDPO v1 | `train_self_eval_step_gdpo_gpu4_v1.log` | GPU4 | GPU4 local `:8199` | `45.161% @1050/1500` | `38.556% @1844` | Completed |
| Format Step-GDPO | `train_format_step_gdpo_a800_gpu0_v1.log` | A800 GPU0 | none | `44.700% @600/1400` | `36.252% @1844` | A800 baseline |
| Format Tree-GAE | `train_format_tree_gae_a800_gpu1_v1.log` | A800 GPU1 | none | `40.399% @550` | `28.879% @1844` | Short-tree collapse by the end |
| Outcome-only Tree-GAE | `train_outcome_tree_gae_a800_gpu2_v1.log` | A800 GPU2 | none | `39.939% @400` | `31.183% @1844` | Short-tree collapse by the end |
| FOL Tree-GAE v4 | `train_fol_tree_gae_gpu3_v4.log` | GPU3 | GPU0/1 + GPU5/6 via `:4874` LB | `38.710% @1050` | `27.957% @1844` | Negative result for current tree shaping; final `num_steps/mean=2.0` |
| Self-eval Tree-GAE v1 | `train_self_eval_tree_gae_gpu4_v1.log` | GPU4 | GPU4 local `:8199` | `39.017% @1700` | `31.183% @1844` | Tree baseline remained weak |

Held-out LogiQA test results use `data/logiqa2k_prompt_v2/test.parquet` as
`data.val_files` with `trainer.val_only=true`. LogiQA validation and test each
contain 651 examples in this preprocessing.

| Method | Checkpoint | Test Acc | Test Log | Notes |
|---|---:|---:|---|---|
| Qwen2.5-1.5B-Instruct base | none | TBD | TBD | Priority next test-only run on H20 |
| GSPO outcome-only | `global_step_1844` | `43.318%` | `test_gspo_outcome_only_logiqa_final1844_remote.log` | Final checkpoint |
| FOL Step-GDPO v4 | `global_step_1844` | `47.158%` | `test_fol_step_gdpo_logiqa_final1844.log` | Pure answer accuracy; `validate_with_step_reward=false`, so no FOL judge calls |
| DAPO outcome-only | `global_step_1844` | `42.550%` | `test_dapo_outcome_only_logiqa_final1844_remote.log` | Final checkpoint |

## ReClor Prompt-v2 Snapshot

ReClor uses `data/reclor_prompt_v2`; the official test split is unlabeled, so
these are validation results on 500 labeled validation examples.

| Method | Log | Best Val Acc | Final Val Acc | Notes |
|---|---|---:|---:|---|
| Qwen2.5-1.5B-Instruct base | `train_fol_step_gdpo_reclor_gpu2_v4.log` | `48.600% @0` | `48.600% @0` | Initial validation before RL |
| FOL Step-GDPO v4 | `train_fol_step_gdpo_reclor_gpu2_v4.log` | `58.600% @450` | `58.000% @579` | Final train score `52.344%`; final FOL step reward `55.469%` |
| Format Step-GDPO | `train_format_step_gdpo_reclor_gpu2_v1.log` | `59.000% @579` | `59.000% @579` | Strongest completed 1.5B ReClor baseline so far |
| Self-eval Step-GDPO | `train_self_eval_step_gdpo_reclor_gpu2_v1.log` | `58.400% @450` | `57.800% @579` | Completed |
| GSPO outcome-only | `train_gspo_outcome_only_reclor_gpu4_v1.log` | `59.600% @400` | `55.000% @579` | Outcome-only + GSPO loss, KL off |
| DAPO outcome-only | `train_dapo_outcome_only_reclor_gpu3_v1.log` | `59.400% @550` | `58.800% @579` | Outcome-only baseline |
| FOL Step-GDPO KL-off | `train_reclor_fol_kloff_gpu2_v1.log` | `56.400% @200` | `54.800% @400` stopped | Stopped after step `404/579`; weaker than FOL v4 / format / DAPO |
| Qwen2.5-7B FOL Step-GDPO | `train_reclor7b_fol_gpu1_v1.log` | `75.000% @100` | `74.600% @200` crashed | Crashed after step `206/579` when the `017_2` judge forward to `:4874` disappeared; `global_step_200` checkpoint exists |

## GSM8K Snapshot

GSM8K uses `data/gsm8k` with `fol_task_type=math`; validation below is the
GSM8K test set used as `data.val_files`.

| Method | Log | Best Val/Test Acc | Final / Current Val/Test Acc | Notes |
|---|---|---:|---:|---|
| Qwen2.5-1.5B-Instruct base | `test_base_qwen25_15b_gsm8k_0172_gpu2.log` | `40.788% @0` | `40.788% @0` | Greedy/test-only base model evaluation |
| FOL Step-GDPO 1.5B | `train_fol_step_gdpo_gsm8k_gpu4_v1.log` | `75.739% @700/900` | `74.223% @934` | Math-mode FOL reward |
| Format Step-GDPO 1.5B | `train_format_step_gdpo_gsm8k_0172_gpu0_v1.log` | `77.331% @700` | `74.905% @934` | Completed on `017_2` |
| Self-eval Step-GDPO 1.5B | `train_self_eval_step_gdpo_gsm8k_gpu3_v1.log` | `79.682% @934` | `79.682% @934` | Strongest completed GSM8K 1.5B run so far |
| GSPO outcome-only 1.5B | `train_gspo_outcome_only_gsm8k_0172_gpu1_v1.log` | `77.938% @800` | `75.284% @934` | Completed on `017_2` |
| DAPO outcome-only 1.5B | `train_dapo_outcome_only_gsm8k_0172_gpu3_v1.log` | `77.180% @400` | `73.086% @934` | Completed on `017_2` |
| Qwen2.5-Math-1.5B-Instruct FOL | `train_gsm8k_math1.5b_fol_gpu4_v1.log` | stopped | stopped | Produced natural-language math CoT without XML steps, so FOL process reward was skipped |
| Qwen2.5-7B-Instruct FOL | `train_gsm8k7b_fol_gpu34_v1.log` | `91.054% @200` | `90.447% @934` | Completed on GPU3/4; final train score `100.000%`, recent-10 train score `94.531%` |

Current priority after this snapshot:

1. Run LogiQA 7B FOL Step-GDPO on two GPUs; the earlier one-GPU attempt OOMed during actor update.
2. Resume or restart ReClor 7B FOL Step-GDPO only after a stable two-GPU slot and judge route are available.
3. Keep both evaluation protocols: in-domain A->A comparisons and single-checkpoint mixed-train multi-test evaluation for universal-verifier claims.
4. Finish FOLIO/FLDx2 and ProcessBench GSM8K probes to quantify out-of-domain FOL judge behavior.
5. Deprioritize tree search until step-level baselines and 7B scaling are clearer; if revisited, start from reward-shaping rather than deeper tree length alone.

## Related-Work Positioning

| Work | Category | What It Does | How To Compare / Position Ours |
|---|---|---|---|
| Logic-RL (`arXiv:2502.14768`) | Rule-based RLVR on logic | Trains on synthetic logic puzzles with a strict format reward and exact answer verification; the paper reports that a 7B model trained on 5K logic problems generalizes to harder math benchmarks such as AIME/AMC. | Cite as evidence that rule-based verifiers plus strict format constraints can teach reasoning behavior. It is not a PRM baseline; the closest comparison is our format/rule-reward ablation and cross-dataset generalization from logic data. |
| AURORA (`arXiv:2502.11520`) | Automated universal PRM training | Uses LLM-as-a-judge ensemble prompting plus reverse verification to label reasoning processes, then trains a universal PRM evaluated on diverse policies and long-CoT trajectories. | Position our method as replacing learned/judge-generated process labels with executable formal verification online. A fair PRM-style comparison should include mixed-train, multi-test evaluation and ProcessBench-style diagnostics; optional follow-up is distilling FOL labels into a PRM. |
| Process Reward Models That Think / ThinkPRM (`arXiv:2504.16828`) | Generative PRM | Fine-tunes a verifier to generate verification CoT for each reasoning step, using far fewer process labels than discriminative PRMs; evaluates on ProcessBench, MATH-500, AIME, and OOD reasoning/code subsets. | Closest conceptual baseline for "a reward model that reasons before scoring." Our distinction is formal fail-closed checking through Z3 rather than verbalized judging; report ProcessBench/FOLIO/FLDx2 probes to show where executable FOL verification helps or fails. |

Protocol note: report both in-domain A->A results and single-checkpoint mixed-train multi-test results. The latter is the right comparison point for universal PRM / LLM-as-judge verifier work.

Version 1, 2000 samples, plain text format:

```bash
python examples/data_preprocess/logiqa.py \
    --version 1 \
    --num_samples 2000 \
    --local_save_dir ./data/logiqa2k \
    --system_prompt_file logical_reasoning.txt
```

Version 1, all samples, plain text format:

```bash
python examples/data_preprocess/logiqa.py \
    --version 1 \
    --num_samples -1 \
    --local_save_dir ./data/logiqa \
    --system_prompt_file logical_reasoning.txt
```

Version 2, 5000 samples, XML format:

```bash
python examples/data_preprocess/logiqa.py \
    --version 2 \
    --num_samples 5000 \
    --format xml \
    --local_save_dir ./data/logiqa5k_v2_xml \
    --system_prompt_file logical_reasoning.txt
```

Injection of Logic Reasoning Prompt:

```bash
python examples/data_preprocess/logiqa.py \
    --version 2 \
    --num_samples 5000 \
    --format xml \
    --local_save_dir ./data/logiqa5k_v2_xml \
    --system_prompt_file logical_reasoning.txt
```

### Prompting

```bash
# 只加 system prompt（读取 verl/prompts/logical_reasoning.txt）
python examples/data_preprocess/logiqa.py \
    --system_prompt_file logical_reasoning.txt

# 只加 user prompt（在题目后追加）
python examples/data_preprocess/logiqa.py \
    --user_prompt_file logical_reasoning.txt

# 两个都加（system + user 各用不同的 txt）
python examples/data_preprocess/logiqa.py \
    --system_prompt_file my_system.txt \
    --user_prompt_file my_user_instructions.txt

# 传绝对路径也支持
python examples/data_preprocess/logiqa.py \
    --system_prompt_file /path/to/any_prompt.txt
```

## Training Parameters

### Process Reward Type

Process reward (step-level reward) 用于对推理链中的**每一步**独立评分，而非只看最终答案。通过 `algorithm.step_reward_type` 配置。

| 参数 | 说明 |
|------|------|
| `algorithm.step_reward_type` | 步级奖励类型，支持以下值 |
| `algorithm.step_reward_weights` | `[outcome_weight, process_weight]`，控制结果奖励与过程奖励的混合比例，默认 `[1.0, 1.0]`。fol/self_eval 推荐 `[0.8, 0.2]`，format 可用 `[0.5, 0.5]` |
| `algorithm.use_xml_steps` | 是否使用 XML 标签（`<step>...</step>`）解析步骤边界，默认 `False` |

**可选 reward type：**

| Type | 计算方式 | 返回值 | 外部依赖 |
|------|----------|--------|----------|
| `format` | 正则匹配 XML 格式 | 二值 0.0 / 1.0 | 无 |
| `fol` | Z3 求解器验证一阶逻辑可满足性 | 连续 [0, 1] | OpenAI API |
| `self_eval` | LLM 按 rubric 评分（0-10 → 归一化到 [0,1]） | 连续 [0, 1] | OpenAI 兼容 API |
| `random` | 随机分数（调试用） | 连续 [0, 1] | 无 |

**FOL / Self-Eval API 环境变量：**

```bash
export OPENAI_API_KEY="sk-YOUR-KEY-HERE"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export FOL_MODEL="gpt-4o-mini-2024-07-18"       # fol 模式
export SELF_EVAL_MODEL="Qwen2.5-1.5B-Instruct"  # self_eval 模式
```

### Step-GDPO

Step-GDPO（`algorithm.adv_estimator=step_gdpo`）在 GRPO 基础上引入步级奖励，将 outcome reward 和 process reward 以 "big-pool" 归一化方式组合。

**核心配置：**

```bash
algorithm.adv_estimator=step_gdpo
reward_model.reward_manager=step
+algorithm.step_reward_type=fol          # 或 format / self_eval
+algorithm.step_reward_weights='[0.8, 0.2]'  # [outcome, process]  (format 可用 [0.5, 0.5])
algorithm.use_xml_steps=true
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.adv_estimator` | — | 设为 `step_gdpo` |
| `reward_model.reward_manager` | — | 设为 `step` |
| `algorithm.step_reward_type` | — | 步级奖励类型（`fol` / `format` / `self_eval`） |
| `algorithm.step_reward_weights` | `[1.0, 1.0]` | `[outcome_weight, process_weight]`。fol/self_eval 推荐 `[0.8, 0.2]` |
| `algorithm.use_xml_steps` | `False` | 使用 XML 标签解析步骤边界 |

**Advantage 计算流程：**

1. **Outcome Advantage**：标准 GRPO 组内归一化（标量奖励 → token 级 advantage）
2. **Process Advantage**：将同组所有 rollout 的步级分数汇入 "big pool"，统一 mean/std 归一化后放回各步结束位置
3. **加权求和**：`A[i,t] = w_outcome × A_outcome + w_process × A_process`
4. **Reward-to-Go**：从右向左累积求和
5. **Batch Whitening**：最终 batch 级白化

### Tree Search (TreeRL)

Tree-GAE（`algorithm.adv_estimator=tree_gae`）基于 EPTree（arXiv:2506.11902）实现树搜索 RL 训练。在推理过程中对高不确定性节点进行分叉搜索，通过树结构探索更多推理路径。

**核心配置：**

```bash
algorithm.adv_estimator=tree_gae
reward_model.reward_manager=tree
+trainer.tree_sampling=True
+trainer.tree_rounds=1
+trainer.tree_top_n=2
+trainer.tree_branches=2
+trainer.tree_mask_tail_ratio=0.1
```

**树搜索参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainer.tree_sampling` | `False` | 开启树搜索模式 |
| `trainer.tree_rounds` | `1` | 树搜索轮数 L |
| `trainer.tree_top_n` | `2` | 每轮选 top-N 高不确定性节点扩展 |
| `trainer.tree_branches` | `2` | 每节点分叉数 T |
| `trainer.tree_mask_tail_ratio` | `0.1` | 尾部 token 遮蔽比例，防止退化扩展 |

EPTree 参数组合示例 **(M=6, N=2, L=1, T=2)**：初始采样 6 条（`rollout.n=6`），每轮选 2 节点各分 2 叉，最终约 **30 条叶子路径**。

**Advantage Pipeline 参数：**

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `trainer.tree_step_reward_mode` | `la` | `ga_la` / `ga` / `value_only` | 步级奖励计算方式（la = V(sn) - V(parent)） |
| `trainer.tree_overall_norm_style` | `token` | `step` / `none` | 步级奖励归一化粒度 |
| `trainer.tree_use_weighted_value` | `False` | `True` | 是否使用加权 value 计算叶子得分 |
| `trainer.tree_weighted_value_style` | `sqrt` | `uniform` / `original` | 加权方式（仅 `use_weighted_value=True` 时生效） |
| `algorithm.tree_ext_reward_dedup` | `True` | `False` | 共享前缀节点的外部 PRM 分数去重 |

**可选外部 PRM：**

Tree-GAE 可叠加外部 process reward（`format` / `fol` / `self_eval`），此时 `step_reward_weights` 语义变为 `[tree_weight, ext_prm_weight]`：

```bash
+algorithm.step_reward_type=fol             # 或 format / self_eval
+algorithm.step_reward_weights='[0.8, 0.2]' # format 可用 [0.5, 0.5]
+algorithm.tree_ext_reward_dedup=True
```

若不配置外部 PRM（如 `outcome_tree_gae.sh`），则退化为纯树结构 advantage。

## Training Scripts

### DAPO

DAPO is the original baseline method explored prior to Step-GDPO. It mitigates mode collapse via an overlong-buffer mechanism.

#### Sanity Check

```bash
bash bash_scripts/sanity_check_dapo.sh
```

#### One Epoch Training

```bash
bash bash_scripts/one_epoch_dapo.sh
```

### Step-GDPO + Parallel Sampling

Step-GDPO is the core algorithm currently under development, leveraging First-Order Logic (FOL) API evaluations as step-wise rewards during training.

#### Sanity Check with Random Reward

Useful for validating the local training loop with a dummy random reward provider:

```bash
bash bash_scripts/sanity_check_step_gdpo.sh
```

#### One Epoch Training with FOL Reward

Set up the OpenAI-compatible API details for remote FOL step evaluation:

```bash
export OPENAI_API_KEY="sk-YOUR-KEY-HERE"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export FOL_MODEL="gpt-4o-mini-2024-07-18"

bash bash_scripts/fol_step_gdpo.sh
```

### Step-GDPO + TreeRL (Entropy-guided Branching Tree Search) Sampling

TODO: Tree search configurations and documentation to be added.

## Slurm Integration

The repository is built to work flexibly with Slurm workloads. You can use `srun` to submit your jobs. Here is an example of running the GDPO sanity check on a single A800 GPU:

```bash
srun -p gpu_a800 -G1 bash -c "export PYTHONUNBUFFERED=1; bash bash_scripts/sanity_check_step_gdpo.sh" 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
```

# Baseline 训练脚本 Walkthrough

> 所有脚本位于 `bash_scripts/`，统一使用 **Qwen2.5-1.5B-Instruct** 模型、**logiqa2k** 数据集、1 GPU 单节点、1 epoch 训练。

---

## 1. 脚本总览

| 类别 | 脚本 | adv_estimator | reward_manager | step_reward_type | weights | rollout.n | 外部依赖 |
|------|------|---------------|----------------|------------------|---------|-----------|----------|
| **DAPO** | `one_epoch_dapo.sh` | grpo | dapo | 无 (纯 outcome) | — | 16 | 无 |
| | `sanity_check_dapo.sh` | grpo | dapo | 无 (纯 outcome) | — | 16 | 无 |
| **Step-GDPO** | `fol_step_gdpo_{1gpu,local,remote}.sh` | step_gdpo | step | fol | [0.8, 0.2] | 16 | OpenAI API |
| | `fol_step_gdpo_localjudge_boost.sh` | step_gdpo | step | fol | [0.8, 0.2] | 16 | 本地 vLLM |
| | `fol_slm_step_gdpo_{1gpu,local,remote}.sh` | step_gdpo | step | fol | [0.8, 0.2] | 16 | 本地 vLLM (GPU 1in2) |
| | `format_step_gdpo.sh` | step_gdpo | step | format | [0.5, 0.5] | 16 | 无 |
| | `self_eval_step_gdpo_{1gpu,local,remote}.sh` | step_gdpo | step | self_eval | [0.8, 0.2] | 16 | OpenAI API / 本地 vLLM |
| | `sanity_check_step_gdpo.sh` | step_gdpo | step | format | [0.5, 0.5] | 16 | 无 |
| | `sanity_check_fol_step_gdpo.sh` | step_gdpo | step | fol | [0.8, 0.2] | 16 | OpenAI API |
| | `sanity_check_self_eval_step_gdpo.sh` | step_gdpo | step | self_eval | [0.8, 0.2] | 16 | OpenAI API / 本地 vLLM |
| **Tree-GAE** | `fol_tree_gae_{1gpu,local,remote}.sh` | tree_gae | tree | fol | [0.8, 0.2] | 6 (30) | OpenAI API |
| | `fol_tree_gae_localjudge_boost.sh` | tree_gae | tree | fol | [0.8, 0.2] | 6 (30) | 本地 vLLM |
| | `fol_slm_tree_gae_{1gpu,local,remote}.sh` | tree_gae | tree | fol | [0.8, 0.2] | 6 (30) | 本地 vLLM (GPU 1in2) |
| | `format_tree_gae.sh` | tree_gae | tree | format | [0.5, 0.5] | 6 (30) | 无 |
| | `outcome_tree_gae.sh` | tree_gae | tree | 无 (纯 outcome) | — | 6 (30) | 无 |
| | `self_eval_tree_gae_{1gpu,local,remote}.sh` | tree_gae | tree | self_eval | [0.8, 0.2] | 6 (30) | OpenAI API / 本地 vLLM |
| | `sanity_check_tree_gae.sh` | tree_gae | tree | format | [0.5, 0.5] | 6 (30) | 无 |

> **weights 列说明**：`[outcome, process]`。fol/self_eval 等 LLM-based reward 使用 `[0.8, 0.2]` 以防止 process reward hacking（详见下文）；format 等低 exploit 风险的 reward 保持 `[0.5, 0.5]`。

---

## 2. 训练算法

### DAPO（对照组）

DAPO 是最基础的 GRPO baseline，用于对照实验。

**核心配置**

```bash
algorithm.adv_estimator=grpo
reward_model.reward_manager=dapo
```

**特有参数：overlong_buffer**

DAPO 必须开启超长惩罚，否则模型会出现模式崩溃（重复生成 token）：

```bash
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
+reward_model.reward_kwargs.overlong_buffer_cfg.len=512        # 缓冲区长度
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
+reward_model.reward_kwargs.max_resp_len=2048
```

惩罚逻辑：如果响应长度超过 `max_resp_len - overlong_buffer_len`（即 2048 - 512 = 1536 token），则按超出比例扣分。

| 脚本 | 用途 | 训练步数 | WandB |
|------|------|----------|-------|
| `one_epoch_dapo.sh` | 完整 1 epoch 训练 | 全量 | 开启 |
| `sanity_check_dapo.sh` | 快速验证 | 5 步 | 关闭 |

---

### Step-GDPO

Step-GDPO 在 GRPO 基础上引入**步级奖励**，将每个推理步骤独立评分，而非只看最终答案。

**与 DAPO 的关键差异**

```diff
- algorithm.adv_estimator=grpo
+ algorithm.adv_estimator=step_gdpo

- reward_model.reward_manager=dapo
+ reward_model.reward_manager=step

+ algorithm.step_reward_type=format|fol|self_eval  # 步级奖励类型（见第 3 节）
+ algorithm.step_reward_weights=[0.8, 0.2]          # [outcome_weight, process_weight]
+ algorithm.use_xml_steps=true                       # 用 XML 标签解析步骤边界

- overlong_buffer_cfg (DAPO 特有，Step-GDPO 不需要)
```

`step_reward_weights=[0.8, 0.2]`：第一个权重对应结果正确性（outcome），第二个对应步级过程质量（process reward）。fol/self_eval 等 LLM-based reward 推荐 `[0.8, 0.2]`（见下文 Anti-Reward-Hacking 说明），format 等低风险 reward 可用 `[0.5, 0.5]`。

| 脚本 | step_reward_type | 说明 |
|------|------------------|------|
| `format_step_gdpo.sh` | format | 纯格式奖励，无外部依赖 |
| `fol_step_gdpo.sh` | fol | FOL 一阶逻辑奖励，需要 OpenAI API |
| `self_eval_step_gdpo_remote.sh` | self_eval | LLM 评分，远程 API |
| `self_eval_step_gdpo_local.sh` | self_eval | LLM 评分，本地 vLLM (GPU 1/2) |
| `sanity_check_step_gdpo.sh` | 可选 | 快速验证（5 步） |

---

### Tree-GAE（TreeRL 树搜索）

Tree-GAE 基于 EPTree（arXiv:2506.11902）实现树搜索 RL 训练。与 Step-GDPO 的"线性推理链"不同，Tree-GAE 在推理过程中进行分叉搜索，通过树结构探索更多推理路径。

**与 Step-GDPO 的关键差异**

```diff
- algorithm.adv_estimator=step_gdpo
+ algorithm.adv_estimator=tree_gae

- reward_model.reward_manager=step
+ reward_model.reward_manager=tree

- rollout.n=16
+ rollout.n=6    # 树会分叉扩展，实际评估路径数远大于 6

+ trainer.tree_sampling=True
+ trainer.tree_rounds=1          # 树搜索轮数 L
+ trainer.tree_top_n=2           # 每轮选 top-N 节点扩展
+ trainer.tree_branches=2        # 每节点分叉数 T
+ trainer.tree_mask_tail_ratio=0.1
```

**EPTree 参数**（当前配置 M=6, N=2, L=1, T=2）：

- **M=6**：初始采样 6 条响应（`rollout.n=6`）
- **N=2**：每轮选 top-2 节点 (`tree_top_n=2`)
- **L=1**：1 轮树搜索 (`tree_rounds=1`)
- **T=2**：每节点 2 个分支 (`tree_branches=2`)
- 最终产生约 **30 条叶子路径** 用于 advantage 计算

**Advantage Pipeline 参数**

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `tree_step_reward_mode` | la | ga_la / ga / value_only | 步级奖励模式（la = local advantage） |
| `tree_overall_norm_style` | token | step / none | 归一化粒度 |
| `tree_use_weighted_value` | False | True | 是否使用加权 value |
| `tree_weighted_value_style` | sqrt | uniform / original | 加权方式（仅 use_weighted_value=True 时生效） |
| `tree_ext_reward_dedup` | True | False | 去重共享前缀的 ext PRM 分数 |

在 Tree-GAE 中，`step_reward_weights` 的语义变为：第一个权重对应树结构内生 advantage（GA+LA），第二个对应外部 PRM 奖励。fol/self_eval 推荐 `[0.8, 0.2]`，format 可用 `[0.5, 0.5]`。

| 脚本 | step_reward_type | 说明 |
|------|------------------|------|
| `outcome_tree_gae.sh` | — (纯 outcome) | 退化为 (GA+LA)/sqrt(n) 作为唯一 advantage |
| `format_tree_gae.sh` | format | 树搜索 + format 外部 PRM |
| `self_eval_tree_gae_remote.sh` | self_eval | 树搜索 + LLM 评分，远程 API |
| `self_eval_tree_gae_local.sh` | self_eval | 树搜索 + LLM 评分，本地 vLLM (GPU 1/2) |
| `sanity_check_tree_gae.sh` | 可选 | 快速验证（5 步） |

---

## 3. Process Reward 模式

三种步级奖励类型可与 Step-GDPO 或 Tree-GAE 组合使用，通过 `+algorithm.step_reward_type=<type>` 指定。

| 维度 | format | fol | self_eval |
|------|--------|-----|-----------|
| **计算方式** | 正则匹配 XML 格式 | Z3 求解器验证逻辑可满足性 | LLM 按 rubric 评分 |
| **返回值** | 二值 0.0 / 1.0 | 连续 [0, 1] | 连续 [0, 1]（10分制/10） |
| **外部依赖** | 无 | OpenAI API + Z3 | OpenAI 兼容 API |
| **延迟** | 极低（纯文本匹配） | 中（API 调用） | 中（API 调用） |
| **终止步检测** | 不区分 | 不区分 | 区分（\boxed{} 启发式，结论加权） |
| **步骤历史** | 只看当前步 | 只看问题上下文 | 传入完整累积推理历史 |
| **适用场景** | 验证输出格式规范 | 逻辑推理题 (LogiQA) | 通用推理任务 |
| **可用训练算法** | Step-GDPO / Tree-GAE | Step-GDPO | Step-GDPO / Tree-GAE |

---

### format

正则匹配每步 XML 标签格式，无外部依赖，二值返回。脚本：`format_step_gdpo.sh`、`format_tree_gae.sh`。

---

### fol

当前仓库的 FOL rewarding pipeline 如下（整合自 T0nglinziyong 的方案）：

```mermaid
graph TD
    A["输入: context + question + options"] --> B["1. Z3 Declaration Generation<br/>z3_declaration_generation.txt → LLM 生成 Z3 声明代码"]
    A --> C["2. parse_reasoning_step<br/>解析 rollout 中的 &lt;premise&gt;/&lt;conclusion&gt; 标签"]
    B --> D["3. Z3 Implication Conversion<br/>z3_implication_conversion.txt → LLM 将步骤翻译为 Z3 蕴含代码"]
    C --> D
    D --> E["4. parse_python_logic_steps<br/>AST 解析 premises_N / conclusion_N"]
    E --> F["5. verify_step_fol<br/>Z3 Solver: And(premises) + Not(conclusion)<br/>UNSAT → 1.0（蕴含成立）"]
    F --> G["返回 reward"]
```

- TODO: 是否加回来 DeBERTa NLI 双轨验证 — 原版有 `verify_steps_nli`（DeBERTa）和 `verify_steps_fol`（Z3）两条轨道做对比，整合版只保留 Z3 单轨

环境变量：

```bash
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-YOUR-KEY-HERE"}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"https://api.openai.com/v1"}
export FOL_MODEL=${FOL_MODEL:-"gpt-4o-mini-2024-07-18"}
```

LLM 调用次数：每道题 1 次（declarations），每步 1 次（implication conversion）。

#### `api_config` 参数

通过 reward manager 的 `api_config` dict 传入，控制 FOL pipeline 的行为：

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `fol_task_type` | `"logic"` | `"logic"` / `"math"` / `"math_z3"` | 任务类型。`logic` = Z3 entity-predicate schema（LogiQA、FOLIO、AR-LSAT）；**`math` = Isabelle/HOL 验证**（自 2026-06-14，见 [`verl/utils/isabelle_utils/README.md`](verl/utils/isabelle_utils/README.md)）；`math_z3` = 旧 Z3 纯 Int/Real 算术 schema（deprecated，原 `math` 语义） |
| `fol_preprocess` | `"direct"` | `"direct"` / `"structured"` | 预处理管线。`direct` = 1 次 LLM 调用生成 Z3 声明；`structured` = rephrase + object/predicate 提取的多步管线 |
| `fol_translation` | `"implication"` | `"implication"` / `"assertion"` | 翻译模式。`implication` = 源分离的前提/结论翻译（推荐）；`assertion` = premise_fol/conclusion_fol 直接翻译 |
| `fol_cumulative_mode` | `"current_only"` | `"current_only"` / `"step"` / `"dependency_graph"` | 累积推理模式。`current_only` = 只用当前步骤；`step` = 包含所有前序结论；`dependency_graph` = 按前提-结论依赖图选择祖先步骤 |
| `fol_judge_use_outlines` | `false` | `true` / `false` | 是否请求结构化 JSON 输出（structured generation）。需要 judge 模型支持 json_schema response_format |
| `max_tries` | `1` | int ≥ 0 | 声明/表达式修复的最大重试次数 |
| `old_max_tries` | `0` | int ≥ 0 | 整段 Z3 代码纠错循环的最大重试次数（旧版纠错路径） |
| `timeout` | `30.0` | float (秒) | Z3 求解器单步超时时间 |
| `model` | — | string | judge LLM 的模型名称 |
| `base_url` | — | string | judge LLM 的 OpenAI 兼容 API 地址 |
| `temperature` | — | float | judge LLM 采样温度 |
| `max_tokens` | — | int | judge LLM 最大输出 token 数 |
| `top_p` | — | float | judge LLM top-p 采样 |
| `fol_shared_state_disk_cache` | `true` | bool | 跨进程磁盘缓存声明预处理结果 |
| `fol_shared_state_cache_dir` | `"/tmp/verl_fol_shared_preprocess_cache"` | path | 声明缓存目录 |
| `fol_verify_disk_cache` | `true` | bool | 跨进程磁盘缓存验证结果 |
| `fol_verify_cache_dir` | `"/tmp/verl_fol_verify_cache"` | path | 验证缓存目录 |

#### `fol_task_type: "math_z3"` 模式（旧 Z3 数学路径，deprecated）

> ⚠️ 自 2026-06-14，`fol_task_type=math` 已改为路由到 **Isabelle/HOL 验证**（正本文档 [`verl/utils/isabelle_utils/README.md`](verl/utils/isabelle_utils/README.md)）。本节描述的纯 Z3 Int/Real 路径，开关现为 `fol_task_type=math_z3`。

为数学题设计的算术/代数验证路径（Z3）。与默认 `logic` 模式的区别：

| | `logic`（默认） | `math` |
|--|---|---|
| **Schema** | Entity sorts + predicate functions（`DeclareSort`, `EnumSort`, `Function → BoolSort()`） | 纯 `Int`/`Real` 算术变量（`Const → IntSort()/RealSort()`） |
| **Declaration prompt** | `z3_declaration_generation.txt` | `z3_declaration_generation_math.txt` |
| **Translation prompt** | `z3_implication_conversion.txt` | `z3_implication_conversion_math.txt` |
| **验证语义** | 相同：`And(premises) ∧ Not(conclusion) → UNSAT = 1.0` | 相同 |
| **适用数据集** | LogiQA, FOLIO, AR-LSAT, ReClor, SCONE | GSM8K, AQUA-RAT, MATH-500, AMC 10/12, Minerva, AIME 2024, Olympiad Bench |

> **不适合 FOL verification 的数据集**：ARC (AI2 Reasoning Challenge) — 需要世界知识和因果常识推理，Z3 无法编码（覆盖率 ~20-30%）。

**MATH-500 子类别覆盖情况：**

| 子类别 | Z3 覆盖 | 说明 |
|--------|---------|------|
| Prealgebra | ✅ 完全 | 基础算术、分数、百分比 |
| Algebra | ✅ 大部分 | 方程、多项式、不等式（Z3 nlsat solver） |
| Number Theory | ⚠️ 部分 | 整除 `%`、模运算、GCD/LCM 可以；归纳证明不行 |
| Counting & Probability | ⚠️ 有限 | 小规模计数可展开；无原生阶乘/组合数，归纳和组合恒等式不可表达 |
| Geometry | ⚠️ 部分 | 坐标几何（距离、斜率、面积）可以；综合几何证明和三角函数不行 |
| Intermediate Algebra | ⚠️ 部分 | 多项式运算可以；复杂非线性可能 timeout |
| Precalculus | ❌ 极有限 | Z3 无原生三角函数、复数、矩阵支持 |

**数学数据集 Z3 step-level 覆盖率总览：**

| 数据集 | 难度 | Z3 覆盖率 | 说明 |
|--------|------|-----------|------|
| GSM8K | 小学算术 | ~95% | 主力对标数据集，FoVer 也报了此数据集 |
| AQUA-RAT | 数学应用题（多选） | ~80-85% | 类似 GSM8K 但稍难，含 rationale，多选格式 A-E |
| MATH-500 | 高中竞赛混合 | ~50% | 可跑，需标注覆盖率限制 |
| AMC 10/12 | 高中竞赛 | ~40-50% | 算术/代数/数论可以，组合计数和几何不行 |
| Minerva | STEM 混合 | ~30-40% | 含微积分/物理/化学，Z3 无 sin/cos/积分/微分 |
| AIME 2024 | 竞赛进阶 | ~20-30% | 大量组合/三角/复数/数列，信号太稀疏 |
| Olympiad Bench | 奥赛级 | ~15-20% | 需归纳/构造性证明/高等技巧，基本不可用 |

Z3 覆盖不了的步骤会自然 fail-closed（返回 0.0），不会产生错误的正面奖励。覆盖率随难度递增骤降。**数学题的 step-level PRM 现在走 Isabelle 路径（`fol_task_type=math`），覆盖率高于 Z3（GSM8K 91%、MATH-500 ~53%）；见正本 [`verl/utils/isabelle_utils/README.md`](verl/utils/isabelle_utils/README.md)。** 本节保留的是 deprecated 的 `math_z3` Z3 路径说明。

使用示例（训练配置中）：

```yaml
algorithm:
  step_reward_type: fol
  step_reward_api_config:
    fol_task_type: "math_z3"    # 旧 Z3 数学路径（deprecated）；Isabelle 路径用 "math"，见 isabelle_utils/README.md
    fol_preprocess: "direct"
    fol_translation: "implication"
    model: "Qwen3.6-35B-A3B"
    base_url: "http://localhost:4869/v1"
```

核心文件：

- `verl/utils/reward_score/formal_verify.py` — reward function 入口（2026-07-03 由 `fol.py` 更名）
- `verl/utils/fol_utils/engine.py` — 统一 FOL 验证引擎
- `verl/prompts/z3_declaration_generation.txt` — Z3 声明生成 prompt（logic）
- `verl/prompts/z3_implication_conversion.txt` — Z3 蕴含转换 prompt（logic）
- `verl/prompts/z3_declaration_generation_math.txt` — Z3 声明生成 prompt（math）
- `verl/prompts/z3_implication_conversion_math.txt` — Z3 蕴含转换 prompt（math）

------

### fol_slm

当前仓库的 FOL SLM rewarding pipeline 如下（整合自 ZhenbinChan 的方案，SLM = Small Language Model）：

```mermaid
graph TD
    A["输入: context + question + options"] --> B["1. Rephrase<br/>rephrase.txt → 改写问题使其更清晰"]
    A --> C["2. Object Extract<br/>object_extract.txt → 提取实体 JSON"]
    A --> D["3. Predicate Extract<br/>predicate_extraction.txt → 提取谓词关系 JSON"]
    C --> E["4. generate_z3_declarations<br/>确定性代码生成 Z3 类型/常量/变量声明"]
    D --> F["5. generate_z3_functions<br/>确定性代码生成 Z3 Function 声明"]
    E --> G["6. translate_step_to_z3<br/>translate_step.txt → LLM 逐步翻译为 Z3 代码"]
    F --> G
    B --> G
    G --> H{"7. run_code<br/>执行 Z3 代码"}
    H -->|Error| I["8. correct_z3_code<br/>correct_code.txt → LLM 自动修复<br/>temperature 每次 +0.1，最多 8 次"]
    I --> H
    H -->|Success| J["返回 reward<br/>SAT → 1.0 / UNSAT → 0.0"]
```

环境变量：

```bash
export FOL_SLM_MODEL=${FOL_SLM_MODEL:-"qwen2.5-3b"}
export FOL_SLM_BASE_URL=${FOL_SLM_BASE_URL:-"http://localhost:4869/v1"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"EMPTY"}
```

与 `fol` 的关键差异：

| 维度       | fol                       | fol_slm                                       |
| ---------- | ------------------------- | --------------------------------------------- |
| LLM        | 外部大模型（GPT-4o-mini） | 本地小模型（qwen2.5-3b via vLLM）             |
| 声明生成   | LLM 直接生成 Z3 代码      | 结构化提取实体/谓词 → **确定性**代码生成      |
| 错误处理   | 单次执行                  | 自动修复循环（最多 8 次重试）                 |
| 验证语义   | 蕴含检查（UNSAT = 成立）  | 可满足性检查（SAT = 一致）                    |
| LLM 调用数 | 每题 1 + 每步 1           | 每题 3（rephrase/object/predicate）+ 每步 1~9 |

核心文件（历史实现已并入统一引擎：`fol_slm.py` / `nl2fol_slm.py` 已删除，
入口统一为 `verl/utils/reward_score/formal_verify.py`，structured pipeline
helpers 在 `verl/utils/fol_utils/common.py`，通过 `fol_preprocess="structured"` 启用）：

- `verl/prompts/rephrase.txt`、`object_extract.txt`、`predicate_extraction.txt`、`translate_step.txt`、`correct_code.txt` — prompt 模板



---

### fol_old

（旧版本fol，不再使用）

调用 OpenAI API 使用 Z3 求解器验证一阶逻辑可满足性。环境变量：

```bash
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-YOUR-KEY-HERE"}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"https://api.openai.com/v1"}
export FOL_MODEL=${FOL_MODEL:-"gpt-4o-mini-2024-07-18"}
```

脚本：`fol_step_gdpo.sh`。

流程：

```mermaid
graph LR
    A["输入:<br/>context + question + options"] --> B["1. Premise Extraction<br/>premise_extraction.txt"]
    B --> C["2. Declaration Extraction<br/>extract_declaration.txt"]
    C --> D["3. Translate to FOL<br/>translate2fol.txt"]
    D --> E["CodeTranslator<br/>FOL→Z3 Python"]
    E --> F["subprocess 执行<br/>SAT/UNSAT→reward"]
```



---

### self_eval

使用 LLM（通常是参考模型本身）对每个推理步骤进行 0-10 评分，归一化到 [0, 1]。

**核心实现**（`verl/utils/reward_score/self_eval.py`）

```
compute_step_reward_self_eval(step_text, prompt_text, step_history, ...)
    -> 判断是否为终止步（包含 \boxed{}）
    -> 选择对应的 system prompt (terminal / non_terminal)
    -> 将累积推理历史拼接为 user prompt
    -> 调用 LLM API 评分
    -> 正则提取 "Overall Score: <float>"
    -> 返回 score / 10.0，范围 [0, 1]
```

**评分 Rubric**

非终止步（`verl/prompts/self_eval/non_terminal.txt`）：

| 维度 | 分值 | 说明 |
|------|------|------|
| Premise Establishment | 0-2 | 前提信息和假设的清晰度 |
| Step Validity | 0-2 | 每步逻辑是否有效、格式良好 |
| Justification Quality | 0-2 | 是否引用了规则/公理/推理依据 |
| Logical Progression | 0-2 | 步骤间过渡是否流畅，无跳跃 |
| Conclusion | 0-2 | 当前步结论是否从前提中正确推出 |

终止步（`verl/prompts/self_eval/terminal.txt`）：结论维度加权到 4 分（占 40%），其余维度降权：

| 维度 | 分值 |
|------|------|
| Premise Establishment | 0-1 |
| Step Validity | 0-2 |
| Justification Quality | 0-1 |
| Logical Progression | 0-2 |
| **Conclusion** | **0-4** |

**部署模式**

Mode A（远程 API，1 GPU）：训练与评分共用同一 GPU，评分请求发往远程 API：

```bash
export OPENAI_BASE_URL="https://your-remote-server/v1"
export OPENAI_API_KEY="your-key"
export SELF_EVAL_MODEL="Qwen2.5-1.5B-Instruct"   # 可选
bash self_eval_step_gdpo_remote.sh
```

Mode B（本地 vLLM，2 GPU）：GPU 0 跑训练，GPU 1 启动 vLLM 服务充当评分 API：

```bash
export CUDA_VISIBLE_DEVICES=0,1
bash self_eval_step_gdpo_local.sh
```

local 脚本自动在 GPU 1 启动 vLLM server（默认端口 8199），等待就绪后在 GPU 0 启动训练，退出时自动 kill vLLM 进程。

**API 环境变量**（优先级：CLI 参数 > 环境变量 > 默认值）

| 环境变量 | 回退 | 默认值 | 说明 |
|----------|------|--------|------|
| `SELF_EVAL_MODEL` | `FOL_MODEL` | `gpt-4o-mini` | 评分模型名称 |
| `OPENAI_API_KEY` | — | `""` | API 密钥（本地用 `EMPTY`） |
| `OPENAI_BASE_URL` | — | `None` | API 端点 |

**Reward Manager 集成**（`step.py:122` / `tree.py:136`，懒加载，与 fol/format 注册方式一致）：

```python
if "self_eval" in self.step_reward_types:
    from verl.utils.reward_score.self_eval import compute_step_reward_self_eval
    if "self_eval" not in self.step_reward_fns:
        self.step_reward_fns["self_eval"] = compute_step_reward_self_eval
```

---

## 4. Anti-Reward-Hacking Penalty

### 背景

当 process reward（如 fol、self_eval）与最终任务目标（选择题正确率）不完全对齐时，模型可能学会 exploit process reward 信号而不是真正提升推理能力。典型症状包括：

- **Step 膨胀**：num_steps 从 ~8 涨到 25+，模型拆碎推理为大量局部正确但不推进解题的 step
- **长度爆炸**：response_length 冲向 max_response_length，clip_ratio 从 ~3% 涨到 40%+
- **格式退化**：多个 `\boxed{}`、`<step>`/`</step>` 不匹配、`<conclusion>` 出现在 `<step>` 外
- **局部重复**：同一 premise 换句话重复说，拆成多个 step 骗取 process reward
- **准确率坍塌**：val acc 持续下降（如 0.28 → 0.07），而 fol_step_reward/mean 持续上升

根因：Step-GDPO 的 reward-to-go（reverse cumsum）会将 step 数量作为乘数放大 process advantage。当 `step_reward_weights=[0.5, 0.5]` 且 step 数从 6 涨到 25 时，process 梯度信号远超 outcome，模型被激励写更多局部可判正的句子而非提高最终答案正确率。

### 修复策略

**权重调整**：fol/self_eval 脚本已从 `[0.5, 0.5]` 改为 `[0.8, 0.2]`，让 outcome advantage 主导优化方向。

**Penalty 机制**（`verl/experimental/reward_loop/reward_manager/format_penalty.py`，StepRewardManager 与 TreeRewardManager 共用同一实现）：训练层的格式处罚全部使用 `penalty_score` 一个分值。除截断外处罚是局部的：坏 step 只罚该 step，其余 step 正常验证。这些规则只作用于训练；validation 返回原始 benchmark 分数（`validate_with_step_reward=false` 时在规则之前返回）。

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.penalty_max_steps` | `0`（禁用） | 超过阈值的后缀 step 各自得到 penalty_score，前缀不受影响（推荐 12） |
| `algorithm.penalty_on_truncated` | `False` | response 被 max_response_length 截断时，全部 step 置 penalty_score 且不再验证 |
| `algorithm.penalty_on_multi_boxed` | `False` | boxed 契约的总开关（名称沿用旧配置，0 个的规则也由它控制）：0 个 `\boxed{}` 时 outcome 置为 penalty_score、process 正常；多个时以第一个为准在其前缀上重评 outcome，之后的每个 `\boxed{}` 在自身位置得到一个 penalty_score |
| `algorithm.penalty_on_bad_format` | `False` | 坏 step（未闭合、缺 premise/conclusion 等）单独置 penalty_score，其余 step 正常验证；`<step>` 块外散落的 reasoning tag 在其位置得到一个 penalty_score；response 没有任何 `<step>` 时在末 token 置一个 penalty_score |
| `algorithm.penalty_score` | `0.0` | 上述全部处罚共用的分值（生产训练用 -1.0） |

### 使用示例

```bash
# 在现有脚本基础上追加 penalty 参数
+algorithm.penalty_max_steps=12 \
+algorithm.penalty_on_truncated=true \
+algorithm.penalty_on_multi_boxed=true \
+algorithm.penalty_on_bad_format=true \
+algorithm.penalty_score=0.0 \
```

当 penalty 触发时，reward_extra_info 中会记录 `process_reward_penalized=True` 和 `penalty_reason`（如 `"num_steps=25>12|truncated"`），方便日志排查。

---

## 5. 公用参数

以下参数在所有脚本中保持一致：

| 参数 | 值 | 说明 |
|------|-----|------|
| `model.path` | Qwen2.5-1.5B-Instruct | 基础模型 |
| `data` | logiqa2k (train + validation) | 数据集 |
| `max_prompt_length` | 2048 | 最大 prompt 长度 |
| `max_response_length` | 2048 | 最大响应长度 |
| `actor.optim.lr` | 1e-6 | 学习率 |
| `actor.use_kl_loss` | True | 开启 KL 散度损失 |
| `actor.kl_loss_coef` | 0.02 | KL 系数 |
| `actor.kl_loss_type` | low_var_kl | 低方差 KL |
| `rollout.temperature` | 0.8 | 采样温度 |
| `rollout.top_p` | 0.95 | top-p 采样 |
| `rollout.gpu_memory_utilization` | 0.5 | vLLM 显存占比 |
| `use_kl_in_reward` | False | reward 中不加 KL |
| `total_epochs` | 1 | 总训练轮次 |
| `test_freq` | 100 | 测试频率（步） |
| `n_gpus_per_node` | 1 | 每节点 GPU 数 |
