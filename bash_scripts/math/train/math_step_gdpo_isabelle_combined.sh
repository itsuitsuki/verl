#!/bin/bash
set -x

# Production Step-GDPO + Isabelle verification on the combined math set.
# Math counterpart of fol_step_gdpo_combined.sh (single-layer, self-contained;
# absorbed the former fv_step_gdpo_math.sh core on 2026-07-05).
# Requires: Isabelle 2025 installed, judge vLLM running (tmux "judge") at
# OPENAI_BASE_URL -- or set JUDGE_MODEL/JUDGE_DEVICES to auto-start one.
#
# Model via MODEL (HF-style id resolved under the shared models dir, or a
# literal path); training GPUs via plain CUDA_VISIBLE_DEVICES:
#   MODEL=Qwen/Qwen3-4B CUDA_VISIBLE_DEVICES=0,2 bash bash_scripts/math/train/math_step_gdpo_isabelle_combined.sh
#   MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=4   bash bash_scripts/math/train/math_step_gdpo_isabelle_combined.sh
#
# Design (2026-07-05):
#   train = [gsm8k 7.5k, MATH 7.5k, bigmath 161k (solve_rate 0-0.9)]
#   val   = 7-bench suite (gsm8k / MATH-500 / AIME24 / AIME25 / AMC23 /
#           Minerva / OlympiadBench), full sets, dual-graded (acc = each
#           bench's standard scorer, acc_mathverify logged alongside)
#   batch 16 (prompts) x rollout 16 = 256 responses/step (feasibility's
#   128-response batches left the Isabelle pool at ~40% utilization)
#   1250 steps = 20k prompts = 320k responses (~= logic combined's 1-epoch
#   gradient budget); extend by resume if curves still climb
#   reward = math-verify boxed-gated on all sources; save/test every 50
# Contamination note: gsm8k/MATH are saturated in the Qwen3 base (report
# tables 6/7); expected movement is on MATH-500/Minerva/OlympiadBench/AIME.
#
# Docs: verl/utils/isabelle_utils/README.md

cd /2022533109/zhouchuyan/verl
source .envrc                       # WANDB_API_KEY (direnv does not fire in tmux)
source /2022533109/liubushi/miniconda3/etc/profile.d/conda.sh
conda activate verl
ulimit -n 65536                     # FD limit: vLLM + Ray + Isabelle JVMs
# PolyML heap cap lives in /2022533109/zhouchuyan/isabelle/user/etc/settings
# (--maxheap 8G). Memory math (2026-07-07 OOM postmortem, corrected): each
# Isabelle worker runs ~2 `poly` heap processes, so ISABELLE_POOL_WORKERS(3) x
# reward.num_workers(4) = 12 JVMs => ~24 poly. Each poly's RSS ratchets up to
# its 5-6GB working set over the first few steps (not the 8G cap). At 5 pool
# workers (40 poly) that was 40 x ~6GB = 240GB provers + ~85GB fixed
# (2x23GB FSDP WorkerDicts + ray + JVMs) => ~325GB, over the 322GB cgroup cap
# (mem climbed +32GB/step, OOM by step ~105). 3 pool workers => 24 poly x 6GB
# = ~145GB + ~71GB fixed = ~216GB, ~100GB margin. Override via ISABELLE_POOL_WORKERS.

# Source Isabelle environment
if [ -f /2022533109/zhouchuyan/isabelle/env.sh ]; then
    source /2022533109/zhouchuyan/isabelle/env.sh
fi

# fontconfig lives in the ephemeral container overlay and vanishes on every
# container restart (2026-07-11 incident: every session_start FAILED with
# "Fontconfig head is null"). Unlike the user-dir (fixed persistently via
# ISABELLE_HOME_USER in env.sh) there is no shared-disk persistence for an
# apt package, so self-heal idempotently at launch.
if ! command -v fc-list >/dev/null 2>&1; then
    echo "fontconfig missing (container restart) -- reinstalling"
    apt-get install -y fontconfig >/dev/null 2>&1 || {
        echo "ERROR: fontconfig install failed; Isabelle session_start will fail"; exit 1; }
fi

# Ensure the node-local ISABELLE_HOME_USER symlink points at the shared user
# dir. /root/.isabelle is node-local and is WIPED on a container/tmux-server
# restart; if it is missing or gets recreated as a plain dir, the prebuilt
# HOL-Library / HOL-Number_Theory heaps vanish -> every worker's session_start
# fails (missing heap -> failed rebuild -> SQLite registry corruption) and
# training hangs at the first step. Recreate it idempotently so a post-restart
# resume self-heals. (2026-07-07 incident postmortem.)
ISA_USER_LINK=/root/.isabelle/Isabelle2025
ISA_USER_SHARED=/2022533109/zhouchuyan/isabelle/user
if [ "$(readlink -f "$ISA_USER_LINK" 2>/dev/null)" != "$ISA_USER_SHARED" ]; then
    mkdir -p /root/.isabelle
    rm -rf "$ISA_USER_LINK"
    ln -s "$ISA_USER_SHARED" "$ISA_USER_LINK"
    echo "Isabelle user-dir symlink (re)created: $ISA_USER_LINK -> $ISA_USER_SHARED"
fi

export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
export WANDB_MODE=${WANDB_MODE:-online}
# Isabelle per-worker poly-tree RSS cap is now a Hydra knob
# (+algorithm.isabelle_worker_rss_cap_gb, default 12 in the training args
# below), NOT an env var. 2026-07-11 measurement: a healthy 2-poly tree sits
# ~6GB steady, and 12 JVMs x 12GB = 144GB polys + ~106GB fixed = ~250GB stays
# under the ~285GB Ray-kill on the 300GB cgroup. Do NOT raise >=14 (transient
# JVM overlap would overrun). Override per-run: ++algorithm.isabelle_worker_rss_cap_gb=N.
export VLLM_ATTENTION_BACKEND=XFORMERS
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# --- Model resolution ---
MODEL=${MODEL:-Qwen/Qwen3-4B}
MODELS_DIR=/2022533109/zhouchuyan/models
if [ -d "$MODEL" ]; then
    MODEL_PATH="$MODEL"                              # literal path
else
    MODEL_PATH="$MODELS_DIR/$(basename "$MODEL")"    # HF-id form
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model dir not found: $MODEL_PATH (from MODEL=$MODEL)" >&2
    exit 1
fi
MODEL_TAG=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

# 8B-on-H20 memory rule (CLAUDE.md rule 7): offload trio all-on
# (optimizer_offload alone asserts at engine init). 4B fits without offload.
case "$MODEL_TAG" in
    *8b*|*14b*|*32b*)
        OFFLOAD_ACTOR=True
        ;;
    *)
        OFFLOAD_ACTOR=False
        ;;
esac

export FOL_MODEL="Qwen3.6-35B-A3B"
JUDGE_PORT=${JUDGE_PORT:-4873}
# Default to the resident local judge (tmux "judge") on this node; unset
# OPENAI_BASE_URL explicitly AND set JUDGE_MODEL to auto-start one instead.
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}

RUN_STAMP="math_combined_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_STAMP}"          # judge.log only (created on judge auto-start)
# Training log: flat file under logs/, same convention as the previous
# fol/grpo combined runs (e.g. logs/fol_8b_combined_20260609_194942.log).
mkdir -p logs
exec > >(tee -a "logs/${RUN_STAMP}.log") 2>&1
echo "Logging to logs/${RUN_STAMP}.log"

# --- Judge setup (auto-start only when no external judge is given) ---
JUDGE_PID=""
if [ -z "$OPENAI_BASE_URL" ]; then
    JUDGE_MODEL=${JUDGE_MODEL:?'JUDGE_MODEL or OPENAI_BASE_URL must be set'}
    JUDGE_TP=${JUDGE_TP:-2}
    JUDGE_DEVICES=${JUDGE_DEVICES:-0,1}
    mkdir -p "$LOG_DIR"
    echo "Starting local judge on GPU $JUDGE_DEVICES (TP=$JUDGE_TP)... log: $LOG_DIR/judge.log"
    CUDA_VISIBLE_DEVICES=$JUDGE_DEVICES vllm serve $JUDGE_MODEL \
        --served-model-name Qwen3.6-35B-A3B \
        --port $JUDGE_PORT \
        --max-model-len 12288 \
        --tensor-parallel-size $JUDGE_TP \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --max-num-seqs 256 \
        > "$LOG_DIR/judge.log" 2>&1 &
    JUDGE_PID=$!
    export OPENAI_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"

    echo "Waiting for judge..."
    for i in $(seq 1 300); do
        if curl -s http://127.0.0.1:$JUDGE_PORT/health > /dev/null 2>&1; then
            echo "Judge ready after ${i}s"
            break
        fi
        sleep 1
    done
    if ! curl -s http://127.0.0.1:$JUDGE_PORT/health > /dev/null 2>&1; then
        echo "ERROR: Judge failed to start"
        kill $JUDGE_PID 2>/dev/null
        exit 1
    fi
else
    echo "Using external judge at $OPENAI_BASE_URL"
fi

TRAIN_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
N_GPUS=$(echo "$TRAIN_DEVICES" | tr ',' '\n' | wc -l)
echo "Model: $MODEL_PATH | Training GPUs: $TRAIN_DEVICES ($N_GPUS)"

# W&B run name, following the logic-side convention
# (${MODEL_TAG}_step_gdpo_fol_combined_v1): model_algo_verifier_dataset_ver.
# v3 (2026-07-11): official from-scratch run. Adds the giant-number guard
# (dangerous claims -> eval, dangerous/timeout probes -> undetermined safe
# mode instead of grinding simp/presburger), verify_timeout=60, worker RSS
# cap 12GB. This changes the reward for the dangerous/timeout tail vs v2, so
# the whole run must be from scratch for one reward definition (v2's step-0..N
# checkpoints are retained but must NOT be auto-resumed into v3 -- hence the
# separate name; resume mode is the base-config default `auto`, overridable
# on the CLI: pass trainer.resume_mode=disable for a fresh from-scratch run).
EXP_NAME=${EXP_NAME:-${MODEL_TAG}_step_gdpo_isabelle_math_combined_v3}

# --- Training ---
# fol_task_type=math routes step rewards to Isabelle verification (not Z3).
CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_type=fol \
    +algorithm.fol_task_type=math \
    +algorithm.fol_max_tries=1 \
    +algorithm.verify_timeout=60 \
    +algorithm.api_timeout=200 \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.8, 0.2]' \
    +algorithm.penalty_max_steps=30 \
    +algorithm.penalty_on_truncated=true \
    +algorithm.penalty_on_multi_boxed=true \
    +algorithm.penalty_on_bad_format=true \
    +algorithm.penalty_score=-1.0 \
    +algorithm.validate_with_step_reward=false \
    ++algorithm.fol_cumulative_mode=step \
    reward_model.reward_manager=step \
    reward.num_workers=4 \
    +algorithm.step_reward_max_workers=128 \
    +algorithm.isabelle_pool_workers=3 \
    +algorithm.isabelle_worker_rss_cap_gb=12 \
    "data.train_files=[data/gsm8k/train.parquet,data/math/train.parquet,data/bigmath/train.parquet]" \
    "data.val_files=[data/gsm8k/test.parquet,data/math500/test.parquet,data/aime24/test.parquet,data/aime25/test.parquet,data/amc23/test.parquet,data/minervamath/test.parquet,data/olympiadbench/test.parquet]" \
    data.train_batch_size=16 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    ++data.apply_chat_template_kwargs.enable_thinking=false \
    ++data.seed=42 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD_ACTOR \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD_ACTOR \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD_ACTOR \
    algorithm.use_kl_in_reward=False \
    critic.data_loader_seed=42 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl-fol-2 \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=checkpoints/verl-fol/$EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_training_steps=1250 \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=true \
    "$@"
# resume_mode is left at the base-config default (auto); pass it and any other
# Hydra override straight through "$@". Examples:
#   from-scratch official run:
#     bash math_step_gdpo_isabelle_combined.sh trainer.resume_mode=disable
#   resume with 4 pool workers + full pattern logging:
#     bash math_step_gdpo_isabelle_combined.sh \
#         ++algorithm.isabelle_pool_workers=4 \
#         ++algorithm.isabelle_worker_rss_cap_gb=12 \
#         +trainer.print_all_step_patterns=true
# print_all_step_patterns defaults OFF via self.config.trainer.get(..., False).

TRAIN_EXIT=$?
echo "Training finished with exit code $TRAIN_EXIT"
if [ -n "$JUDGE_PID" ]; then
    kill $JUDGE_PID 2>/dev/null
fi
exit $TRAIN_EXIT
