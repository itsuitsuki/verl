#!/bin/bash
set -x

# Math Step-GDPO on Combined Math Set
# Math counterpart of fol_step_gdpo_combined.sh

# Requires: Isabelle 2025 installed, translator vLLM running at OPENAI_BASE_URL,
# or set JUDGE_MODEL/JUDGE_DEVICES to auto-start one.

# Model via TRAIN_MODEL (HF-style id resolved under the shared models dir, or a
# literal path); training GPUs via plain CUDA_VISIBLE_DEVICES:
#   TRAIN_MODEL=Qwen/Qwen3-4B CUDA_VISIBLE_DEVICES=x,y bash bash_scripts/math/train/math_step_gdpo_isabelle_combined.sh
#   TRAIN_MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=x,y bash bash_scripts/math/train/math_step_gdpo_isabelle_combined.sh

# Source Isabelle environment
if [ -f /2022533109/zhouchuyan/isabelle/env.sh ]; then
    source /2022533109/zhouchuyan/isabelle/env.sh
fi

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
export VLLM_ATTENTION_BACKEND=XFORMERS
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Model configuration
TRAIN_MODEL=${MODEL_PATH:?'MODEL_PATH must be set'}
MODELS_DIR=/2022533109/zhouchuyan/models
if [ -d "$TRAIN_MODEL" ]; then
    MODEL_PATH="$TRAIN_MODEL"                              # literal path
else
    MODEL_PATH="$MODELS_DIR/$(basename "$TRAIN_MODEL")"    # HF-id form
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model dir not found: $MODEL_PATH (from TRAIN_MODEL=$TRAIN_MODEL)" >&2
    exit 1
fi
MODEL_TAG=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

export FOL_MODEL="Qwen3.6-35B-A3B"
JUDGE_PORT=${JUDGE_PORT:-4873}
# Default to the resident local judge (tmux "judge") on this node; unset
# OPENAI_BASE_URL explicitly AND set JUDGE_MODEL to auto-start one instead.
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}

RUN_STAMP="math_combined_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_STAMP}"          # judge.log only (created on judge auto-start)

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
    "data.train_files=[data/gsm8k/train.parquet,data/math/train.parquet,data/bigmath_clean/train.parquet]" \
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
    actor_rollout_ref.model.path=$TRAIN_MODEL \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
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
