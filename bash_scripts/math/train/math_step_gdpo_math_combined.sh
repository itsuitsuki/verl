#!/bin/bash
set -x

# Math Step-GDPO on Combined Math Set
# Math counterpart of fol_step_gdpo_logic_combined.sh

# Requires: Isabelle 2025 installed, translator vLLM running at OPENAI_BASE_URL,
# or set TRANSLATOR_MODEL/TRANSLATOR_DEVICES to auto-start one.

# Model via TRAIN_MODEL (HF-style id resolved under the shared models dir, or a
# literal path); training GPUs via plain CUDA_VISIBLE_DEVICES:
#   TRAIN_MODEL=Qwen/Qwen3-4B CUDA_VISIBLE_DEVICES=x,y bash bash_scripts/math/train/math_step_gdpo_math_combined.sh
#   TRAIN_MODEL=Qwen/Qwen3-8B CUDA_VISIBLE_DEVICES=x,y bash bash_scripts/math/train/math_step_gdpo_math_combined.sh

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
TRANSLATOR_PORT=${TRANSLATOR_PORT:-4873}
# Default to the resident local translator (tmux "translator1-4873") on this node;
# unset OPENAI_BASE_URL explicitly AND set TRANSLATOR_MODEL to auto-start one instead.
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:${TRANSLATOR_PORT}/v1}

RUN_STAMP="math_combined_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_STAMP}"          # translator.log only (created on translator auto-start)

mkdir -p logs
exec > >(tee -a "logs/${RUN_STAMP}.log") 2>&1
echo "Logging to logs/${RUN_STAMP}.log"

# --- Translator setup (auto-start only when no external translator is given) ---
TRANSLATOR_PID=""
if [ -z "$OPENAI_BASE_URL" ]; then
    TRANSLATOR_MODEL=${TRANSLATOR_MODEL:?'TRANSLATOR_MODEL or OPENAI_BASE_URL must be set'}
    TRANSLATOR_TP=${TRANSLATOR_TP:-2}
    TRANSLATOR_DEVICES=${TRANSLATOR_DEVICES:-0,1}
    mkdir -p "$LOG_DIR"
    echo "Starting local translator on GPU $TRANSLATOR_DEVICES (TP=$TRANSLATOR_TP)... log: $LOG_DIR/translator.log"
    CUDA_VISIBLE_DEVICES=$TRANSLATOR_DEVICES vllm serve $TRANSLATOR_MODEL \
        --served-model-name Qwen3.6-35B-A3B \
        --port $TRANSLATOR_PORT \
        --max-model-len 12288 \
        --tensor-parallel-size $TRANSLATOR_TP \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --max-num-seqs 256 \
        > "$LOG_DIR/translator.log" 2>&1 &
    TRANSLATOR_PID=$!
    export OPENAI_BASE_URL="http://127.0.0.1:${TRANSLATOR_PORT}/v1"

    echo "Waiting for translator..."
    for i in $(seq 1 300); do
        if curl -s http://127.0.0.1:$TRANSLATOR_PORT/health > /dev/null 2>&1; then
            echo "Translator ready after ${i}s"
            break
        fi
        sleep 1
    done
    if ! curl -s http://127.0.0.1:$TRANSLATOR_PORT/health > /dev/null 2>&1; then
        echo "ERROR: Translator failed to start"
        kill $TRANSLATOR_PID 2>/dev/null
        exit 1
    fi
else
    echo "Using external translator at $OPENAI_BASE_URL"
fi

TRAIN_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
N_GPUS=$(echo "$TRAIN_DEVICES" | tr ',' '\n' | wc -l)
echo "Model: $MODEL_PATH | Training GPUs: $TRAIN_DEVICES ($N_GPUS)"

EXP_NAME=${EXP_NAME:-${MODEL_TAG}_step_gdpo_isabelle_math_combined_v3}

# --- Reap leftover Isabelle/Poly processes from a previous crash ---
_jvms=$(pgrep -cf '[I]sabelle2025' || true)
_polys=$(pgrep -c -x poly || true)
if [ "${_jvms:-0}" -gt 0 ] || [ "${_polys:-0}" -gt 0 ]; then
    echo "Reaping leftovers: ${_jvms:-0} Isabelle JVMs, ${_polys:-0} Poly/ML"
    pgrep -f '[I]sabelle2025' | xargs -r kill -9 2>/dev/null
    pgrep -x poly | xargs -r kill -9 2>/dev/null
    sleep 3
fi

# --- Training ---
# fol_task_type=math routes step rewards to Isabelle verification (not Z3).
# nccl_timeout raises the collective watchdog above PyTorch's 30-minute default.
# A step here takes 18-45 minutes and the slowest single reward computation has been measured at 34 minutes,
# so a rank that trails one phase behind the others exceeds the default window; the ranks already waiting in
# the collective then abort and take the whole job down (measured 2026-08-06 on step 539: ranks 0/2/3 waited
# 1800 s for an FSDP all-gather that rank 1 never joined).
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
    +algorithm.isabelle_each_worker_proc_tree_mem_max_gb=12 \
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
    actor_rollout_ref.nccl_timeout=7200 \
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
#     bash math_step_gdpo_math_combined.sh trainer.resume_mode=disable
#   resume with 4 pool workers + full pattern logging:
#     bash math_step_gdpo_math_combined.sh \
#         ++algorithm.isabelle_pool_workers=4 \
#         ++algorithm.isabelle_each_worker_proc_tree_mem_max_gb=12 \
#         +trainer.print_all_step_patterns=true
# print_all_step_patterns defaults OFF via self.config.trainer.get(..., False).

TRAIN_EXIT=$?
echo "Training finished with exit code $TRAIN_EXIT"
if [ -n "$TRANSLATOR_PID" ]; then
    kill $TRANSLATOR_PID 2>/dev/null
fi
exit $TRAIN_EXIT
