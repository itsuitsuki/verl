#!/bin/bash
set -x

# FOL Step GDPO on Combined Logic (LogiQA+Reclor+AR-LSAT)
# Env setup (conda, WANDB_API_KEY, LD_PRELOAD etc.) should be done before running this script.
#
# Mode 1 - embedded judge (same node):
#   JUDGE_MODEL=/path/to/model bash bash_scripts/logic/train/fol_step_gdpo_combined.sh
#   Override: JUDGE_TP, JUDGE_DEVICES, TRAIN_DEVICES
#
# Mode 2 - external judge (cross-node, judge already running):
#   OPENAI_BASE_URL=http://<judge-host>:4873/v1 bash bash_scripts/logic/train/fol_step_gdpo_combined.sh

export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
export WANDB_MODE=${WANDB_MODE:-online}
export VLLM_ATTENTION_BACKEND=XFORMERS
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

TRAIN_MODEL=${MODEL_PATH:?'MODEL_PATH must be set'}
DATA_DIR=${DATA_DIR:-data/combined_logic}
MODEL_TAG=$(basename "$TRAIN_MODEL" | tr '[:upper:]' '[:lower:]')
JUDGE_PORT=${JUDGE_PORT:-4873}

export FOL_MODEL="Qwen3.6-35B-A3B"

LOG_DIR="logs/fol_combined_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# --- Judge setup ---
JUDGE_PID=""
if [ -z "$OPENAI_BASE_URL" ]; then
    # Mode 1: start local judge
    JUDGE_MODEL=${JUDGE_MODEL:?'JUDGE_MODEL or OPENAI_BASE_URL must be set'}
    JUDGE_TP=${JUDGE_TP:-2}
    JUDGE_DEVICES=${JUDGE_DEVICES:-0,1}
    echo "Starting local judge on GPU $JUDGE_DEVICES (TP=$JUDGE_TP)... logs: $LOG_DIR/judge.log"
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

    echo "Waiting for judge to be ready..."
    for i in $(seq 1 300); do
        if curl -s http://127.0.0.1:$JUDGE_PORT/health > /dev/null 2>&1; then
            echo "Judge ready after ${i}s"
            break
        fi
        sleep 1
    done

    if ! curl -s http://127.0.0.1:$JUDGE_PORT/health > /dev/null 2>&1; then
        echo "ERROR: Judge failed to start within 300s"
        kill $JUDGE_PID 2>/dev/null
        exit 1
    fi
    TRAIN_DEVICES=${TRAIN_DEVICES:-2,3}
else
    # Mode 2: external judge
    echo "Using external judge at $OPENAI_BASE_URL"
    TRAIN_DEVICES=${TRAIN_DEVICES:-0,1}
fi

# --- Count training GPUs ---
N_GPUS=$(echo "$TRAIN_DEVICES" | tr ',' '\n' | wc -l)
echo "Training on GPUs: $TRAIN_DEVICES ($N_GPUS GPUs)"

# --- Start training ---
CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_type=fol \
    +algorithm.fol_max_tries=1 \
    +algorithm.fol_timeout=10 \
    +algorithm.api_timeout=200 \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.8, 0.2]' \
    +algorithm.penalty_max_steps=12 \
    +algorithm.penalty_on_truncated=true \
    +algorithm.penalty_on_multi_boxed=true \
    +algorithm.penalty_on_bad_format=true \
    +algorithm.penalty_score=-1.0 \
    +algorithm.validate_with_step_reward=false \
    reward_model.reward_manager=step \
    reward.num_workers=64 \
    +algorithm.step_reward_max_workers=128 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    ++data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=$TRAIN_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl-fol-2 \
    trainer.experiment_name=${MODEL_TAG}_step_gdpo_fol_combined_v1 \
    trainer.default_local_dir=checkpoints/verl-fol/${MODEL_TAG}_step_gdpo_fol_combined_v1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=true \
    trainer.resume_mode=auto \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 \
    ++algorithm.fol_cumulative_mode=current_only \
    ++algorithm.fol_judge_use_outlines=true \
    "$@"

TRAIN_EXIT=$?
echo "Training finished with exit code $TRAIN_EXIT"
if [ -n "$JUDGE_PID" ]; then
    echo "Judge log: $LOG_DIR/judge.log"
    kill $JUDGE_PID 2>/dev/null
fi
exit $TRAIN_EXIT
