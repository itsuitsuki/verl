#!/usr/bin/env bash
set -euo pipefail
set -x

# Outcome-only greedy pass@1 test runner.
#
# This script is intentionally independent of training reward scripts. It does
# not launch FOL judges, self-eval judges, or step-reward validation. It only
# loads the base actor or a checkpoint and evaluates dataset outcome accuracy.
#
# Typical use:
#   CUDA_VISIBLE_DEVICES=1 \
#   DATA_DIR=/2022533109/zhouchuyan/verl/data/logiqa2k_prompt_v2 \
#   MODEL_PATH=/2022533109/zhouchuyan/models/Qwen2.5-1.5B-Instruct \
#   CHECKPOINT_PATH=checkpoints/verl-fol/qwen1.5b_step_gdpo_fol_gpu2_v4/global_step_1844 \
#   RUN_NAME=qwen1.5b_step_gdpo_fol_logiqa_test_final1844 \
#   bash bash_scripts/eval/pass_at_1_test.sh 2>&1 | tee test_fol_step_gdpo_logiqa_final1844.log

HOME_DIR=${HOME:-/root}
DATA_NAME=${DATA_NAME:-logiqa2k_prompt_v2}
DATA_DIR=${DATA_DIR:-"$HOME_DIR/run/work/verl/data/${DATA_NAME}"}

if [ ! -f "$DATA_DIR/train.parquet" ] && [ -d "/2022533109/zhouchuyan/verl/data/${DATA_NAME}" ]; then
    DATA_DIR="/2022533109/zhouchuyan/verl/data/${DATA_NAME}"
fi

MODEL_PATH=${MODEL_PATH:-"$HOME_DIR/run/models/Qwen2.5-1.5B-Instruct"}
if [ ! -d "$MODEL_PATH" ] && [ -d "/2022533109/zhouchuyan/models/Qwen2.5-1.5B-Instruct" ]; then
    MODEL_PATH="/2022533109/zhouchuyan/models/Qwen2.5-1.5B-Instruct"
fi

VAL_FILE=${VAL_FILE:-${TEST_FILE:-"$DATA_DIR/test.parquet"}}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${RESUME_FROM_PATH:-}}
RUN_NAME=${RUN_NAME:-"$(basename "$MODEL_PATH")_${DATA_NAME}_pass_at_1"}

VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.50}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: train parquet does not exist: $DATA_DIR/train.parquet" >&2
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: VAL_FILE does not exist: $VAL_FILE" >&2
    exit 1
fi

if [ -n "$CHECKPOINT_PATH" ] && [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH" >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATA_DIR=$DATA_DIR"
echo "VAL_FILE=$VAL_FILE"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH:-<base model>}"
echo "RUN_NAME=$RUN_NAME"

resume_args=()
if [ -n "$CHECKPOINT_PATH" ]; then
    resume_args=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$CHECKPOINT_PATH"
    )
fi

EVAL_RUNS=${EVAL_RUNS:-3}
ALL_ACCS=()

for run_i in $(seq 1 "$EVAL_RUNS"); do
    echo ""
    echo "========== Eval run $run_i / $EVAL_RUNS =========="
    run_output=$(python3 -u -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        +algorithm.validate_with_step_reward=false \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len="$MAX_RESPONSE_LENGTH" \
        data.train_files="$DATA_DIR/train.parquet" \
        data.val_files="$VAL_FILE" \
        data.train_batch_size=4 \
        data.val_batch_size="$VAL_BATCH_SIZE" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.dataloader_num_workers=0 \
        ++data.apply_chat_template_kwargs.enable_thinking=false \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
        actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
        actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=false \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name=verl-fol-2 \
        trainer.experiment_name="$RUN_NAME" \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=9999 \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.test_freq=50 \
        trainer.total_epochs=1 \
        trainer.val_before_train=true \
        trainer.val_only=true \
        trainer.default_local_dir="checkpoints/verl-fol/${RUN_NAME}" \
        ++data.seed=42 \
        actor_rollout_ref.actor.data_loader_seed=42 \
        critic.data_loader_seed=42 \
        "${resume_args[@]}" \
        "$@" 2>&1) || true

    echo "$run_output"

    acc=$(echo "$run_output" | grep -oP "val-core/.+?/acc/mean@1[^0-9]*\K[0-9.]+" | tail -1)
    if [ -z "$acc" ]; then
        echo "WARNING: could not parse accuracy from run $run_i"
    else
        ALL_ACCS+=("$acc")
        echo ">>> Run $run_i accuracy: $acc"
    fi
done

echo ""
echo "========== Summary ($EVAL_RUNS runs) =========="
if [ ${#ALL_ACCS[@]} -eq 0 ]; then
    echo "ERROR: no accuracy values collected"
    exit 1
fi

python3 -c "
import sys
accs = [float(x) for x in sys.argv[1:]]
n = len(accs)
mean = sum(accs) / n
std = (sum((x - mean)**2 for x in accs) / n) ** 0.5
print('Individual: %s' % ['%.4f' % a for a in accs])
print('Mean:  %.4f' % mean)
print('Std:   %.4f' % std)
print('Range: %.4f ~ %.4f' % (min(accs), max(accs)))
print('Result: %.2f%% +/- %.2f%%' % (mean * 100, std * 100))
" "${ALL_ACCS[@]}"
