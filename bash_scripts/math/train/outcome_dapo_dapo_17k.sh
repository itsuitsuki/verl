#!/bin/bash
set -x

# DAPO outcome-only on DAPO-Math-17k: the outcome baseline for the math Step-GDPO Isabelle run.
# Matches that run's controlled variables (data, batch sizes, seeds, rollout sampling, lr, KL,
# lengths, total steps, val suite); differs only in the algorithm/reward pair: GRPO advantage +
# DAPO outcome reward with the overlong buffer, no step rewards, no Isabelle, no translator.
# Outcome grading routes through default_compute_score, which maps the
# open-r1/DAPO-Math-17k-Processed data_source to math-verify boxed-gated scoring
# (verl/utils/reward_score/__init__.py), the same route the Step-GDPO run uses for its
# outcome component and validation.
# Env setup (conda, WANDB_API_KEY) should be done before running this script.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-4B CUDA_VISIBLE_DEVICES=0,1,2,3 bash bash_scripts/math/train/outcome_dapo_dapo_17k.sh

export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
export WANDB_MODE=${WANDB_MODE:-online}
export VLLM_ATTENTION_BACKEND=XFORMERS
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:?'MODEL_PATH must be set'}
MODEL_TAG=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    N_GPUS=$(nvidia-smi -L | wc -l)
fi
echo "Training on $N_GPUS GPUs"

EXP_NAME=${EXP_NAME:-${MODEL_TAG}_dapo_outcome_dapo17k_v1}

mkdir -p logs
RUN_STAMP="dapo_outcome_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "logs/${RUN_STAMP}.log") 2>&1
echo "Logging to logs/${RUN_STAMP}.log"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=1536 \
    "data.train_files=[data/dapo_math/train.parquet]" \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.data_loader_seed=42 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl-fol-2 \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=checkpoints/verl-fol/$EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_training_steps=1056 \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=true \
    "$@"
