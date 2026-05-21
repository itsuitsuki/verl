set -x

# FOL Step-GDPO — Reclor 7B on H100
#
# 2× H100 layout:
#   GPU 0: Qwen2.5-7B-Instruct training (actor + rollout + ref)
#   GPU 1: Qwen3.6-35B-A3B local vLLM judge (TP1)
#
# Usage:
#   srun -p gpu_h100 --gres=gpu:2 -t 24:00:00 bash bash_scripts/fol_step_gdpo_reclor_7b_h100.sh

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-7B-Instruct
JUDGE_MODEL_PATH=~/run/models/Qwen3.6-35B-A3B
DATA_NAME=reclor_prompt_v2
DATA_DIR=${DATA_DIR:-"$HOME/run/work/verl/data/${DATA_NAME}"}
JUDGE_PORT=4872

export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# ── Start local judge on GPU 1 ──
echo "Starting vLLM judge on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model ${JUDGE_MODEL_PATH} \
    --served-model-name Qwen3.6-35B-A3B \
    --port ${JUDGE_PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --max-cudagraph-capture-size 128 &
JUDGE_PID=$!

cleanup() {
    echo "Cleaning up judge (PID=$JUDGE_PID)..."
    kill $JUDGE_PID 2>/dev/null
    wait $JUDGE_PID 2>/dev/null
}
trap cleanup EXIT

# Wait for judge to be ready
echo "Waiting for judge to start on port ${JUDGE_PORT}..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:${JUDGE_PORT}/v1/models > /dev/null 2>&1; then
        echo "Judge is ready."
        break
    fi
    if ! kill -0 $JUDGE_PID 2>/dev/null; then
        echo "Judge process died. Exiting."
        exit 1
    fi
    sleep 5
done

if ! curl -s http://127.0.0.1:${JUDGE_PORT}/v1/models > /dev/null 2>&1; then
    echo "Judge failed to start within 10 minutes. Exiting."
    exit 1
fi

# ── FOL judge env ──
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export FOL_MODEL="Qwen3.6-35B-A3B"
export FOL_RPM=0
export FOL_OPENAI_TPM=0
export FOL_OPENAI_MAX_INFLIGHT=${FOL_OPENAI_MAX_INFLIGHT:-128}
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
export STEP_REWARD_MAX_WORKERS=${STEP_REWARD_MAX_WORKERS:-16}

# ── Training on GPU 0 ──
echo "Starting Reclor 7B FOL Step-GDPO training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_type=fol \
    +algorithm.fol_max_tries=1 \
    +algorithm.fol_timeout=10 \
    +algorithm.api_timeout=200 \
    +algorithm.fol_cumulative_mode=current_only \
    +algorithm.fol_judge_use_outlines=true \
    +algorithm.validate_with_step_reward=false \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.8, 0.2]' \
    +algorithm.penalty_max_steps=12 \
    +algorithm.penalty_on_truncated=true \
    +algorithm.penalty_on_multi_boxed=true \
    +algorithm.penalty_on_bad_format=true \
    +algorithm.penalty_score=-1.0 \
    reward_model.reward_manager=step \
    reward.num_workers=${REWARD_NUM_WORKERS} \
    +algorithm.step_reward_max_workers=${STEP_REWARD_MAX_WORKERS} \
    +reward.api_config.api_context_shrink_min_tokens=16 \
    +reward.api_config.api_context_shrink_retries=6 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-fol-2' \
    trainer.experiment_name="qwen7b_step_gdpo_reclor_fol_h100_v1" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=true \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 $@
