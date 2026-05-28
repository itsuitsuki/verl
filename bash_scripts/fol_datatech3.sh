#!/bin/bash
set -x

# DataTech-3 FOL Step GDPO wrapper
# GPU 6,7: judge (Qwen3.6-35B-A3B TP2), GPU 4: training
# Usage: MODEL_SIZE=4b bash bash_scripts/fol_datatech3.sh
#        MODEL_SIZE=8b bash bash_scripts/fol_datatech3.sh

source /2022533109/liubushi/miniconda3/etc/profile.d/conda.sh && conda activate verl
cd /2022533109/zhouchuyan/verl

export WANDB_API_KEY=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
export WANDB_ENTITY=verl-fol
export JUDGE_DEVICES=6,7
export TRAIN_DEVICES=4
export JUDGE_TP=2
export JUDGE_MODEL=/2022533109/zhouchuyan/models/Qwen3.6-35B-A3B
unset OPENAI_BASE_URL

EXTRA_ARGS=""
if [ "$MODEL_SIZE" = "8b" ]; then
    export MODEL_PATH=/2022533109/zhouchuyan/models/Qwen3-8B
    EXTRA_ARGS="actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
else
    export MODEL_PATH=/2022533109/zhouchuyan/models/Qwen3-4B
fi

echo "Model: $MODEL_PATH, Judge GPU: $JUDGE_DEVICES, Train GPU: $TRAIN_DEVICES"
exec bash bash_scripts/fol_step_gdpo_combined.sh $EXTRA_ARGS "$@"
