#!/bin/bash
set -x

# DataTech-2 baseline wrapper for Qwen3-8B
# Usage: METHOD=grpo CUDA_VISIBLE_DEVICES=0,1 bash bash_scripts/baseline_datatech2.sh
#        METHOD=gspo CUDA_VISIBLE_DEVICES=2,3 bash bash_scripts/baseline_datatech2.sh
#        METHOD=dapo CUDA_VISIBLE_DEVICES=4,5 bash bash_scripts/baseline_datatech2.sh

source /2022533109/liubushi/miniconda3/etc/profile.d/conda.sh && conda activate verl
cd /2022533109/zhouchuyan/verl

export WANDB_API_KEY=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
export MODEL_PATH=/2022533109/zhouchuyan/models/Qwen3-8B
export DATA_DIR=/2022533109/zhouchuyan/verl/data/combined_logic
export WANDB_ENTITY=verl-fol

METHOD=${METHOD:?'METHOD must be set (grpo|gspo|dapo)'}
echo "Running $METHOD baseline with Qwen3-8B on GPU $CUDA_VISIBLE_DEVICES"
exec bash bash_scripts/${METHOD}_combined.sh trainer.resume_mode=auto "$@"
