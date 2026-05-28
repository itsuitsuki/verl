#!/bin/bash
set -x

# Paratera baseline wrapper for Qwen3-4B
# Usage: srun -G2 -p gpu_a800 -t 480 bash bash_scripts/baseline_paratera.sh grpo
#        srun -G2 -p gpu_a800 -t 480 bash bash_scripts/baseline_paratera.sh gspo
#        srun -G2 -p gpu_a800 -t 480 bash bash_scripts/baseline_paratera.sh dapo

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh && conda activate verl
cd ~/run/work/verl

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export MODEL_PATH=~/run/models/Qwen3-4B
export WANDB_ENTITY=verl-fol

METHOD=${1:?'Usage: baseline_paratera.sh {grpo|gspo|dapo}'}
echo "Node: $(hostname), GPUs: $(nvidia-smi -L | wc -l)"
exec bash bash_scripts/${METHOD}_combined.sh trainer.resume_mode=auto
