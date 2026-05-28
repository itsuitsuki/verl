#!/bin/bash
set -x

# Same-node 4×A800 validation: Qwen3-4B FOL Step GDPO on combined logic
# GPU 0-1: judge (Qwen3.6-35B-A3B TP2), GPU 2-3: training
# Usage: srun -G4 -p gpu_a800 -t 360 bash bash_scripts/fol_samenode_4b_a800.sh

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate verl
cd ~/run/work/verl

export LD_PRELOAD=/data/home/scyb676/run/.conda/envs/verl/lib/libstdc++.so.6
unset OPENAI_BASE_URL
export MODEL_PATH=~/run/models/Qwen3-4B
export JUDGE_MODEL=~/run/models/Qwen3.6-35B-A3B
export WANDB_ENTITY=verl-fol

echo "Node: $(hostname), GPUs: $(nvidia-smi -L | wc -l)"

exec bash bash_scripts/fol_step_gdpo_combined.sh trainer.total_epochs=1
