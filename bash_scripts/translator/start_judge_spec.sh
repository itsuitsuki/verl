#!/bin/bash
# Launch a verl-fol judge (Qwen3.6-35B-A3B) with ngram speculative decoding.
# Usage: start_judge_spec.sh <gpus> <port> <tag>
#   e.g. start_judge_spec.sh 4,5 4873 j1
#
# Same flags as the proven judge command (CLAUDE.md) plus --speculative-config:
# translation output largely COPIES tokens from the prompt (steps text -> formal
# props), which is exactly what prompt-lookup (ngram) speculation accelerates.
# Speculative decoding is exact (rejection sampling preserves the output
# distribution), so rewards are unaffected.
# NOTE: no `set -u` -- conda's activate/deactivate hooks reference unset vars
# (CONDA_BACKUP_CXX) and nounset would abort the script inside conda activate.
GPUS=${1:?usage: start_judge_spec.sh <gpus> <port> <tag>}
PORT=${2:?usage: start_judge_spec.sh <gpus> <port> <tag>}
TAG=${3:?usage: start_judge_spec.sh <gpus> <port> <tag>}
source /2022533109/liubushi/miniconda3/etc/profile.d/conda.sh
conda activate verl
cd /2022533109/zhouchuyan/verl
TS=$(date +%Y%m%d_%H%M%S)
echo "Starting judge ${TAG} on GPUs ${GPUS} port ${PORT} (ngram spec decode)"
CUDA_VISIBLE_DEVICES=$GPUS vllm serve /2022533109/zhouchuyan/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B --port "$PORT" \
    --max-model-len 12288 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 256 \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 8, "prompt_lookup_max": 8, "prompt_lookup_min": 2}' \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    2>&1 | tee "logs/judge_${TAG}_spec_${TS}.log"
# PIECEWISE cudagraphs: default FULL_AND_PIECEWISE capture crashes with ngram
# spec decode on this hybrid GDN (linear-attention) model -- the Triton GDN
# kernel is not capturable under multi-token verify shapes ("operation not
# permitted when stream is capturing", 2026-07-10). PIECEWISE splits around
# the GDN op (gdn_attention_core is in vllm's splitting_ops) and keeps the
# cudagraph benefit for the rest of the model.
