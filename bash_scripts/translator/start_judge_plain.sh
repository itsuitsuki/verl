#!/bin/bash
# Launch a verl-fol translator (Qwen3.6-35B-A3B) with the PROVEN plain config —
# NO speculative decoding. On vLLM 0.19 + hybrid GDN (Qwen3.5/3.6) + TP=2,
# every spec-decode path is broken (2026-07-10, verified live + upstream):
#   - ngram: GDN/SSM state cannot roll back on draft rejection -> silently
#     corrupted outputs, then engine crashes (vllm#39273, #39809-type assert)
#   - qwen3_next_mtp / DFlash: cudaErrorIllegalAddress at TP=2 (vllm#41190)
#   - MTP + prefix caching: ~20% accuracy drop (vllm#43559)
# Revisit only after those fixes land in a vLLM upgrade.
# Usage: start_judge_plain.sh <gpus> <port> <tag>   e.g. 4,5 4873 j1
# NOTE: no `set -u` (conda activate hooks reference unset vars).
GPUS=${1:?usage: start_judge_plain.sh <gpus> <port> <tag>}
PORT=${2:?usage: start_judge_plain.sh <gpus> <port> <tag>}
TAG=${3:?usage: start_judge_plain.sh <gpus> <port> <tag>}
source /2022533109/liubushi/miniconda3/etc/profile.d/conda.sh
conda activate verl
cd /2022533109/zhouchuyan/verl
TS=$(date +%Y%m%d_%H%M%S)
echo "Starting translator ${TAG} on GPUs ${GPUS} port ${PORT} (plain, no spec decode)"
CUDA_VISIBLE_DEVICES=$GPUS vllm serve /2022533109/zhouchuyan/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B --port "$PORT" \
    --max-model-len 12288 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 256 \
    2>&1 | tee "logs/judge_${TAG}_plain_${TS}.log"
