#!/usr/bin/env bash
export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

PROMPT_FILE=${PROMPT_FILE:-logical_reasoning.txt}
DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data/ar_lsat_prompt_v2"}
NUM_SAMPLES=${NUM_SAMPLES:--1}
FORMAT=${FORMAT:-xml}

python "$REPO_ROOT/examples/data_preprocess/ar_lsat.py" \
    --num_samples "$NUM_SAMPLES" \
    --format "$FORMAT" \
    --local_save_dir "$DATA_DIR" \
    --system_prompt_file "$PROMPT_FILE" \
    "$@"

echo "Wrote AR-LSAT prompt-v2 parquet files to: $DATA_DIR"
echo "Prompt file: $PROMPT_FILE"
