#!/usr/bin/env bash
export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# Base prompt mode:
# - no system prompt file
# - uses built-in instruction from examples/data_preprocess/logiqa.py:
#   Please reason step by step with steps separated by "\n\n", and put the
#   letter of the correct option within \boxed{{}}.

DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data/logiqa_base_prompt"}
VERSION=${VERSION:-1}
NUM_SAMPLES=${NUM_SAMPLES:--1}
FORMAT=${FORMAT:-xml}

python "$REPO_ROOT/examples/data_preprocess/logiqa.py" \
    --version "$VERSION" \
    --num_samples "$NUM_SAMPLES" \
    --format "$FORMAT" \
    --local_save_dir "$DATA_DIR" \
    "$@"

echo "Wrote base-prompt parquet files to: $DATA_DIR"
