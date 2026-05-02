#!/usr/bin/env bash
export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data/gsm8k"}
FORMAT=${FORMAT:-xml}
ANSWER_FORMAT=${ANSWER_FORMAT:-boxed}
PROMPT_FILE=${PROMPT_FILE:-math_reasoning.txt}

ARGS=(
    --format "$FORMAT"
    --answer_format "$ANSWER_FORMAT"
    --local_save_dir "$DATA_DIR"
)

if [[ -n "${PROMPT_FILE:-}" ]]; then
    ARGS+=(--system_prompt_file "$PROMPT_FILE")
fi

if [[ -n "${USER_PROMPT_FILE:-}" ]]; then
    ARGS+=(--user_prompt_file "$USER_PROMPT_FILE")
fi

python "$REPO_ROOT/examples/data_preprocess/gsm8k.py" "${ARGS[@]}" "$@"

echo "Wrote GSM8K parquet files to: $DATA_DIR"
echo "Format: $FORMAT"
echo "Answer format: $ANSWER_FORMAT"
if [[ -n "${PROMPT_FILE:-}" ]]; then
    echo "System prompt file: $PROMPT_FILE"
fi
