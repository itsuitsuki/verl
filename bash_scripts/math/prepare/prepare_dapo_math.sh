#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data/dapo_math"}
PROMPT_FILE=${PROMPT_FILE:-math_reasoning.txt}
DATA_CONFIG=${DATA_CONFIG:-en}

python "$REPO_ROOT/examples/data_preprocess/dapo_math.py" \
    --local_save_dir "$DATA_DIR" \
    --system_prompt_file "$PROMPT_FILE" \
    --config "$DATA_CONFIG" \
    "$@"

printf 'Wrote DAPO-Math %s configuration to: %s\n' "$DATA_CONFIG" "$DATA_DIR"
