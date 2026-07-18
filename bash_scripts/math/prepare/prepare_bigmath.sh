#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# Raw snapshot of open-r1/Big-Math-RL-Verified-Processed. Download once with:
#   HF_ENDPOINT=https://hf-mirror.com hf download open-r1/Big-Math-RL-Verified-Processed \
#       --repo-type dataset --local-dir "$RAW_DIR"
RAW_DIR=${RAW_DIR:-"$REPO_ROOT/data/raw_bigmath_processed"}
DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data/bigmath_clean"}
PROMPT_FILE=${PROMPT_FILE:-math_reasoning.txt}
SOLVE_RATE_MIN=${SOLVE_RATE_MIN:-0.0}
SOLVE_RATE_MAX=${SOLVE_RATE_MAX:-0.9}

ARGS=(
    --raw_dir "$RAW_DIR"
    --local_save_dir "$DATA_DIR"
    --solve_rate_min "$SOLVE_RATE_MIN"
    --solve_rate_max "$SOLVE_RATE_MAX"
)

if [[ -n "${PROMPT_FILE:-}" ]]; then
    ARGS+=(--system_prompt_file "$PROMPT_FILE")
fi

python "$REPO_ROOT/examples/data_preprocess/bigmath.py" "${ARGS[@]}" "$@"

echo "Wrote Big-Math parquet to: $DATA_DIR"
echo "Solve-rate band: [$SOLVE_RATE_MIN, $SOLVE_RATE_MAX]"
