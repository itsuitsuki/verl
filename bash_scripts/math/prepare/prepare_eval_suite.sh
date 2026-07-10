#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# Builds the 6 held-out math eval parquets (gsm8k test already exists via
# prepare_gsm8k.sh): math500 / aime24 / aime25 / amc23 / minervamath /
# olympiadbench, each as data/<bench>/test.parquet with the gsm8k-identical
# extra_info struct. ONLY=aime25 rebuilds a single bench.
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data"}
PROMPT_FILE=${PROMPT_FILE:-math_reasoning.txt}

ARGS=(
    --local_save_root "$DATA_ROOT"
)

if [[ -n "${PROMPT_FILE:-}" ]]; then
    ARGS+=(--system_prompt_file "$PROMPT_FILE")
fi

if [[ -n "${ONLY:-}" ]]; then
    ARGS+=(--only "$ONLY")
fi

# datatech: prepend HF_ENDPOINT=https://hf-mirror.com (no direct HF access)
python "$REPO_ROOT/examples/data_preprocess/eval_suite.py" "${ARGS[@]}" "$@"

echo "Wrote eval-suite parquets under: $DATA_ROOT"
