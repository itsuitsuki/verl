#!/bin/bash
# Math Step-GDPO Isabelle on DAPO-17k (override of math_step_gdpo_math_combined.sh)
exec bash "$(dirname "$0")/math_step_gdpo_math_combined.sh" \
    "data.train_files=[data/dapo_math/train.parquet]" \
    "$@"
