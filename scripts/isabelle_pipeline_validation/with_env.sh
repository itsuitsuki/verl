#!/bin/bash
# Environment wrapper for the full-pipeline validation scripts on datatech: Isabelle 2025,
# the verl conda env, and the shared repo on PYTHONPATH. Usage:
#   bash with_env.sh python -u <script> [args...]
source /2022533109/zhouchuyan/isabelle/env.sh
source /2022533109/liubushi/miniconda3/bin/activate verl
export PYTHONPATH=/2022533109/zhouchuyan/verl:${PYTHONPATH:-}
exec "$@"
