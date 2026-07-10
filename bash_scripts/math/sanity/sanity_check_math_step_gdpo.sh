#!/bin/bash
# Sanity check for math_step_gdpo_isabelle_combined.sh: 3 training steps,
# tiny val, no checkpoints. Confirms Isabelle pool startup, judge
# translation, verification, and a non-trivial step_gdpo/fol_step_reward.
# (Replaces the former sanity_check_fv_math.sh / launch_isabelle_feas1.sh.)
#
# Usage (judge must already be serving at OPENAI_BASE_URL):
#   MODEL=Qwen/Qwen3-4B CUDA_VISIBLE_DEVICES=0,2 bash bash_scripts/math/sanity/sanity_check_math_step_gdpo.sh
set -e

EXP_NAME=sanity_math_step_gdpo \
bash bash_scripts/math/train/math_step_gdpo_isabelle_combined.sh \
    trainer.total_training_steps=3 \
    trainer.save_freq=9999 \
    trainer.test_freq=9999 \
    trainer.val_before_train=false \
    trainer.resume_mode=disable \
    "data.val_files=[data/aime24/test.parquet]" \
    trainer.logger='["console"]' \
    "$@"
