#!/bin/bash
# Health checker for the Isabelle 4B training. Emits ONE status line:
#   DEAD   - trainer process gone (crash)
#   STALL  - training log frozen >40min = ANY wedge the in-process guards did not
#            prevent (reward-worker python/C spin, pool wedge, etc). This is the
#            real backstop; the runaway-poly reaper handles poly spins in-process,
#            so we do NOT re-flag poly here (cumulative CPU is a false-positive
#            signal -- a long-lived poly running a normal proof also shows R + high
#            cumulative CPU).
#   OK     - healthy, reports current step + log age
L=$(ls -t /2022533109/zhouchuyan/verl/logs/math_combined_qwen3-4b_*.log 2>/dev/null | head -1)
P=$(pgrep -f 'trainer[.]main_ppo' | head -1)
[ -z "$P" ] && { echo "DEAD"; exit 0; }
[ -z "$L" ] && { echo "OK step=? logage=?"; exit 0; }
S=$(grep -oE 'training/global_step:[0-9]+' "$L" | tail -1 | grep -oE '[0-9]+$')
NOW=$(date +%s); MT=$(stat -c %Y "$L" 2>/dev/null); AGE=$((NOW - MT))
if [ "$AGE" -gt 2400 ]; then echo "STALL logfrozen=${AGE}s step=${S:-?}"; exit 0; fi
echo "OK step=${S:-?} logage=${AGE}s"
