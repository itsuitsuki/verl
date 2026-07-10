#!/bin/bash
# Health checker for the Isabelle 4B training. Emits ONE status line:
#   DEAD          - trainer process gone (crash)
#   STALL         - training log frozen >40min (wedge of any kind)
#   REWARD_DEAD   - SILENT reward outage: latest batch has format_ok>0.3 but
#                   givens_ok==0 => every translation failing (translator dead
#                   or judge pipeline broken) while training keeps running.
#                   This is the 2026-07-10 failure mode the old checker missed:
#                   spec-decode translators crashed at 08:50, training ran 14
#                   more steps with all process rewards silently zeroed.
#   TRANSLATOR_DOWN - fewer than 2 judge ports listening
#   OK            - healthy, reports step + log age + givens_ok
L=$(ls -t /2022533109/zhouchuyan/verl/logs/math_combined_qwen3-4b_*.log 2>/dev/null | head -1)
P=$(pgrep -f 'trainer[.]main_ppo' | head -1)
[ -z "$P" ] && { echo "DEAD"; exit 0; }
PORTS=$(ss -ltn 2>/dev/null | grep -cE ':4873|:4874')
[ "$PORTS" -lt 2 ] && { echo "TRANSLATOR_DOWN ports=$PORTS"; exit 0; }
[ -z "$L" ] && { echo "OK step=? logage=?"; exit 0; }
S=$(grep -oE 'training/global_step:[0-9]+' "$L" | tail -1 | grep -oE '[0-9]+$')
NOW=$(date +%s); MT=$(stat -c %Y "$L" 2>/dev/null); AGE=$((NOW - MT))
if [ "$AGE" -gt 2400 ]; then echo "STALL logfrozen=${AGE}s step=${S:-?}"; exit 0; fi
LAST=$(grep 'isabelle/givens_ok/mean' "$L" | tail -1)
GO=$(echo "$LAST" | grep -oE 'isabelle/givens_ok/mean:[0-9.]+' | cut -d: -f2)
FO=$(echo "$LAST" | grep -oE 'isabelle/format_ok/mean:[0-9.]+' | cut -d: -f2)
if [ -n "$GO" ] && [ -n "$FO" ]; then
  DEAD_REWARD=$(awk -v go="$GO" -v fo="$FO" 'BEGIN{print (fo>0.3 && go<0.01) ? 1 : 0}')
  [ "$DEAD_REWARD" = 1 ] && { echo "REWARD_DEAD step=${S:-?} format_ok=$FO givens_ok=$GO"; exit 0; }
fi
echo "OK step=${S:-?} logage=${AGE}s givens_ok=${GO:-n/a}"
