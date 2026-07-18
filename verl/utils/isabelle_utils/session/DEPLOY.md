# Isa_Step Isabelle session — source bundle & deploy

This directory is the **version-controlled source** for the Isabelle/HOL session
that the step-reward engine (`../server_pool.py`, `../engine.py`, `../tactics.py`)
proves theorems against. The compiled heap lives only on the datatech shared disk
and is a build artifact; these files are what regenerate it.

## Files

| File | Deploy target (datatech, shared) | Purpose |
|------|----------------------------------|---------|
| `ROOT` | `/2022533109/zhouchuyan/isabelle/isa_step/ROOT` | session definition (parent `HOL-Analysis` + `HOL-Number_Theory`/`HOL-Real_Asymp`/`HOL-Probability`/`HOL-Algebra`; no `quick_and_dirty` — nothing uses `sorry`, and the runtime options pin `quick_and_dirty=false` so the kernel enforces it) |
| `Isa_Step_Base.thy` | `/2022533109/zhouchuyan/isabelle/isa_step/Isa_Step_Base.thy` | base theory: re-exports Analysis/Real_Asymp/Sum_of_Squares/Code_Target_Numeral/Number_Theory/Approximation **plus** HOL-Algebra (Coset/Multiplicative_Group/Ring, for the direct group/ring/field checks since the 2026-07-17 pool merge) **plus** the general trigonometric value and sign meta-theorems (`div_sqrt_eq`, `cos_abs_from_tan`, `sin/cos_from_tan_c{pos,neg}`, `cos_from_sin_c{pos,neg}`, `sin_from_cos_s{pos,neg}`, `cos_{pos,neg}_q1..q4`, `sin_pos_upper`/`sin_neg_lower`) that the trig proof attempts instantiate (`verl/utils/isabelle_utils/trigonometry.py`), and the pigeonhole `min_rep` battery |
| `env.sh` | `/2022533109/zhouchuyan/isabelle/env.sh` | sets `ISABELLE_HOME` + `ISABELLE_HOME_USER` to the shared dir (no node-local `~/.isabelle` symlink; survives container restart) |

## Two out-of-repo edits NOT captured as full files (documented here)

1. **Distribution `etc/settings` patch** — `Isabelle2025/etc/settings` wraps the
   default `ISABELLE_HOME_USER` assignment in a guard so `env.sh`'s value wins
   (the distribution unconditionally reset it to `$USER_HOME/.isabelle`, a
   node-local path). Backup: `settings.bak-homeuser`. The change (around line 77):

   ```sh
   # The place for user configuration, heap files, etc.
   if [ -z "$ISABELLE_HOME_USER" ]; then          # <-- added
   if [ -z "$ISABELLE_IDENTIFIER" ]; then
     ISABELLE_HOME_USER="$USER_HOME/.isabelle"
   else
     ISABELLE_HOME_USER="$USER_HOME/.isabelle/$ISABELLE_IDENTIFIER"
   fi
   fi                                              # <-- added
   ```

2. **User `ROOTS`** — `/2022533109/zhouchuyan/isabelle/user/ROOTS` must list the
   session dir so `isabelle build` finds it:

   ```
   /2022533109/zhouchuyan/isabelle/afp-2026-06-12/thys
   /2022533109/zhouchuyan/isabelle/isa_step
   ```

## Rebuild (after editing Isa_Step_Base.thy)

```sh
source /2022533109/zhouchuyan/isabelle/env.sh
cd /2022533109/zhouchuyan/isabelle/isa_step
isabelle build -b Isa_Step        # heap -> shared user/heaps/, restart-safe; was ~1m10s pre-merge, longer with the HOL-Algebra theories (HOL-Algebra's own heap already exists in user/heaps from the former domain pools)
```

Then in `_server_pool/config.py`:
- `THEORY_IMPORTS` must be the **session-qualified** import `"Isa_Step.Isa_Step_Base"`
  (a bare `Isa_Step_Base` fails to resolve to the session theory).
- NO theorem-cache version bump is needed for a heap rebuild: cache entries are keyed
  by a fingerprint that includes the session heap, so the rebuild makes every old entry
  unreachable automatically (project CLAUDE.md rule 8). `_THM_CACHE_VERSION` changes only
  when the CACHE ENTRY FORMAT changes. Current: `v0`.
- Stop any live pool BEFORE the rebuild (never rebuild a heap a running pool is using),
  and restart reward workers afterwards so new pools load the new heap.

## Transfer note (post-container-restart)

`scp`/SFTP can land outside the Singularity container where `/2022533109` is not
the bind-mounted path. If `scp` fails with "No such file or directory", transfer
via `base64 local | ssh datatech-N "base64 -d > remote"` (runs inside the
container where the mount is visible).
