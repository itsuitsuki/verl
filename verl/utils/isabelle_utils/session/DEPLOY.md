# Math_Verify Isabelle session — source bundle & deploy

This directory is the **version-controlled source** for the Isabelle/HOL session
that the step-reward engine (`../server_pool.py`, `../engine.py`, `../tactics.py`)
proves theorems against. The compiled heap lives only on the datatech shared disk
and is a build artifact; these files are what regenerate it.

## Files

| File | Deploy target (datatech, shared) | Purpose |
|------|----------------------------------|---------|
| `ROOT` | `/2022533109/zhouchuyan/isabelle/math_verify/ROOT` | session definition (parent `HOL-Analysis` + `HOL-Number_Theory`/`HOL-Real_Asymp`/`HOL-Probability`, `quick_and_dirty` so `sorry` is allowed) |
| `Math_Verify_Base.thy` | `/2022533109/zhouchuyan/isabelle/math_verify/Math_Verify_Base.thy` | base theory: re-exports Analysis/Real_Asymp/Sum_of_Squares/Code_Target_Numeral/Number_Theory/Approximation **plus** the precompiled quadrant-trig meta-theorems (`div_sqrt_eq`, `cos_abs_from_tan`, `sin/cos_from_tan_c{pos,neg}`, `cos_{pos,neg}_q1..q4`, `sin_pos_upper`/`sin_neg_lower`) that `engine.trig_quadrant_theorem` instantiates |
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
   /2022533109/zhouchuyan/isabelle/math_verify
   ```

## Rebuild (after editing Math_Verify_Base.thy)

```sh
source /2022533109/zhouchuyan/isabelle/env.sh
cd /2022533109/zhouchuyan/isabelle/math_verify
isabelle build -b Math_Verify        # ~1m10s; heap -> shared user/heaps/, restart-safe
```

Then in `server_pool.py`:
- `THEORY_IMPORTS` must be the **session-qualified** import `"Math_Verify.Math_Verify_Base"`
  (a bare `Math_Verify_Base` fails to resolve to the session theory).
- Bump `_THM_CACHE_VERSION` (session content changed) so stale disk-cached
  verdicts are invalidated. Current: `v3`.

## Transfer note (post-container-restart)

`scp`/SFTP can land outside the Singularity container where `/2022533109` is not
the bind-mounted path. If `scp` fails with "No such file or directory", transfer
via `base64 local | ssh datatech-N "base64 -d > remote"` (runs inside the
container where the mount is visible).
