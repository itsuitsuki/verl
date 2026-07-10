# Isabelle/HOL Step-Reward Integration (`verl.utils.isabelle_utils`)

Isabelle/HOL step-level verification wired into verl's Step-GDPO RL loop as a
per-step process reward for **math** reasoning. Selected at runtime with
`fol_task_type=math`. This file is the single source of truth for using and
understanding the integration; it folds in the former
`isabelle-integration-plan.md` design doc (that standalone doc has been retired
and `E:/AMPR/isabelle-integration.md` now symlinks back here).

**At a glance**

- **Run training:** `bash bash_scripts/fv_step_gdpo_math.sh` (needs a judge vLLM + Isabelle env sourced).
- **Sanity check:** `bash bash_scripts/sanity_check_fv_math.sh` (3 steps, console-only).
- **Engine:** `verl/utils/isabelle_utils/engine.py` — `IsabelleEngine.verify_solution(problem, response, ground_truth)`.
- **Reward entry:** `verl/utils/reward_score/formal_verify.py:compute_solution_reward_isabelle` (renamed from `fol.py` on 2026-07-03; dispatched from `experimental/reward_loop/reward_manager/step.py` when `verify_task_type=math`).
- **Status (2026-06-14):** D1–D6 complete (integration + 3-step sanity green on dt3); full-epoch training pending.

---

## Usage

Isabelle/HOL step-level verification for **math** reasoning RL. When
`fol_task_type=math`, each policy rollout is verified whole-solution: a judge
LLM translates the natural-language `<step>` chain into constrained Python
boolean expressions, which are transpiled to Isabelle terms and discharged by a
pool of resident `isabelle server` workers (session `HOL-Number_Theory`).
Per-step 0/1 rewards feed Step-GDPO.

> **CORRECTION:** `fol_task_type=math` now routes to Isabelle verification
> (changed 2026-06-14); the old Z3 Int/Real arithmetic path is
> `fol_task_type=math_z3`. The top-level README's `fol_task_type=math`
> description ("pure Int/Real 算术 schema") was stale — fixed 2026-06-15 to point to this README.

### 1. TL;DR

One runnable training command (external judge already serving on `:4873`):

```bash
source /2022533109/zhouchuyan/isabelle/env.sh      # ISABELLE_HOME + PATH + shared user heaps
OPENAI_BASE_URL=http://127.0.0.1:4873/v1 \
MODEL_PATH=/2022533109/zhouchuyan/models/Qwen3-4B \
bash bash_scripts/fv_step_gdpo_math.sh
```

`+algorithm.fol_task_type=math` (baked into the script) is what routes the
reward path to Isabelle instead of Z3.

### 2. Prerequisites

- **Isabelle 2025 env sourced** — `source /2022533109/zhouchuyan/isabelle/env.sh`
  sets `ISABELLE_HOME` (default in `server_pool.py`:
  `/2022533109/zhouchuyan/isabelle/Isabelle2025`) and points the per-node
  `~/.isabelle` at the shared user dir. The training scripts source it
  automatically if the file exists.
- **`fontconfig` installed on each node** — `apt install fontconfig`. Without
  it `isabelle build` silently exits 0 producing no heaps (dt1/dt2/dt3 already
  have it).
- **`HOL-Number_Theory` heaps prebuilt (shared)** — the pool starts session
  `HOL-Number_Theory` and additionally imports `Complex_Main`,
  `HOL-Library.Sum_of_Squares` (sos), `HOL-Library.Code_Target_Numeral`,
  `HOL-Number_Theory.Number_Theory`, `HOL-Decision_Procs.Approximation`. These
  heaps live in the shared user dir and are reused across nodes. To rebuild on
  a fresh node: `isabelle build -b -j 16 HOL-Number_Theory` (≈10 min wall on an
  H20 node; `HOL-Analysis` is the critical-path dependency).
- **Judge vLLM reachable** — the `OPENAI_BASE_URL` endpoint must answer
  `/health` before launch. `fv_step_gdpo_math.sh` polls `/health` for up to
  300s when it starts its own judge; for an external judge, verify reachability
  yourself first.

### 3. Judge vLLM launch

Serve the judge (Qwen3.6-35B-A3B, TP=2) on port 4873, in a tmux session named
`judge`:

```bash
CUDA_VISIBLE_DEVICES=4,5 vllm serve /root/run/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B --port 4873 \
    --max-model-len 12288 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 256
```

Training connects via `OPENAI_BASE_URL=http://127.0.0.1:4873/v1`. On DataTech
nodes the entrypoint form is used instead, and **`LD_PRELOAD` is mandatory** or
vLLM crashes on the old system libstdc++:

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model /2022533109/zhouchuyan/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B --port 4873 \
    --tensor-parallel-size 2 --max-model-len 12288 \
    --gpu-memory-utilization 0.95 --max-num-seqs 256 --enable-prefix-caching
```

### 4. Run training

`bash_scripts/fv_step_gdpo_math.sh` (GSM8K, Qwen3-4B/8B, Step-GDPO + Isabelle).

> NOTE: renamed from `fol_step_gdpo_math.sh`; use `fv_step_gdpo_math.sh`.

**External judge** (recommended — judge on another node/GPU):

```bash
source /2022533109/zhouchuyan/isabelle/env.sh
OPENAI_BASE_URL=http://127.0.0.1:4873/v1 \
MODEL_PATH=/2022533109/zhouchuyan/models/Qwen3-8B \
TRAIN_DEVICES=0,1,2,3 \
bash bash_scripts/fv_step_gdpo_math.sh
```

**Self-hosted judge** (script launches vLLM itself, then trains on the rest):

```bash
source /2022533109/zhouchuyan/isabelle/env.sh
MODEL_PATH=/2022533109/zhouchuyan/models/Qwen3-4B \
JUDGE_MODEL=/2022533109/zhouchuyan/models/Qwen3.6-35B-A3B \
JUDGE_DEVICES=0,1 TRAIN_DEVICES=2,3 \
bash bash_scripts/fv_step_gdpo_math.sh
```

Recognized env vars:

| Env var | Required | Default | Meaning |
|---|---|---|---|
| `MODEL_PATH` | yes | — | Policy model to train (also sets experiment tag). |
| `OPENAI_BASE_URL` | one of these | — | External judge endpoint. If set, `TRAIN_DEVICES` defaults to `0,1`. |
| `JUDGE_MODEL` | (if no `OPENAI_BASE_URL`) | — | Model path for the script-launched judge; `TRAIN_DEVICES` then defaults to `2,3`. |
| `TRAIN_DEVICES` | no | `0,1` / `2,3` | GPUs for training; `N_GPUS` is derived from the comma count. |
| `JUDGE_DEVICES` | no | `0,1` | GPUs for the self-hosted judge. |
| `JUDGE_PORT` | no | `4873` | Judge port. |
| `JUDGE_TP` | no | `2` | Judge tensor-parallel size. |
| `DATA_DIR` | no | `data/gsm8k` | Must contain `train.parquet` + `test.parquet`. |
| `WANDB_ENTITY` | no | `verl-fol` | Do not let it default to a personal entity. |

The Isabelle path is selected by `+algorithm.fol_task_type=math` and
`reward_model.reward_manager=step` (both baked in). Extra Hydra overrides can be
appended (they flow through `"$@"`).

### 5. Sanity check

`bash_scripts/sanity_check_fv_math.sh` — 3 training steps on GSM8K, console-only
logging, no checkpoints. Fastest way to confirm pool startup + judge
translation + Isabelle verification + reward return all work end to end.

> NOTE: renamed from `sanity_check_fol_math_isabelle.sh`; use `sanity_check_fv_math.sh`.

```bash
source /2022533109/zhouchuyan/isabelle/env.sh
OPENAI_BASE_URL=http://127.0.0.1:4873/v1 \
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/2022533109/zhouchuyan/models/Qwen3-4B \
bash bash_scripts/sanity_check_fv_math.sh
```

It sets `trainer.total_training_steps=3`, `trainer.logger='["console"]'`,
`save_freq=-1`, `train_batch_size=4`. `OPENAI_BASE_URL` is mandatory (it errors
out immediately if unset).

### 6. Config knobs

User-facing knobs (Hydra `+algorithm.*` flags; the reward manager forwards them
into `api_config`, which `_get_isabelle_engine` reads):

Naming note (2026-07-03): each `fol_*` knob gained a `verify_*` alias, which
**takes precedence** when both are set. The Isabelle-side scripts use the new
names; `fol_*` remains fully supported for the Z3 scripts and old runs. The
"fol" prefix is historical — it means "formal-verification step reward", not
First-Order Logic specifically.

| Knob (alias ← legacy) | Default | Values / effect |
|---|---|---|
| `verify_task_type` ← `fol_task_type` | `logic` | `math` → **Isabelle** whole-solution verification; `math_z3` → old Z3 Int/Real arithmetic path; `logic` → Z3 entity-predicate logic path. |
| `isabelle_pool_workers` | `32` | Resident `isabelle server` JVMs per engine (`IsabelleConfig.pool_workers`). **Lower this for multi-worker RL** — each reward worker process builds its own engine + pool. |
| `verify_timeout` ← `fol_timeout` | `60` (engine); scripts set `30` | Forwarded as `api_config["timeout"]`; `_get_isabelle_engine` reads it as `check_deadline` — per-`use_theories` deadline (s) before a stuck worker is restarted and the call counts as failure. |
| `verify_cumulative_mode` ← `fol_cumulative_mode` | `current_only` | Set `step` (as the scripts do) so each step is verified against its prior context. |
| `base_url` | `http://127.0.0.1:4873/v1` | Judge endpoint → `IsabelleConfig.judge_url`. Sourced from env `OPENAI_BASE_URL`. |
| `model` | `Qwen3.6-35B-A3B` | Judge model name → `IsabelleConfig.judge_model`. Sourced from env `FOL_MODEL` / `SELF_EVAL_MODEL`. |
| `reward.num_workers` | script `64` | Reward-side worker/thread count. Multiplies against `isabelle_pool_workers` for total JVM count. |

`IsabelleConfig` also fixes `session="HOL-Number_Theory"` and
`max_model_len=12288` (not exposed as Hydra flags).

### 7. Direct engine use

Call the engine standalone (condensed from
`scripts/isabelle_poc_math500/test_isabelle_engine.py`):

```python
from verl.utils.isabelle_utils.engine import IsabelleEngine, IsabelleConfig

config = IsabelleConfig(
    judge_url="http://127.0.0.1:4873/v1",
    judge_model="Qwen3.6-35B-A3B",
    pool_workers=8,          # 8 resident isabelle servers (~8s each to warm up)
    check_deadline=60.0,     # per-verification deadline in seconds
)
engine = IsabelleEngine(config)   # starts the pool; blocks until all workers ready

result = engine.verify_solution(
    problem="...",           # natural-language problem statement
    response="...",          # policy output: <step><premise/><conclusion/></step>... + \boxed{}
    ground_truth="42",       # final answer string
    dataset="gsm8k",
)

# result dict (same as engine.process_one):
#   format_ok, givens_ok, steps_ok, outcome_correct, n_steps, boxed,
#   premise_inconsistent_at, translate_a, translate_b,
#   steps: [ {step, verified, rewarded, neutral, guard_ok,
#             transcription_missing, premises_inconsistent, ...}, ... ]
print(result["steps_ok"], result["n_steps"])
for s in result["steps"]:
    print(s["step"], s["verified"], s["rewarded"], s["neutral"])

engine.shutdown()            # stops every isabelle server in the pool
```

`verify_solution(problem, response, ground_truth, dataset="math", idx=0,
sample=0) -> dict`. A definition-only step is marked `neutral=True` and excluded
from the reward denominator; `rewarded` requires `verified AND guard_ok AND NOT
premises_inconsistent AND NOT transcription_missing`.

### 8. Metrics

With the `ray_trainer.py` patch, every training/validation batch emits these
W&B keys (each as `.../mean`, `.../max`, `.../min` over the batch). **They
appear only from the next launch after that patch.**

Per-response counts:
`isabelle/format_ok`, `isabelle/givens_ok`, `isabelle/steps_ok`,
`isabelle/outcome_correct`, `isabelle/n_steps`, `isabelle/verified_steps`,
`isabelle/rewarded_steps`, `isabelle/neutral_steps`,
`isabelle/guard_failed_steps`, `isabelle/judge_calls_givens`,
`isabelle/judge_calls_steps`, `isabelle/judge_calls` (givens + steps).

Batch-normalized rates (`count / max(n_steps, 1)`):
`isabelle/verified_rate`, `isabelle/rewarded_rate`, `isabelle/neutral_rate`,
`isabelle/guard_failed_rate`.

`isabelle/rewarded_rate` is the primary training-signal health metric;
`isabelle/outcome_correct` tracks final-answer accuracy independently of
step verification.

**Unified `stepverify/*` namespace (2026-07-03):** both verifier backends
(Z3 and Isabelle) additionally emit the same backend-neutral names, so runs
overlay on one W&B panel: `stepverify/verified_steps` (verifier accepted:
Z3 "entailed" / Isabelle "verified"), `stepverify/verified_rate`,
`stepverify/proven_steps` (steps that earned reward 1 — for Z3 this equals
entailed; for Isabelle, verified AND guards passed), `stepverify/n_steps`,
`stepverify/translator_calls` (judge LLM translation calls). The
backend-specific namespaces above are kept unchanged.

**Per-response verdict pattern:** the engine builds a per-step verdict
string `<pattern>` per verification (no longer printed per-response — the
prints were removed as log spam; it surfaces in the trainer's `[Step Rewards]`
sample print via batch key `isabelle_pattern`). Symbols (evaluated in this
priority order at `engine.py` pattern build, `o`>`c`>`m`>`g`>`x`):

| Sym | Condition | Meaning |
|---|---|---|
| `o` | `rewarded` | verified AND guard_ok AND not inconsistent AND not transcription-missing — earns reward 1 |
| `c` | `premises_inconsistent` | at this step index the accumulated-premise set is provably self-contradictory (see below) |
| `m` | `verified` AND `transcription_missing` | Isabelle proved it, but the judge's translated term dropped a number the student's conclusion actually asserts — proof doesn't cover the real claim; fail-closed |
| `g` | `verified` but not rewarded | proved but reward withheld: either a definition-only step (`neutral`, no proof obligation) or the translated term invented a number absent from source (`guard_invented`) |
| `x` | none of the above | unverified — the step REACHED the prover but Isabelle could not derive its conclusion from the current premises (a genuine proof failure OR a timeout: watchdog abort / worker wedge both map to `success=False`) |

**`t` (translation-failed) — NOT a pattern symbol.** The pattern only holds
one symbol per step that *reached the prover*. When the judge cannot
formalize a step (givens/steps translation rejected — e.g. a bare-value
conclusion `0.088` that is not a boolean proposition, so `steps_ok=False`),
that XML step never produces a symbol. `formal_verify.py` counts these as
`t_steps = n_steps - len(pattern)` and exposes `isabelle/t_steps` +
`isabelle/t_rate`. Reward is 0 (fail-closed, same as `x`), but `t` is kept
SEPARATE so `x_rate` reads as "model's reasoning was wrong" and `t_rate` as
"translator/format could not formalize" — different fixes (RL reduces `x`;
`t` is a translator-prompt / data-shape limit). **Invariant:
`o + x + c + g + m + t == n_steps`**, so the six per-step rates sum to 1.
Distinct from `m`: `m` steps DID reach the prover and DID verify (they just
have an unbacked number); `t` steps never reached the prover at all.

**On `c` (the point most people get wrong):** each step is verified
*independently* under the sorry trick — every prior step's conclusion is
admitted as an axiom regardless of whether it verified. `c` does **not**
mean "a step failed, so everything after cascades". It is a *separate,
active* check (`engine.py:613-617`): a dedicated `⊢ False` theorem is run
against the accumulated premises, and `premise_inconsistent_at` is latched
to the **first** index where Isabelle actually *proves* `False`. From that
index onward every step is tagged `c` (`k >= premise_inconsistent_at`).

Consequences:
- An `x` does **not** force the rest to `c`. Admitting an unprovable-but-
  non-contradictory claim as an axiom leaves the premise set consistent, so
  later steps are judged on their own merits — patterns like `ooxoo` and
  `ooxoxox outcome=N` are normal.
- `c` appears only once a wrong step makes the axioms provably contradictory
  (e.g. admitting `36-10=27` clashes with arithmetic → `⊢ False`), and only
  *from that index on*; steps before it keep their real verdicts.
- Because inconsistent premises prove anything (ex falso), a genuinely
  contradictory point makes the step `verified` via explosion — which is
  exactly why it is shown as `c`, not `o`. Conversely an `x` is weak evidence
  that the premises are still consistent *as far as the prover's automation
  could tell* — Isabelle's `ALTERNATION` tactic is incomplete, so a truly
  contradictory set whose `⊢ False` it cannot find stays `x`, not `c`.

### 9. Operational gotchas

- **Raise the file-descriptor limit:** `ulimit -n 65536` before launch — each
  resident `isabelle server` holds sockets + files; the default 1024 is quickly
  exhausted.
- **Keep `isabelle_pool_workers` small for RL:** the engine is a per-process
  singleton, so total Isabelle JVMs ≈ `N_reward_workers × isabelle_pool_workers`.
  The default `32` is fine for a standalone smoke test but will OOM under many
  reward workers — size it so the product fits node RAM.
- **Batch divisibility:** `train_batch_size × rollout.n` must be divisible by
  `n_gpus`, and `train_batch_size >= 2 × n_gpus`. The shipped script uses
  `train_batch_size=8`, `rollout.n=16`; the sanity script uses
  `train_batch_size=4`.
- **GSM8K has no `validation.parquet`:** use `test.parquet` as `data.val_files`
  (both scripts already do).
- **Singularity 300 GB cgroup cap is shared across tenants** on DataTech nodes;
  a large pool competing with other jobs can be OOM-killed even if the node
  looks free — check with `htop`/`nvitop` (not `nvidia-smi`, which can't see
  cross-container procs).
- **Isabelle JVMs auto-die with the parent:** each server is spawned with
  `PR_SET_PDEATHSIG(SIGKILL)` and the engine registers an `atexit` shutdown, so
  a crashed/killed trainer will not leave orphan JVMs. If a worker wedges past
  `check_deadline`, the pool restarts just that worker and fails that one call
  closed.
- **Per-process engine singleton ignores later `api_config`:** the first
  `fol_task_type=math` caller's `judge_url` / `pool_workers` / `check_deadline`
  win for the whole process; subsequent differing configs are silently ignored.
  Fine for a single-reward-type run; would need a config-keyed engine dict to
  mix two Isabelle configs.

---

## Design & Background

> This section folds in the durable content of the original `isabelle-integration-plan.md` (drafted 2026-06-10, based on first-hand reading of the FoVer paper + repo). Naming has been corrected to shipped reality: the Isabelle reward path is selected with **`fol_task_type=math`** (the old Z3 math path is `fol_task_type=math_z3`, deprecated), the engine lives at **`verl/utils/isabelle_utils/engine.py`**, and training is driven by **`bash_scripts/fv_step_gdpo_math.sh`** (+ smoke test `bash_scripts/sanity_check_fv_math.sh`).

**Goal**: wire step-level Isabelle/HOL verification into verl's online RL loop as a *process reward* signal. The Z3 path only covers logical reasoning (LogiQA / FOLIO / Reclor) and GSM8K arithmetic; it cannot handle the induction / number-theory / algebraic tricks that MATH-500 needs. Isabelle fills the GSM8K → MATH-500 gap. AIME / Olympiad is a stretch goal requiring a dedicated prover (see the Layer-4 roadmap). Z3 is **not** replaced — Isabelle is an additive verifier routed by `fol_task_type` in parallel with Z3.

---

### 1. Terminology

- **Isabelle/HOL**: an interactive theorem prover (ITP). You write the proof as code and Isabelle checks every step is logically valid — a "math compiler". HOL = Higher-Order Logic, its built-in logic.
- **Z3**: an SMT solver — far more automated than Isabelle but weaker in expressivity (first-order logic + linear arithmetic, fully automatic "is this set of formulas contradictory?"). The current verl-fol online reward uses Z3.
- **ITP vs SMT (Isabelle vs Z3)**: ITP is expressive (Olympiad, induction) but needs explicit proof steps; SMT is weak (first-order ceiling) but fully automatic. Medium-difficulty math sits in the awkward gap where SMT can't and ITP is heavy — the target of the Isabelle integration.
- **tactic**: the "proof strategy" code for one step. Common ones: `by simp` (algebraic simplification, most common), `by auto` (simp + FOL search), `by arith`/`by linarith` (linear-arithmetic decision), `by blast` (FOL proof search), `by force` (aggressive combined automation).
- **sledgehammer**: Isabelle's brute-force tool — dispatches external SMT/ATP solvers (Z3/CVC4/Vampire/E) to find a proof for an open goal. Expensive (seconds to tens of seconds). FoVer uses it to *generate* training data; **we do not use it as the primary path** (the translator picks the tactic; sledgehammer is a slow fallback).
- **sorry**: Isabelle keyword meaning "skip proving this step, assume it holds". Disabled by default (anti-cheat); requires `quick_and_dirty=true`.
- **Sorry trick (core method)**: to verify step K of an N-step proof, `sorry`-out the other N−1 steps and let Isabelle really check only step K. Turns "whole proof judged pass/fail as a unit" into "each step judged independently". This is how FoVer produces a step-level signal.
- **Autoformalize**: automatically translate NL math into Isabelle syntax (`.thy`). Here the translator (judge model) does it — the analogue of the teacher model translating NL → Z3 code in the Z3 path.
- **miniF2F**: Polu et al. 2022 formal-math benchmark (AMC/AIME/IMO level), the standard Layer 3-4 benchmark.
- **PutnamBench**: harder than miniF2F (Putnam level); current SOTA ~15-20%.
- **DSP (Draft-Sketch-Prove)**: Jiang et al. 2023 — LLM writes a proof sketch, sledgehammer fills each step. FoVer's Checker class descends from this.
- **PISA**: Portal-to-ISAbelle (Albert Qjiang), a JVM RPC server wrapping Isabelle for long-lived reuse.
- **`isabelle client`**: Isabelle 2025's official protocol client, effectively "official PISA". We use this instead of PISA.
- **PIDE protocol**: Isabelle's official IPC protocol (Prover IDE), JSON over TCP. `isabelle server` + `isabelle client` speak this.
- **AFP**: Archive of Formal Proofs — the Isabelle community's shared proof library (PyPI-like), with number theory / advanced math / algorithms. GSM8K needs none; AIME number theory does.
- **PRM**: Process Reward Model — scores each reasoning step (vs ORM which scores only the final answer). FoVer trains a PRM.
- **Step-GDPO**: verl-fol's current RL algorithm (step-level variant of Group-Decoupled Policy Optimization) — normalizes process and outcome rewards separately then combines.

---

### 2. Isabelle capability tiers (Layer 1-4) + overall expectation

The question these tiers answer: under our "LLM translation + automatic solver" stack, what is the realistic per-tier coverage ceiling. Literature numbers below were re-measured in-house during the PoC (see Measurement history).

**Layer 1 — Isabelle kernel itself (human-expert hand-written)**: theoretically unbounded. Already formalized: Prime Number Theorem, Kepler conjecture (Flyspeck 2014), Four Color Theorem (Isabelle port), most of modern algebra / topology / probability. AFP has 700+ entries. The bottleneck is never the kernel — it's *who writes it and how to find it automatically*.

**Layer 2 — pure Isabelle automation (sledgehammer + heuristic tactics, no LLM)**:

| Dataset | sledgehammer alone | Note |
|---|---|---|
| GSM8K-class elementary arithmetic | 95-100% | linear arithmetic, finite-variable algebra |
| MATH-500 (overall) | 40-60% (est.) | mostly HS algebra; combinatorics/NT sparse |
| **miniF2F-test** (AMC/AIME/IMO) | **9.9%** (DSP paper) | near-total wipeout |
| AIME overall | ~15-25% | easy items pass; inductive/constructive fail |
| Olympiad / IMO | <10% | needs construction/geometry, sledgehammer can't |

AIME's root failures: induction (must construct induction structure), Diophantine NT tricks, combinatorial enumeration (search explosion), geometry (coordinate + trig).

**Layer 3 — LLM-augmented autoformalization (FoVer's path, and ours)**: LLM translates NL → thy, then sledgehammer/heuristics find the proof (DSP mode: Draft NL solution → Sketch NL→thy → Prove tactic-fill).

| Dataset | DSP (Jiang 2023, GPT-4 base) | Note |
|---|---|---|
| miniF2F-test | **47.6%** (vs 9.9% sledgehammer-alone) | LLM-augmented is 4-5× |
| AIME (in miniF2F) | ~30-40% | far from saturated |
| MATH-500 | est. 60-70% | easier than miniF2F |

**Our stack = Layer 3 equivalent**: a vllm judge (Qwen3.6-35B-A3B) is served, and training calls it via `OPENAI_BASE_URL` to translate NL → formal. The Isabelle integration just swaps the Z3 backend for the Isabelle backend (production script `bash_scripts/fv_step_gdpo_math.sh`); the translator stays 35B-A3B (or larger).

**Layer 4 — dedicated prover models (SOTA, mostly Lean)**:

| Model | Framework | miniF2F-test | Note |
|---|---|---|---|
| DeepSeek-Prover-V1.5 (2024) | Lean | ~55% | RL on DeepSeek-Math |
| DeepSeek-Prover-V2 (2025) | Lean | **~70.8%** | RL + subgoal decomposition |
| GoedelProver (2024) | Lean | ~64% | large-scale expert iteration |
| Lean Copilot / LeanDojo (2023) | Lean | ~50% | retrieval-augmented |
| Strongest Isabelle (DSP + retrieval) | Isabelle | 50-55% | Isabelle lags Lean in LLM-prover work |

Practical meaning: Layer 3 caps AIME at ~30-40%; Layer 4 reaches 50-70% but is **all Lean**. Going Layer 4 on Isabelle requires either fine-tuning an Isabelle-specific prover (huge effort) or waiting for a community Isabelle DeepSeek-Prover. **First phase does not do Layer 4.**

**Overall expectation (under the Layer-3 stack)**:

| Target dataset | Expected step-level coverage | Note |
|---|---|---|
| GSM8K | 95%+ | FoVer-verified, we reproduce |
| **MATH-500** | **40-50%** | primary PoC target |
| AIME | 10-20% | sparse but non-zero |
| Olympiad | <10% | basically unusable, Layer-4 later |

---

### 3. FoVer's actual method (verified against first-hand code)

Key files read under `src/dataset_creation/base_dataset_specific/isabelle/`: `error_detection/error_detection.py` (main loop), `proof_checker/proof_checker.py` + `ntptutorial/dsp_utils.py` (PISA Checker), `error_detection/preprocessing.py` (sorry-trick injection), `informal_to_formal/get_few_shot_prompt.py` (autoformalization prompt), `formal_proof_generation_few_shot_examples/correct_proofs/*.thy` (real few-shot examples).

#### 3.1 Engineering stack

| Item | FoVer | Adapted to us |
|---|---|---|
| Isabelle version | **2022** | 2025 (installed). sorry/oops/quick_and_dirty/tactic syntax unchanged 2022→2025 |
| Server | **PISA** (JVM RPC), Python client `pisa_client` | See §5 selection |
| Concurrency | **40 parallel Isabelle processes** (paper App. E.4) via 8 physical copies × 5 PISA servers each (`isabelle_copy_{0..7}`, 35 GB each) | See §5 selection |
| Carrier thy | reuses `Interactive.thy`, imports HOL Complex_Main + `Sum_of_Squares`/`Vieta`/`Computational_Algebra`/`Number_Theory` (needs AFP) | PoC starts with only HOL Complex_Main; coverage measured empirically |
| AFP heaps | prebuilt tarball from archive.org | Isabelle 2025 has none prebuilt: build `HOL-Library` etc. or import only HOL Complex_Main. MATH-500 mostly needs no AFP; AIME NT does |
| Training data | **GSM8K + MetaMathQA + Big-Math** (all GSM8K-level, "to simplify the verification pipeline") | Our target is MATH-500 (FoVer never touched it) |

#### 3.2 PISA long-connection mechanism

`dsp_utils.py:Checker` is the core client. Flow: `env.initialise()` boots the PISA server (JVM + Poly/ML), loads `Interactive.thy` and resolves imports (one-time ~10s); then `env.post("<parse text> ...")` parses the theorem into a step list; then each step runs `env.step_to_top_level_state(action=step, tls_name=..., new_name=...)`.

Key design: Isabelle is internally a stateful "top-level state" + step model (PIDE); PISA exposes this as RPC. Each step is ~100-500ms (`by simp`-class) or ~3s+ (sledgehammer). The server process is long-lived, the session reused across steps. Note: PISA's parser has a `sorry` bug, so FoVer replaces `sorry`/`sledgehammer` with `using assms oops` at parse time and swaps back after. **For us this hack is unnecessary** — with `isabelle client` (option B) native `sorry` works directly.

#### 3.3 Sorry trick (confirmed core method)

`preprocessing.py:replace_step_lemma_with_sorry` proves the sorry trick is central. `generate_theorems_for_each_step` produces N variants for an N-step proof: for each `keep_step`, every other step's `by tactic` is replaced with `sorry`, keeping only `keep_step` really proved. **N-step proof → N PISA `check()` calls**, each verifying one step.

#### 3.4 Strict proof-format constraint

`preprocessing.py:is_proof_valid_format` enforces exactly 2 lines per step: proposition line + tactic line.

```
proof -
have "..."                    ← step 1 line 1 (proposition)
    using assms by simp       ← step 1 line 2 (tactic)
then have "..."               ← step 2 line 1 (then = implicit ref to previous)
    using assms by simp
...
thus ?thesis                  ← second-to-last
    using assms by simp
qed
```

#### 3.5 Real few-shot example (gsm8k_train_03332.thy)

```isabelle
theorem example:
    assumes "(Nancy_steps::nat) = 3 * (Jason_steps::nat)"
        and "(Nancy_steps::nat) + (Jason_steps::nat) = 32"
    shows "(Jason_steps::nat) = 8"
proof -
    have "Nancy_steps = 3 * Jason_steps"       using assms by simp
    then have "Nancy_steps + Jason_steps = 32" using assms by simp
    then have "3 * Jason_steps + Jason_steps = 32" using assms by simp
    then have "4 * Jason_steps = 32"           using assms by simp
    then have "Jason_steps = 32 div 4"         using assms by simp
    then have "Jason_steps = 8"                using assms by simp
    thus ?thesis                               using assms by simp
qed
```

The uniform `using assms` prefix lets Isabelle pick which assumptions to use — equivalent to "assume all givens + prior steps correct, can this step be derived". This maps exactly onto the Z3 `fol_cumulative_mode=step` semantics.

#### 3.6 Pre-check optimization

`error_detection.py`: (1) **all_sorry** version — all steps `sorry`, one PISA run; failure = whole-proof syntax error → skip the proof. (2) **all_sledgehammer** version — all steps sledgehammer; success = whole proof auto-provable → label all steps True, skip step-level. For us the all-sorry syntax filter is retained (see §8 structural pre-checks); the all-sledgehammer shortcut is not used.

#### 3.7 Heuristic tactic list

`dsp_utils.py:_run_sledgehammer` tries 11 heuristics in order, taking the first that closes the goal, before really invoking sledgehammer:

```
by auto, by simp, by blast, by fastforce, by force,
by eval, by presburger, by sos, by arith, by linarith,
by (auto simp: field_simps)
```

This is the core fallback of the Layer-3 stack. Our shipped verifier evolved this into an ALTERNATION + EVAL_RESCUE tactic set (`verl/utils/isabelle_utils/tactics.py`); if all fail we fail-closed to 0 (we do not invoke real sledgehammer on the primary path).

#### 3.8 Autoformalization pipeline (two-step)

`informal_to_formal/generate_statement_and_proof.py` default: **Statement conversion** = Llama-3.3-70B (NL question → `theorem ... assumes ... shows`); **Proof conversion** = Qwen3-32B (NL solution → `proof - have ... by ... qed`); **Base model** (raw NL solution) = Llama-3.1-8B. The statement prompt insists on faithfulness ("final answer can be wrong, but the formal statement must be faithful and must not correct mistakes"; "shows" formatted as `variable = number`). Two-step is higher quality but ×2 calls; our Isabelle path starts single-step (see §7).

---

### 4. Sorry trick × cumulative-mode precise semantics

verl's `fol_cumulative_mode` (alias `verify_cumulative_mode`) has three modes (`verl/utils/reward_score/formal_verify.py:_normalize_cumulative_mode`).

#### 4.1 Why sorry is needed — and the sorry-free alternative

In Isar every `have` must be discharged; there is no "declare a fact but don't prove it". To make prior conclusions available to step K's proof obligation without really proving them, two paths:

- **Path A (sorry trick, FoVer)**: prior `have`s discharged with `sorry` (the only "accept without proof" mechanism, needs `quick_and_dirty`). Per-step isolation **inside one theorem** — sorry is mandatory here.
- **Path B (assumes-lifting, no sorry)**: instead of rewriting inside one theorem, generate an **independent mini-theorem** for step K where prior conclusions are lifted into `assumes` and `shows` is step K's conclusion:

  ```isabelle
  theorem step3_check:
    assumes a1: "..."                            (* original givens *)
        and prev1: "betty_savings = 50"          (* prior conclusion as premise *)
        and prev2: "grandparent_contribution = 30"
    shows "parent_contribution + grandparent_contribution = 45"
    using assms by tactic
  ```

  This format corresponds one-to-one with Z3's entailment check (premises ∧ ¬conclusion tested UNSAT) and needs no sorry / quick_and_dirty.

FoVer picks Path A because its translation unit is the whole theorem, so tactic-line → `sorry` is the minimal in-place string edit. Our choice: PoC (whole-solution translation) follows Path A; production (per-step XML translation) can use either, and Path B matches the Z3 construction (cache keys, debug output align).

**Shared ex-falso caveat**: both paths (and the Z3 path itself) share one property — if prior conclusions contradict the givens, anything follows and step K passes vacuously. This is intrinsic to cumulative semantics, not introduced by Isabelle. It is not rare: in fully-numeric problem types (GSM8K / most MATH word problems) *any wrong intermediate conclusion automatically contradicts the givens*. A wrong step scores 0 (its own premises are still consistent); steps *after* it pass vacuously with 1 (step mode: all later steps; dependency_graph mode: only downstream steps on the dependency graph). Under online RL this is exploitable, so **premise-consistency pre-check is Phase-2 standard**: Z3 side `solver.check()` on premises alone (SAT to continue); Isabelle side try `have False using assms prev_steps by tactic` — if False derivable, premises contradictory → that step fail-closes to 0 + a `premises_inconsistent` debug flag, no vacuous positive reward. Note: the shipped default is `current_only` (premises = original givens, naturally consistent), so current Z3 experiments are largely unaffected; this check is required config when switching to step / dependency_graph.

#### 4.2 Precise construction of the three modes

Easy pitfall: `assms` in Isar strictly means the theorem-head `assumes` set — **sorry-placeholder `have` facts are NOT in assms**. Prior facts enter step K's obligation only via `then` chaining (passes only the immediate previous step) or named reference (`have stepN:` then explicit `using step1 step2 ...`).

| Mode | Z3 behavior | Isabelle construction (Path A, named-fact style) |
|---|---|---|
| `current_only` | only original givens verify step K | prior `have`s all **deleted** (not sorry); step K is a plain `have` (drop `then`), `using assms by tactic` |
| `step` | givens + **all** prior conclusions | prior all named + sorry (`have stepN: "..." sorry`); step K `using assms step1 ... step(K-1) by tactic` |
| `dependency_graph` | givens + prior conclusions step K **actually references** | prior named + sorry; step K `using assms step_i step_j` lists only dependencies |

**FoVer's literal fourth construction** (`then have` chain + no naming + uniform `using assms`) gives step K "givens + step K−1 conclusion" — **between `current_only` and `step`**, not strictly equal to Z3 `step`. Fine for GSM8K linear chains; differs for MATH-500 multi-branch reasoning. Production alignment: `dependency_graph` needs to know which prior steps step K references — the policy model's XML `<premise>` tags declare dependencies explicitly, mapped directly to the `using` list; both paths parse the same source, isomorphic.

#### 4.3 Whole-proof vs per-step relation

Each step's proof obligation is **identical** in both modes (the tactic sees the same premise set; whether prior conclusions are proved or sorry-axiomatized is irrelevant to the tactic), so **all-steps-pass ⟺ whole-proof-passes**. The difference is entirely on failed proofs: whole-proof mode halts at the first failing step (later steps get no label — prefix truncation); per-step mode gives every step a label regardless (full-length reward vector). RL most needs the signal at failures — that is the entire reason the sorry trick exists. The cost is ×N verification calls, which is why the server pool (amortizing ~8s cold start) is mandatory.

---

### 5. Server selection

#### 5.1 Options

- **Option A — install PISA (FoVer's choice)**: clone Portal-to-ISAbelle, sbt + Java 11, `sbt assembly`, run `PisaOneStageServer`. Pro: FoVer-validated, controllable concurrency. Con: heavy deps (sbt+JVM+Scala), 10+ min build, designed for Isabelle 2022 (2022→2025 PIDE compat untested), 35 GB × N disk copies, maintenance stalled (last commit ~2yr). Effort: medium (1-2 days).
- **Option B — native `isabelle client` (recommended)**: Isabelle 2025 ships `isabelle server` + `isabelle client` over native PIDE (JSON over TCP). Write a Python socket wrapper aligning to the PISA interface. Pro: 0 extra deps; latest PIDE (no 2022→2025 compat risk); official support; multi-session concurrency built in. Con: must write the Python wrapper (PISA `dsp_utils.py` ~200 lines as reference). Effort: medium (1-3 days).
- **Option C — subprocess `isabelle process` per call**: simplest but **5-10s cold start per call** (loading HOL heap, +15s for AFP) — unusable for online RL (128 steps/batch = 25 min pure verification). Offline PoC only.

#### 5.2 Decision + Option-B result (shipped)

**Main line: Option B (`isabelle client` + self-written Python wrapper)** — no PISA 2022→2025 compat risk, no sbt/JVM heavy deps (container-friendly), controllable effort. Fall back to A if B's protocol proves too hard.

Shipped as `verl/utils/isabelle_utils/server_pool.py` (`IsabelleWorker` + `IsabelleServerPool`, pure-Python socket, ~370 lines). **Measured (73 single-step variants, 4 workers, dt2)**: median verify 9.8s (subprocess) → **0.41s (24×)**; p95 10.1s → 0.87s; 73-variant wall ~180s → **23.2s**; judgment consistency 73/73. Startup cost: server ~1s + session ~10s, one-time per worker. RL throughput estimate: 768 verify/batch ÷ 16 workers × 0.4s ≈ **19s/batch** (32 workers ≈ 10s).

**Six protocol issues found and handled (all documented in the code):**

1. `headless_consolidate_delay` default 2.0s added ~2.1s per `use_theories` → set 0.05 (2.1s → 0.4s).
2. Concurrent worker startup shares an SQLite registry (NFS) → `SQLITE_BUSY`; serialize server spawn (registry writes are fast), keep heavy session loads parallel.
3. Same-named theory node re-submitted after purge → server document model appends new content after the old blob (error offset past EOF) → use a **unique theory name per request**; a resident Prelude is never purged.
4. Resident Prelude theory: V-theories only `imports Prelude`, so the four-library import resolution is paid once per worker, not per verification.
5. `purge_theories` occasionally blocks 60-70s (hits the server's background cleanup cycle) → purge on a second management connection + background thread; the verify hot path does not wait on it.
6. A theory with a corrupt `theorem` header never consolidates → `use_theories` waits forever (default watchdog 600s) → set `headless_watchdog_timeout=15`; the server kills the bad theory within 15s; an outer 60s deadline + worker auto-restart is the last-resort guard.

---

### 6. IsabelleEngine class design

Mirrors `verl/utils/fol_utils/engine.py:FOLEngine` structure with key differences.

#### 6.1 Class structure (design sketch — shipped `verl/utils/isabelle_utils/engine.py` refined this with the coding-form translator + `tactics.py`)

```python
# verl/utils/isabelle_utils/engine.py

class IsabelleConfig:
    server_mode: str = "client"       # "client" / "pisa" / "subprocess"
    num_workers: int = 8              # server pool size
    isabelle_home: str = os.environ.get("ISABELLE_HOME")
    session_name: str = "HOL"         # or HOL-Library / custom-with-AFP
    quick_and_dirty: bool = True
    cumulative_mode: str = "step"     # maps to Z3's three modes
    tactic_fallback: list[str] = [...]  # the §3.7 heuristics
    timeout_per_step: float = 5.0
    api_config: dict = field(default_factory=dict)

class IsabelleEngine:
    """Mirror of FOLEngine but for Isabelle theorem proofs."""
    def __init__(self, config, server_pool): ...
    def preprocess(self, problem_nl) -> str:            # NL → `theorem ... shows`
    def verify_step(self, formal_statement, step_history, current_step_nl, debug_info=None) -> float:
        formal_step = self._translate_step(current_step_nl, formal_statement, step_history)
        thy = self._build_sorry_thy(formal_statement, step_history, formal_step, self.config.cumulative_mode)
        return 1.0 if self.server_pool.check(thy).success else 0.0
```

#### 6.2 Key differences vs FOLEngine

| Dimension | FOLEngine (Z3) | IsabelleEngine |
|---|---|---|
| Execution base | `subprocess.run([python, -c, z3_code])` | server pool RPC |
| Translation output | Z3 Python code (entailment script) | thy fragment (statement + sorry-trick proof) |
| Fail-closed | AST syntax / sort mismatch / unknown identifier | Isabelle error-code parse (grep `error:`/`Step error` from obs) |
| Repair | LLM fixes Z3 expression (`_repair_implication_expressions`) | LLM fixes tactic choice or missing fixes / re-translate |
| Cache key | `(declarations, step_text, config)` | `(statement, step_history_hash, current_step, config)` |

#### 6.3 Reused verl framework

- **Cache**: the `fol.py` in-memory + disk cache is verifier-agnostic, reused directly (cache key adds `task_type`).
- **LLM calls**: `verl/utils/fol_utils/common.py:call_llm` / `call_llm_structured` reused.
- **Process pool**: `common.py:_get_mp_pool` → `IsabelleServerPool` (daemon lifecycle).
- **`compute_step_reward_fol` routing** (`fol.py`): dispatches by `TaskType` to FOLEngine or IsabelleEngine (shipped as a lazy singleton `_get_isabelle_engine`, one engine per process, gated on `task_type == "math"`).

#### 6.4 IsabelleServerPool design

```python
class IsabelleServerPool:
    """Pool of long-running isabelle server processes (option B)."""
    def __init__(self, num_workers, isabelle_home, session):
        self.workers = [IsabelleWorker(port=9000+i, ...).start() for i in range(num_workers)]
    def check(self, thy) -> CheckResult:
        return self.queue.get_idle_worker().run_check(thy)
    def shutdown(self):
        for w in self.workers: w.terminate()
```

Each Ray worker initializes one pool at startup.

---

### 7. Translator prompt

#### 7.1 Single-step vs two-step

FoVer uses two-step (NL→statement, NL→proof). Our choice: the current Z3 path is single-step (engineering simplicity); two-step is higher quality (a wrong statement fails all step verifications) but ×2 calls. Recommendation: **start single-step, split into two-step only if PoC translation success < 50%**. In the initial single-step design the **LLM only writes the proposition, not the tactic** — it emits `have "..." sledgehammer` and the backend substitutes the heuristic tactics (§3.7). Rationale: FoVer showed 11 heuristics suffice on GSM8K; letting the LLM pick tactics adds a failure dimension; backend auto-fallback keeps the reward signal focused on "is the proposition correct", homogeneous with Z3 entailment.

Type inference matters: GSM8K is almost all `::nat`; MATH-500 has heavy `::int` (NT) / `::real` (algebra) — the prompt must state which. NL comments are kept in `(* ... *)` for debugging. Native `sorry` is used (option B avoids the PISA parser bug). Few-shot examples borrowed from FoVer `correct_proofs/`, supplemented with our own algebra/NT/geometry examples.

#### 7.2 Coding-form translation (shipped production form)

Motivation (judge = Qwen3.6-35B-A3B, chosen partly for coding: SWE-bench Verified 73.4%): convert the translation task into a coding task to raise success. The judge outputs a **constrained Python/boolean expression** (subset: `+ - * / ** sqrt fact choose mod` + comparison/logic connectives); a local `ast` walker validates the AST (whitelist) and transpiles to an Isabelle term. Semantic gaps are pinned in the subset: Python `/` (real) vs Isabelle int `div`; `**` vs `^`; negative `mod` differs; `Rational` vs float. Payoff: syntactic translation failures approach zero (the model's Python training data dwarfs its Isabelle data — LaTeX residue / missing `*` / `f(x)` / int-nat errors nearly vanish); all mechanical checks (type annotation, guard window, transcription completeness, empty proposition) move onto the AST, more reliable than regex; fewer retries. This became the shipped production form: `verl/utils/isabelle_utils/pyexpr.py` (AST whitelist + type-aware transpile) + the translator prompts `prompts/translate_givens.txt` and `prompts/translate_steps.txt`. The earlier Isabelle-direct / sledgehammer-placeholder prompt (§7.1) was superseded and retired. It does **not** rescue semantic-layer failures (figures / combinatorial structure / word-sum conditions) — those remain few-shot / SFT territory.

---

### 8. Integration points (code locations)

| Change | File | Action |
|---|---|---|
| `TaskType.MATH` enum | `verl/utils/fol_utils/engine.py:1781` `TaskType` | shipped: `MATH = "math"` (Isabelle path) + `MATH_Z3 = "math_z3"` (deprecated Z3 math path) |
| IsabelleEngine | `verl/utils/isabelle_utils/engine.py` | see §6.1 |
| IsabelleServerPool | `verl/utils/isabelle_utils/server_pool.py` | see §6.4 |
| Routing | `verl/utils/reward_score/formal_verify.py` `compute_step_reward_fol` | dispatch on `fol_task_type` (alias `verify_task_type`) — `"math"` → IsabelleEngine, `"math_z3"`/`"logic"` → FOLEngine |
| Translator prompts | `verl/utils/isabelle_utils/prompts/{translate_givens.txt, translate_steps.txt}` | see §7.2 |
| Cache key includes task_type | `fol.py:_build_verify_cache_key` | add `task_type` to avoid Z3 / Isabelle cache cross-contamination |
| Training script | `bash_scripts/fv_step_gdpo_math.sh` (+ smoke `bash_scripts/sanity_check_fv_math.sh`) | set `+fol_task_type=math` |
| Disk cache version bump | `fol.py` `_FOL_VERIFY_DISK_CACHE_VERSION` | bump after logic changes |
| Side-condition synthesizer | `engine.py` internal component | see §8.1 |
| Structural pre-check | `engine.py` internal component | see §8.2 |

#### 8.1 Side-condition synthesizer

Motivation (n=100 measured): many residual failures are not weak tactics but **missing implicit boundary conditions** — Isabelle's `field_simps` etc. require "denominator ≠ 0" as a prior fact, which NL solutions never write. Example (ω³=1 ⊢ 1/(1+ω)+1/(1+ω²)=1): 14-way tactic parallel fails in 0.5s; sledgehammer "no proof found" in 14s; but adding `have nz: "1+ω ≠ 0"` then `simp add: field_simps` **passes in 0.37s**, and the aux fact itself is provable (`by (auto simp: add_eq_0_iff)`, no sorry freebie).

Design (aligned with Z3-path autofill): mechanically scan step propositions for `/`, `inverse`, `sqrt`, `ln`, `powr` patterns (no LLM) and synthesize the boundary fact (`b ≠ 0`, `x > 0`, `e ≥ 0`). Conservatism: **the synthesized fact must be proved on the spot** (fast chain + targeted lemma pack `add_eq_0_iff`/`divide_eq_0_iff` + sledgehammer finish on the small isolated goal); if unprovable → step fail-closes + `side_condition_unproven`. Safety: a true division-by-zero can't prove its `≠ 0`, so that step scores 0 (no leniency). Cost: only triggers on steps containing the operator, ~0.4s/condition, deduped across steps sharing a denominator. Metrics: `isa_side_condition_{synthesized,proven,unproven}`.

#### 8.2 Structural pre-check of translation output

Two translation-defect classes to catch mechanically beyond the all-sorry pre-check:

1. **Answer leakage**: translator writes `answer = 12` into both assumes and shows → vacuous theorem. Check: shows's target equality (after var renaming) must not appear in assumes. Mirrors the Z3 path's `FAILED_LEAKED_CONCLUSION`.
2. **Free-variable assertion**: `have "(half_base::int) = 3"` where `half_base` appears from nowhere (not in assumes, not defined) — asserting a value for a free variable is unprovable and is a translation defect, not a solution error. Check: each step proposition's free variables ⊆ assumes variables ∪ variables introduced by prior steps; violation → re-translate feedback.

---

### 9. Measurement history (condensed)

The offline PoC (`scripts/isabelle_poc_math500/`) ran ~18 dated measurement rounds (n=18 → n=100 → six-dataset suite, pipeline v1 → v5.8). **Full dated logs are superseded; see memory `project_isabelle_v58_analysis.md` and the archived integration plan for the blow-by-blow.** Durable frozen results:

- **v5.8 six-dataset baseline** (Qwen3-4B policy, thinking off, production XML prompt, coding-form translation; translation × verification product on the answer-correct subset): **GSM8K 91.3 / Minerva 65.4 / MATH-500 53.5 / OlympiadBench 48.3 / AMC 41.0 / AIME 36.8** (AIME via judge-generated n=57 trusted sample). Only GSM8K clears the 80% target; the rest are coverage-limited, not soundness-limited.
- **Soundness holds**: step-level corrupt-injection false-positive rate **≤ 5%** on GSM8K and MATH (v5.8: 4.4–4.7% on the correct-answer subset) — fail-closed confirmed. Reached only after four soundness fixes: pool task-id result-matching (a `wait_task` mis-binding produced both FP and FN), watchdog false-success ("ok" only means "no error yet", not "consolidated" — any check >15s was silently scored as proved), consistency-check tactic strength ≥ target tactic (ex-falso asymmetry), and removal of the bogus sqrt-metis attempt.
- **Verifier extension gives ~0% rescue on Case B**: upgrading SESSION HOL-Number_Theory → HOL-Analysis and adding strong tactics (`force` / `algebra` / `smt (verit)` / …) rescued **0/100** sampled Case-B failure steps. Case-B failures are structural (conjunction `simp` non-splitting, sorry-axiom non-chaining, `using assms` not triggering rewrite), not tactic-strength-limited.
- **Translator multiplicative bridging (v5.9) nets only +1.25%** (5/400 uniquely-improved problems) and is buried in **±7% (~±25/400) judge translation stochasticity** — the three-run improvement-set intersection was only 10 problems. Not adopted (its conjunction regressions offset the gain); Python-layer conjunction splitting kept as a harmless no-op prep.
- **35B judge thinking-on ≈ 4B parity**: the judge's own AIME solutions (12.3 steps, case-split/set style) are harder to formalize; judge-AIME translation × verification ≈ **34.8%**, matching the 4B policy — i.e. AIME's true position (~35-40%) is generator-independent.
- **Sledgehammer slow-fallback rescues ~10%** (8/79 sampled v5.8 failure steps, 45s timeout + sound replay; MATH 19.2%, others 0-12%) — a robustness add-on after EVAL_RESCUE, not a primary lever.

Failure-step taxonomy (856 failed steps across six datasets): **Case A** (isolated symbol — a symbol in the proposition was never `lhs == rhs`-bound in the premise chain; "model skips + judge didn't bridge") 41.7%; **Case B** (all symbols bound but the tactic didn't compute it) 58.3%. OlympiadBench is the only Case-A-dominant dataset (53.1%). Geometry keywords hit only 0.6% of propositions (geometry gets coordinatized during translation), so AFP-geometry priority was down-ranked.

**Conclusion**: offline improvement levers are essentially exhausted (verifier extension 0%, translator bridging net ~1%, sledgehammer 10% but too slow). Further gains must come from RL training pushing the policy to emit complete derivation steps. GSM8K is training-ready; MATH/Minerva signal is usable but sparse.

---

### 10. Key risks

| Risk | Severity | Mitigation |
|---|---|---|
| MATH-500 autoformalization success too low | High | Caught by PoC (all-sorry 70.6%, passed). If it drops <50%: larger translator (Llama-3.3-70B / Qwen2.5-72B) or split two-step |
| MATH-500 step tactic pass rate too low | High | Add `by algebra` / `by (auto simp: algebra_simps)` / composite tactics. Worst case accept 30-40% — sparse but trainable |
| AIME / Olympiad hard limit | High (irreducible) | AIME's induction / geometry construction / NT tricks defeat heuristics + sledgehammer. **Accept**: no step-level Isabelle for AIME in first phase; step scores fail-closed to 0 or emit a neutral empty step-score list (reward-manager switch); outcome term still applies per weight |
| Isabelle 2025 / `isabelle client` protocol drift | Medium | Entry points: `$ISABELLE_HOME/src/Pure/General/json.scala` + `$ISABELLE_HOME/lib/Tools/server`; test in the wrapper PoC |
| `isabelle client` Python wrapper effort | Medium | PISA `dsp_utils.py` ~200 lines as skeleton; +1-2 days |
| Server pool latency over budget | Medium | Scale workers (16 → 32), strengthen cache, worst-case async fallback |
| Z3 / Isabelle mixed step-reward magnitude imbalance | Medium | Both are binary 0/1, but differing failure modes skew the math distribution — monitor wandb |
| AFP libs compile (Sum_of_Squares / Vieta / Number_Theory) | Medium | GSM8K needs none; MATH-500 mostly HOL+Complex_Main; only AIME NT needs AFP (and AIME goes via outcome reward) |
| Disk cache cross-contamination (Z3 ↔ Isabelle) | Medium | Add `task_type` to cache key; bump `_FOL_VERIFY_DISK_CACHE_VERSION` |
| `quick_and_dirty` lets a truly wrong proof pass silently | Low | sorry is only a step-level placeholder, never in the final reward; a wrong step's `by tactic` still fails |

---

### 11. Toward AIME / Olympiad (Layer-4 roadmap)

To really attack AIME / Olympiad, Layer 3 must upgrade to Layer 4 (dedicated prover models) — an order-of-magnitude more engineering, **not in the first phase**.

**Candidate approaches**: **X1** — plug in a DeepSeek-Prover-V2-class model (but all Lean; needs a Lean↔Isabelle layer, which is immature, or a large rewrite to a Lean path). **X2** — fine-tune an Isabelle-specific prover (data: FoVer-40K + AFP proofs + own RL rollouts; base: Qwen2.5-Math-7B / Llama-3.1-8B + RL on Isabelle reward; enormous effort but naturally compatible with our online-RL framework — the prover *is* an upgraded translator). **X3** — wait for a community Isabelle DeepSeek-Prover.

**Layer-4 ceilings**: DeepSeek-Prover-V1.5 ~55%, V2 ~70.8%, GoedelProver ~64%, Lean Copilot ~50% (all Lean); strongest Isabelle (DSP + retrieval) 50-55%. Layer 4 pushes AIME from ~30% to ~70%, but Isabelle lags Lean by ~a year.

**Decision timing**: after first phase (Phase 1-4) — if MATH-500 training is strong, keep pushing toward AIME; if AIME trains acceptably on the outcome term alone, full Layer-4 config is not urgent; otherwise bring Layer-4 config + translator SFT (LoRA on the same judge base, deployed as base+adapter on one vLLM instance so both reward paths stay consistent) forward, then consider X1/X2.

---

### 12. References

- Paper: [arXiv 2505.15960](https://arxiv.org/abs/2505.15960) — Kamoi et al., "Efficient PRM Training Data Synthesis via Formal Verification", ACL 2026 Findings
- Repo: [psunlpgroup/FoVer](https://github.com/psunlpgroup/FoVer) (local: `E:/AMPR/FoVer/`)
- PISA: [Portal-to-ISAbelle](https://github.com/albertqjiang/Portal-to-ISAbelle) (Albert Qjiang)
- DSP (Draft-Sketch-Prove): Jiang et al. 2023, "Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs" (PISA Checker origin)
- ntptutorial: [wellecks/ntptutorial](https://github.com/wellecks/ntptutorial/) — Sean Welleck's Neural Theorem Proving tutorial
- miniF2F: Polu et al. 2022, "Formal Mathematics Statement Curriculum Learning"
- DeepSeek-Prover-V1.5: Xin et al. 2024, [arXiv 2405.14333](https://arxiv.org/abs/2405.14333)
- DeepSeek-Prover-V2: Ren et al. 2025, [arXiv 2504.21801](https://arxiv.org/abs/2504.21801)
- HF PRM models: [ryokamoi/Llama-3.1-8B-FoVer-PRM-2026](https://huggingface.co/ryokamoi/Llama-3.1-8B-FoVer-PRM-2026), [ryokamoi/Qwen-2.5-7B-FoVer-PRM-2026](https://huggingface.co/ryokamoi/Qwen-2.5-7B-FoVer-PRM-2026)
- Isabelle 2025 install (datatech shared disk): `/2022533109/zhouchuyan/isabelle/Isabelle2025/`; enable via `env.sh`; sorry-trick PoC: `poc/{ROOT,Wallet.thy}`
- Deprecated in-tree analysis memory: `project_isabelle_v58_analysis.md` (per-run measurement detail)

---

## See also

- **Implementation task log & operational quirks:** `docs/isabelle-phase2-2-plan.md` (source: `E:/AMPR/isabelle-phase2-2-plan.md`).
- **Session plan (D1–D7 checklist):** `~/.claude/plans/gentle-leaping-seal.md`.
- **Measurement detail (per-run):** memory `project_isabelle_v58_analysis.md`; build times: memory `reference_isabelle_build_times.md`.
- **Z3/FOL sibling path:** `verl/utils/fol_utils/` + the top-level `README.md` (`fol_task_type=logic` and the deprecated `fol_task_type=math_z3`).
- **AMPR entry point** `isabelle-integration.md` is a symlink back to this file (one physical file, not a copy).
