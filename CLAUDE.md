AGENTS.md

# Project: verl-fol (FOL Step GDPO for Logical Reasoning)

## Critical Rules

1. **WANDB_ENTITY=verl-fol** — Every training launch MUST explicitly set this. Never let it default to personal entity "itsuitsuki".
2. **W&B project**: `verl-fol-2`
3. **Local vLLM judge only** — Always use local GPU judge (Qwen3.6-35B-A3B), never remote API (SiliconFlow etc.)
4. **Never guess commands** — Check logs (`tmux capture-pane`, log files, shell history) or ask the user. Never fabricate startup commands.
5. **Read logs before restart** — Before restarting any training, read the log file for the exact command.
6. **Source conda explicitly** — nohup/background won't trigger direnv; always source conda.sh explicitly in launch commands.

## Cluster Topology

### DataTech (ShanghaiTech, 8×H20 per node)

| Node | SSH | Status | Role |
|------|-----|--------|------|
| datatech-1 | `ssh datatech-1` | Active | 7B training (GPU 0-3) + judge (GPU 4-5) |
| datatech-2 | `ssh datatech-2` | **BROKEN** — never use | GPU partition bug crashes PyTorch |
| datatech-3 | `ssh datatech-3` | Active | GRPO baselines |

- Shared filesystem: `/2022533109/zhouchuyan/verl/`
- Conda: `source /2022533109/liubushi/miniconda3/etc/profile.d/conda.sh && conda activate verl`
- Models (local): `/root/run/models/Qwen2.5-1.5B-Instruct`, `/root/run/models/Qwen2.5-7B-Instruct`, `/root/run/models/Qwen3.6-35B-A3B`
- Data: `/2022533109/zhouchuyan/verl/data/{logiqa2k_prompt_v2,reclor_prompt_v2,gsm8k}/`
- GSM8K has `test.parquet` (no `validation.parquet`)

### Paratera (Slurm: A800/H100/H200)

- SSH: `ssh paratera-46-ali` (lands on ln01 or ln02; only ln01 has internet)
- Conda: `source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh && conda activate verl`
- Code/data: `~/run/work/verl/`
- Models: `~/run/models/{Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct,Qwen3.6-35B-A3B}`
- srun format: `srun -G2 -p gpu_h200 -t 480 bash script.sh` (no --mem, no --gres)
- **LD_PRELOAD required**: `export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` — system libstdc++ is too old for vLLM
- If judge startup times out: fallback to `srun -G2 --pty bash`, start judge interactively, then training
- Storage: `~/run` (large), `~/` only 1GB

## Key Scripts

- `bash_scripts/grpo_outcome_only_1gpu.sh` — 1.5B GRPO outcome-only baseline (1 GPU, reward_manager=dapo)
  - Env vars: `DATA_NAME`, `DATA_DIR`, `MODEL_PATH`
  - Accepts `$@` for extra Hydra overrides
- `bash_scripts/fol_step_gdpo_reclor_7b_h200.sh` — 7B Reclor on Paratera H200 (GPU 0=judge, GPU 1=training)
- `bash_scripts/fol_step_gdpo_logiqa_7b.sh` — 7B LogiQA on DataTech-1 (4 GPU)

## Judge Configuration (DataTech-1)

```bash
CUDA_VISIBLE_DEVICES=4,5 vllm serve /root/run/models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B --port 4873 \
    --max-model-len 12288 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 256
```

Runs in tmux session `judge`. Training connects via `OPENAI_BASE_URL=http://127.0.0.1:4873/v1`.

## Completed Results (1.5B baselines)

| Experiment | Metric |
|-----------|--------|
| 1.5B FOL LogiQA ckpt@1844 → LogiQA test | 47.57% |
| 1.5B FOL LogiQA ckpt@1844 → Reclor val (OOD) | 54.20% |
| 1.5B FOL GSM8K ckpt@934 → GSM8K test | 74.37% |
| 1.5B FOL Reclor ckpt@579 → Reclor val | 58.27% |
