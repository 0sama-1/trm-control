# RUNBOOK — How to run TRM-control experiments

Human-facing guide. Everything runs on the Pawsey **quantum** partition (GH200 GPUs, CUDA
12.6, Python 3.11.6). **Follow the exact template from `setup.sh` and `README.md`** — this
package is single-process; never raise `nproc_per_node` above `1`.

---

## 0. Prerequisites (done once per user)

You need slurmsy access to the Pawsey `quantum` partition with a compute allocation, and a
wandb API key. On your workstation clone the repo; on the login node configure wandb:

```bash
export WANDB_API_KEY=...   # put in ~/.bashrc or your submission script
```

---

## 1. Request an interactive node

```bash
salloc -p quantum -N 1 --gres=gpu:1 --time=01:00:00 -A <allocation>
```

Replace `<allocation>` with your project allocation (e.g. `pawsey1302-q`).

---

## 2. Bootstrap the environment (once per node / new checkout)

```bash
cd /path/to/trm_control_integration      # the repo root

module purge
module load pawsey pawseytools pawseyenv/2026.01
module load PrgEnv-nvidia
module load craype-arm-grace
module load gcc-native-mixed/12.3
module load pawseyenv/2026.01
export MODULEPATH=${MODULEPATH}:${LMOD_CUSTOM_COMPILER_GNU_12_0_PREFIX}
module load cuda/12.6
module load python/3.11.6

export PROJECT="trm-control"
export UV_CACHE_DIR="$PWD/.uv-cache"
export TMPDIR="$PWD/.tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

uv venv --python "$(which python3)"
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install gymnasium imageio "gymnasium[classic-control]" "gymnasium[other]"
uv pip install wandb weave hydra-core psutil
```

Or just run `bash setup.sh` which does the same thing (and prints a sanity line with
`python3 -V`).

> Troubleshooting: if the code fails on CUDA mismatch, ensure the loaded CUDA module version
> ≤ the GPU's (quantum GPUs use CUDA 12.7). Check `nvidia-smi`, `which nvcc`, `nvcc --version`.

---

## 3. Sanity checks

```bash
python3 pyscripts/test_gpu.py      # torch + CUDA availability + GPU name
uname -m                            # hardware architecture
nvidia-smi
```

---

## 4. Run a single experiment

The entry point is always **`main.py`** with Hydra overrides. Compose from five groups:
`arch` (RL algorithm), `model` (backbone trunk), `env`, `delay` (action staleness),
`staleness` (observation staleness), plus a `seed`. Staleness and delay are independent
dials — use them separately or together.

```bash
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=dnn env=pendulum delay=none seed=42
```

| Group | Choices | Notes |
|-------|---------|-------|
| `arch=` | `ddpg`, `td3`, `sac`, `ppo` | RL algorithm + its hyperparameters in `conf/arch/*.yaml` |
| `model=` | `dnn`, `trm`, `trm_hier6`, `trm_singlez`, `trm_carry`, `lstm` | Backbone trunk in `conf/model/*.yaml`. `dnn` = MLP baseline; `trm` = 2-latent TRM; `trm_hier6` = 6 parallel latents; `trm_singlez` = single latent; `trm_carry` = H5 persistent-`h` variant (planned); `lstm` = recurrent memory baseline (**required for H5's interpretability**, see `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §H5) |
| `env=` | `pendulum`, `cartpole` | Currently only continuous-(Box)-action envs are runnable by the actor heads: DDPG/TD3/SAC/PPO. `cartpole` is discrete and **not runnable yet** |
| `delay=` | `none`, `fixed`, `stochastic`, `time_varying` | **Action staleness** — the applied action was chosen `n` steps ago (proxy for horizon). See §5 |
| `staleness=` | `none`, `fixed`, `stochastic`, `time_varying` | **Observation staleness** — the fed observation is `n` steps old, independent of action delay. See §5 |
| `seed=` | int | reproducibility; wandb run name encodes `seed` |

Examples:

```bash
# DDPG on permanent MLP (baseline)
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=dnn env=pendulum delay=none seed=42

# DDPG on 2-latent TRM
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=trm env=pendulum delay=none seed=42

# SAC on hierarchical-6 TRM with fixed delay
torchrun --standalone --nproc_per_node=1 main.py arch=sac model=trm_hier6 env=pendulum delay=fixed seed=0

# H5: PPO on the persistent-carry trunk with one-step h backprop
torchrun --standalone --nproc_per_node=1 main.py arch=ppo model=trm_carry env=pendulum delay=none model.trm_carry.h_backprop=one_step seed=42

# H5 interpretability controls
torchrun --standalone --nproc_per_node=1 main.py arch=ppo model=lstm env=pendulum delay=none seed=42                              # recurrent memory baseline
torchrun --standalone --nproc_per_node=1 main.py arch=ppo model=trm_carry env=pendulum delay=none model.trm_carry.readout=y_on_h seed=42  # readout-placement control

> **H5 gating.** The `lstm` baseline and the y-on-h readout control are **unconditional**
> interpretability cells (run always). The Family-B sibling ablations (depth /
> deep-supervision / transfer) ship behind config but are only **scheduled** after a positive
> pendulum headline (`trm_carry ≥ lstm`); see `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §H5 and §6 decision #9.
```

`model.folder`/`video.folder` resolve to repo-root `models/` and `videos/` automatically.
Checkpoints save to `models/<env>_<arch>_<model>_<delay>_seed<seed>.pt` and auto-resume if
the file exists. Metrics stream to wandb.

> **Do not** run `--nproc_per_node` > 1. `main.py` is single-process; a higher value starts
> multiple uncoordinated copies that race on the same GPU and checkpoint.

---

## 5. Staleness / delay modes (what they mean)

Two independent environment parameters control effective horizon: **action staleness**
(delay on the applied action) and **observation staleness** (the fed observation is `n` steps
old). Use them separately or together.

| Mode | Config | Staleness channel | Behavior |
|------|--------|-------------------|----------|
| `delay=none` | `conf/delay/none.yaml` | action | Action applied immediately (delay 0) |
| `delay=fixed` | `conf/delay/fixed.yaml` | action | Constant action delay (default `1.0` step) |
| `delay=stochastic` | `conf/delay/stochastic.yaml` | action | Action delay sampled each step from a distribution (`uniform`/`normal`/`exponential`), default uniform over `[min,max]` |
| `delay=time_varying` | `conf/delay/time_varying.yaml` | action | Piecewise-constant action-delay schedule over training steps |
| `staleness=none` | `conf/staleness/none.yaml` | observation | Observation fed to the agent is the current one (k=0) |
| `staleness=fixed` | `conf/staleness/fixed.yaml` | observation | Constant observation staleness (default `3.0` steps) |
| `staleness=stochastic` | `conf/staleness/stochastic.yaml` | observation | Observation staleness sampled each step from a distribution (not yet implemented) |
| `staleness=time_varying` | `conf/staleness/time_varying.yaml` | observation | Piecewise-constant observation-staleness schedule over training steps (not yet implemented) |

Increase delay/staleness config values to lengthen the effective horizon being addressed.
Observation staleness is the direct test of the H5 "high-level latent smooths stale input"
mechanism.

**Which algorithm to use.** Per `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §5.5: use **PPO** for mechanism
hypotheses and the H1 headline; use **SAC** (or DDPG/TD3) for the heavy MuJoCo replication
(~35× cheaper per run) plus one short PPO confirm. Before mixing algos, run the parity check
in plan Stage B0 (same cell, both algos) and calibrate the H100-hour model with real
wall-clock.

---

## 6. Run an entire experiment / sweep (multi-config × multi-seed)

Interactive `main.py` runs one config×seed at a time. For the full experiment matrix, use
Slurm batch jobs. The repo currently provides the **pattern** (single-process `main.py` in a
`#SBATCH` script); a convenience **sweep controller** is part of the experiment plan — not yet
shipped.

General batch script (one job per config×seed):

```bash
#!/bin/bash
#SBATCH --partition=quantum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=<allocation>
#SBATCH --job-name=trm-control
#SBATCH --output=logs/%x_%j.out
set -e
cd /path/to/trm_control_integration
# --- module + venv bootstrap (see §2) ---
source .venv/bin/activate
torchrun --standalone --nproc_per_node=1 main.py arch=${ARCH} model=${MODEL} env=${ENV} delay=${DELAY} seed=${SEED}
```

Launch the matrix:

```bash
for seed in 0 1 2 3 4; do
  for model in dnn trm; do
    for delay in none stochastic; do
      ARCH=ppo MODEL=$model ENV=pendulum DELAY=$delay SEED=$seed \
        sbatch scripts/sweep_ppo_pendulum.sh
    done
  done
done
```

(Adapt the env-var / loop to the actual sweep script the coding-manager ships in
**`/workspace/trm/docs/plan/EXPERIMENT_PLAN.md`** / `scripts/`.)

---

## 7. Results

- **wandb** — primary store: `return`, `avg100_return`, actor/critic losses, entropy/alpha,
  `buffer_size`, `total_steps`. Run names: `<timestamp>_<env>_<algo>_<model>_<delay>_seed<seed>`.
- **Wall-clock / steps-per-sec** — must be logged per run so the H100-hour cost model in
  `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §5.5 stays calibrated.
- **`models/`** — `.pt` checkpoints (gitignored).
- **`videos/`** — recorded rollouts every `record_every` episodes (gitignored).
- **`logs/`** — Slurm stdout/stderr (add to `.gitignore`).

For an analysis-ready table across configs×seeds, use wandb's export/Pandas API — the
experiment plan schedules an analysis script to emit `docs/` markdown tables + figures.

---

## 8. Adding a new model/env/experiment

1. **New backbone**: add `conf/model/<name>.yaml` + a class in `src/models/` honoring the
   trunk contract (`out_dim` + `.forward(x)`), e.g. a `TransformerTrunk`. Exception: H5's
   `TRMCarryTrunk` is **stateful across calls** (see `AGENTS.md` §4) and needs rollout/reset
   handling. Update `README.md`, this RUNBOOK, and the plan.
2. **New env** (e.g. MuJoCo): add `conf/env/<name>.yaml` and ensure the dependency is in
   `setup.sh`. Only `Box` (continuous) action spaces are currently supported by the actor heads.
3. **New delay/staleness**: add `conf/delay/<name>.yaml` (action staleness) or
   `conf/staleness/<name>.yaml` (observation staleness) + the matching wrapper in
   `src/delays.py` / `src/staleness.py`. The two channels are independent.
4. Always run a **smoke test** on `env=pendulum` first, then launch the full matrix.

---

## 10. Stage-A shared automation backbone

This section documents the three automation scripts implemented for stage A
(`EXPERIMENT_PLAN.md` §4, stage A — "Sweep controller + eval script + analysis
script"). All three follow the HPC template (module → venv → single-process
torchrun) and are **torch-free at parse time** except where a real run requires
the trainer (which needs torch).

### 10.1 Eval — `pyscripts/evaluate.py`

Fixed-seed, exploration-off evaluation of a checkpoint producing mean/std return
over N ≥ 10 episodes plus a 95% bootstrap CI. Reuses the existing
`RLTrainer._make_env` (so delay/staleness wrappers replay exactly) and the
trainer's own `_build_agent` / `_load_checkpoint` — **no trainer or algo file is
modified**. Deterministic actions are derived from each arch's existing
deterministic forward path (`--help` succeeds without a trained checkpoint; if
no `.pt` is found, eval runs on the random-init policy with a warning, which is
what makes the smoke test runnable).

```bash
# Help (torch-free at parse time; torch only needed to actually run).
python pyscripts/evaluate.py --help

# 10 fixed-seed eval episodes of a specific checkpoint.
python pyscripts/evaluate.py --n-episodes 10 --ckpt models/Pendulum-v1_sac_trm_none_seed0.pt \
    arch=sac model=trm env=pendulum delay=none seed=0 --wandb

# Smoke test: random-init policy, 2 episodes, 5 max steps (no checkpoint needed).
python pyscripts/evaluate.py --n-episodes 2 --max-steps 5 arch=ppo model=dnn \
    env=pendulum delay=none seed=42
```

Output: a `results/eval_<env>_<arch>_<model>_<delay>_seed<n>_n<N>.json` summary
(per-episode returns, mean, std, sem, ci95, min/max, and `eval_wall_clock` /
`eval_steps_per_sec`). The `main.py --mode eval` approach was considered;
`pyscripts/evaluate.py` was chosen because it reuses the real trainer
(ground-truth env/wrapper/contract) without adding eval-mode branching to
`main.py`.

### 10.2 Sweep controller — `scripts/sweep.py`

Takes a config grid (model × env × delay × staleness × seed) and emits the exact
`torchrun --standalone --nproc_per_node=1 main.py <overrides>` commands per
`AGENTS.md` §6.3, then drives them in the chosen mode.

```bash
# Dry-run: print the torchrun commands only (torch-free, no execution).
python scripts/sweep.py --dry-run --arch "ppo sac" --model "dnn trm" \
    --env pendulum --delay "none stochastic" --seed "0 1 2 3 4"

# Local sequential execution on a node (requires torch in the venv).
python scripts/sweep.py --local --arch ppo --model "dnn trm" \
    --env pendulum --delay none --seed "0 1 2 3 4" --timeout-min 15

# Slurm submission (default): one #SBATCH job per cell, logs/ per AGENTS.md §6.4.
python scripts/sweep.py --arch sac --model "dnn trm" \
    --env hopper halfcheetah --delay none --seed "0 1 2 3 4" \
    --account pawsey1302-q --time 04:00:00
```

Extra Hydra overrides (e.g. `model.trm.H_cycles=3`) are passed via `--extra`
(repeatable) or via a YAML grid file (`--grid conf/sweeps/h1_smoke.yaml`).
`--manifest-out` writes a JSON manifest of all commands. Local runs log
per-run wall-clock; Slurm runs rely on the trainer's wandb timing (`run_wall_clock` /
`run_steps_per_sec`), as documented in `AGENTS.md` §6.5.

### 10.3 Analysis — `scripts/analyze.py`

Reads a results CSV (one row per run), computes per-condition mean/std, 95%
bootstrap CI, Cohen's d, and paired/2-sample t-tests between conditions, then
writes a markdown report + bar figure to `docs/plan/reports/`. **Torch-free**
(numpy + scipy only) — runs on the login node.

```bash
# Smoke test: synthesize a tiny CSV and analyse it end-to-end.
python scripts/analyze.py --synthetic

# Analyse a real CSV (e.g. exported from wandb or from evaluate.py summaries).
python scripts/analyze.py results/sweep_h1.csv \
    --out docs/plan/reports/h1.md --fig docs/plan/reports/h1.png
```

CSV contract: must contain `condition` (or enough metadata columns — arch,
model, env, delay, seed — to derive one) and `return`. Optional timing columns
(`wall_clock`, `steps_per_sec`) are forwarded into the timing table.

---

## 9. Change log

- **2026-07-23** — Initial runbook; describes the shipped single-process `main.py` +
  Hydra composition, the HPC bootstrap, delay/staleness modes, and the sweep pattern. Sweep
  controller, MuJoCo envs, and experiment-specific commands to be appended as the plan is
  implemented.
- **2026-07-23** — Amended for **H5**: added `trm_carry` (persistent-`h` trunk, stateful),
  the `h_backprop` override, and **observation staleness** as a second, independent env
  parameter alongside action delay. See `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §H5.
- **2026-07-23** — Added **H100-hour cost estimates** for every experiment (plan §5.5) and a
  requirement to log wall-clock / steps-per-sec per run so the cost model stays calibrated.
  Resolved earlier open decisions: H5 off-policy `h` carry is in scope, and the `full` vs
  `one_step` backprop ablation is included; remaining choices are config-only (plan §6).
- **2026-07-23** — **PPO/SAC split**: PPO = mechanism + headline standard, SAC = cheap MuJoCo
  replication. Added plan Stage B0 (parity/throughput calibration gate) and revised the v0
  cost estimate from ~180–190 → ~40–60 H100-hours.
- **2026-07-23** — **H5 gating**: added `lstm` recurrent memory baseline and y-on-h readout
  control as unconditional interpretability cells. Family-B sibling ablations (depth /
  deep-supervision / transfer) implemented behind config but headline-gated: only scheduled
  after a positive pendulum signal (`trm_carry ≥ lstm`). See `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §6 decision #9.
- **2026-08-18** — **Stage-A automation backbone**: added `pyscripts/evaluate.py`
  (fixed-seed, exploration-off eval: mean/std + 95% bootstrap CI, lazy torch
  import so `--help` works without torch; fixed a `%`-format bug in the
  `--ci-samples` help string), `scripts/sweep.py` (config grid → exact
  `torchrun --standalone --nproc_per_node=1 main.py <overrides>` commands;
  `--dry-run`/`--local`/`--slurm`, YAML grid support, per-run wall-clock for
  local runs), and `scripts/analyze.py` (torch-free CSV analysis: per-condition
  mean/std, 95% bootstrap CI, Cohen's d, paired/2-sample t-tests, markdown
  tables + figure). See `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §3 (shared
  automation) and §4 stage A.