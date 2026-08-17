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

The entry point is always **`main.py`** with Hydra overrides. Compose from four groups:
`arch` (RL algorithm), `model` (backbone trunk), `env`, `delay`, plus a `seed`. (An
observation-staleness group `staleness=` is planned alongside `delay` — see §5.)

```bash
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=dnn env=pendulum delay=none seed=42
```

| Group | Choices | Notes |
|-------|---------|-------|
| `arch=` | `ddpg`, `td3`, `sac`, `ppo` | RL algorithm + its hyperparameters in `conf/arch/*.yaml` |
| `model=` | `dnn`, `trm`, `trm_hier6`, `trm_singlez`, `trm_carry`, `lstm` | Backbone trunk in `conf/model/*.yaml`. `dnn` = MLP baseline; `trm` = 2-latent TRM; `trm_hier6` = 6 parallel latents; `trm_singlez` = single latent; `trm_carry` = H5 persistent-`h` variant (planned); `lstm` = recurrent memory baseline (**required for H5's interpretability**, see `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §H5) |
| `env=` | `pendulum`, `cartpole` | Currently only continuous-(Box)-action envs are runnable by the actor heads: DDPG/TD3/SAC/PPO. `cartpole` is discrete and **not runnable yet** |
| `delay=` | `none`, `fixed`, `stochastic`, `time_varying` | **Action staleness** — the applied action was chosen `n` steps ago (proxy for horizon). See §5 |
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
| `staleness=obs<n>` *(planned)* | `conf/staleness/*` | observation | The observation given to the agent is `n` steps old, independent of action delay |

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
3. **New delay/staleness**: add `conf/delay/<name>.yaml` (action staleness) or a
   `conf/staleness/` entry (observation staleness) + the matching wrapper in `src/delays.py`
   / `src/staleness.py`. The two channels are independent.
4. Always run a **smoke test** on `env=pendulum` first, then launch the full matrix.

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