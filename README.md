# trm-control — TRM Recursive Reasoning for Continuous-Control RL

This repo implements **Tiny Recursive Model (TRM)** trunks as drop-in
replacements for standard feedforward MLP backbones in continuous-control RL
(DDPG, TD3, SAC, PPO). The research question: *Can TRM-style iterative latent
refinement improve policy learning, particularly for long-horizon and
delayed-search problems?*

Reference: A. Jolicoeur-Martineau, *"Less is More: Recursive Reasoning with
Tiny Networks"*, 2025, arXiv:2510.04871.

**Everything runs on the Pawsey `quantum` HPC partition (GH200/H100 GPUs).**
No local-CPU experiment runs. See [Supercomputer Setup](#3-supercomputer-setup)
below.

---

## Table of Contents

1. [Quick start](#1-quick-start)
2. [Repo layout](#2-repo-layout)
3. [Supercomputer setup](#3-supercomputer-setup)
4. [Running experiments](#4-running-experiments)
   - [Single run](#41-single-run)
   - [Sweep controller (Slurm submission)](#42-sweep-controller-slurm-submission)
   - [Eval script](#43-eval-script)
   - [Analysis script](#44-analysis-script)
5. [Hypotheses under test](#5-hypotheses-under-test)
6. [The trunk contract](#6-the-trunk-contract)
7. [Ground rules](#7-ground-rules)
8. [Adding a new model / env / experiment](#8-adding-a-new-model--env--experiment)
9. [Definition of done](#9-definition-of-done)
10. [Current project status & experiment runbook](#10-current-project-status--experiment-runbook)

---

## 1. Quick start

On the Pawsey `quantum` partition (after `salloc` — see §3):

```bash
# Bootstrap the environment (one-time per checkout, ~2 min).
bash setup.sh

# Sanity check.
python pyscripts/test_gpu.py

# Run a single pendulum experiment.
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=dnn env=pendulum delay=none seed=42

# Run a full sweep via Slurm (submits one sbatch per cell).
python scripts/sweep.py --arch "ppo sac" --model "dnn trm" --env pendulum --delay none --seed "0 1 2 3 4"
```

---

## 2. Repo layout

```
trm_control_integration/
├── README.md             # this file — human-facing guide + runbook
├── notes.md              # TRM-port integration notes (read this first)
├── main.py               # Hydra entry point (arch × model × env × delay)
├── conf/
│   ├── config.yaml       # composition defaults + top-level fields (seed, folders, wandb)
│   ├── arch/             # ddpg | td3 | sac | ppo  (RL algorithm + its hyperparams)
│   ├── model/            # dnn | trm | trm_hier6 | trm_singlez | trm_carry | lstm  (backbone trunks)
│   ├── env/              # pendulum | cartpole  (+ MuJoCo envs when added)
│   ├── delay/            # none | fixed | stochastic | time_varying  (action staleness)
│   └── staleness/        # none | fixed | stochastic | time_varying  (observation staleness)
├── src/
│   ├── algos/            # ddpg.py td3.py sac.py ppo.py
│   ├── models/           # dnn.py trm_control.py (+ planned transformer.py, trm_carry.py)
│   ├── trainer.py        # RLTrainer / OffPolicyTrainer / OnPolicyTrainer
│   ├── delays.py         # delay models + DelayedEnv wrapper (action staleness)
│   ├── staleness.py      # StaleObservationWrapper (observation staleness)
│   ├── logger.py         # WandbLogger
│   ├── recorder.py       # VideoRecorder
│   └── utils.py          # seeds, init, soft_update, atanh
├── pyscripts/            # standalone scripts (evaluate.py, sanity checks, quick trials)
├── scripts/
│   ├── sweep.py          # sweep controller (Slurm submission default)
│   └── analyze.py        # torch-free CSV analysis (mean/std/CI/Cohen's d/t-test)
├── setup.sh              # HPC environment bootstrap (module load + venv + deps)
├── setup/                # HPC setup notes
├── models/               # checkpoint output dir (gitignored)
├── results/              # eval JSON / CSV output (gitignored)
├── logs/                 # Slurm stdout/stderr (gitignored)
└── videos/               # recorded rollouts (gitignored)
```

The full experiment plan lives one level up at `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md`.

---

## 3. Supercomputer setup

### 3.1 Request an interactive node

```bash
salloc -p quantum -N 1 --gres=gpu:1 --time=01:00:00 -A pawsey1302-q
```

### 3.2 Bootstrap the environment

```bash
cd /path/to/trm_control_integration

module purge
module load pawsey pawseytools pawseyenv/2026.01
module load PrgEnv-nvidia
module load craype-arm-grace
module load gcc-native-mixed/12.3
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

Or just run `bash setup.sh` (same thing). If the code fails on a CUDA mismatch,
check `nvidia-smi` — the GPU CUDA version (quantum: 12.7) must be ≥ the loaded
CUDA module version.

### 3.3 Useful sanity checks

```bash
python pyscripts/test_gpu.py      # torch + CUDA availability + GPU name
uname -m                           # hardware architecture
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

---

## 4. Running experiments

### 4.1 Single run

The entry point is **`main.py`** with Hydra overrides. Compose from six groups:

| Group | Choices | Notes |
|-------|---------|-------|
| `arch=` | `ddpg`, `td3`, `sac`, `ppo` | RL algorithm + hyperparams in `conf/arch/*.yaml` |
| `model=` | `dnn`, `trm`, `trm_hier6`, `trm_singlez`, `trm_carry`, `lstm` | Backbone trunk in `conf/model/*.yaml` |
| `env=` | `pendulum`, `cartpole` | Only continuous-(Box)-action envs runnable currently |
| `delay=` | `none`, `fixed`, `stochastic`, `time_varying` | Action staleness (applied action was chosen `n` steps ago) |
| `staleness=` | `none`, `fixed`, `stochastic`, `time_varying` | Observation staleness (fed observation is `n` steps old) |
| `seed=` | int | Reproducibility; wandb run name encodes seed |

```bash
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=dnn env=pendulum delay=none seed=42
torchrun --standalone --nproc_per_node=1 main.py arch=ppo model=trm env=pendulum delay=stochastic seed=0
torchrun --standalone --nproc_per_node=1 main.py arch=sac model=trm_hier6 env=pendulum delay=fixed seed=0
```

**Single-process only.** `main.py` does not call `torch.distributed.init_process_group`.
`nproc_per_node` must stay `1`. A higher value starts multiple uncoordinated
copies racing on the same GPU and checkpoint file.

Extra Hydra overrides (e.g. `model.trm.H_cycles=3`) can be appended as
`key=value` pairs. Wandb configuration goes in `src/.env` (`WANDB_API_KEY`), or
disable it with `wandb.enabled=false`.

Checkpoints auto-resume from `models/<env>_<arch>_<model>_<delay>_seed<seed>.pt`.

### 4.2 Sweep controller (Slurm submission)

For multi-config × multi-seed sweeps, use **`scripts/sweep.py`**. Its **default
mode is Slurm submission** (one `#SBATCH` job per cell, following the
`AGENTS.md` §6.4 template). This is how all H1–H5 experiments are launched.

```bash
# Dry-run: print commands only (torch-free, no execution).
python scripts/sweep.py --dry-run --arch "ppo sac" --model "dnn trm" \
    --env pendulum --delay "none stochastic" --seed "0 1 2 3 4"

# Slurm submission (default): one sbatch job per cell.
python scripts/sweep.py --arch sac --model "dnn trm" \
    --env hopper halfcheetah --delay none --seed "0 1 2 3 4" \
    --account pawsey1302-q --time 04:00:00

# Local sequential execution (debugging only, single node).
python scripts/sweep.py --local --arch ppo --model "dnn trm" \
    --env pendulum --delay none --seed "0 1 2 3 4" --timeout-min 15

# Arbitrary grid from a YAML file (full flexibility).
python scripts/sweep.py --grid conf/sweeps/h1_smoke.yaml
```

Extra Hydra overrides (e.g. `model.trm.H_cycles=3`) are passed via `--extra`
(repeatable) or via a YAML grid file. The `--manifest-out` flag writes a JSON
manifest of all commands.

### 4.3 Eval script

**`pyscripts/evaluate.py`** evaluates a checkpoint with fixed-seed,
exploration-off rollouts:

```bash
# Help (torch-free at parse time).
python pyscripts/evaluate.py --help

# 10 fixed-seed eval episodes of a specific checkpoint.
python pyscripts/evaluate.py --n-episodes 10 --ckpt models/Pendulum-v1_sac_trm_none_seed0.pt \
    arch=sac model=trm env=pendulum delay=none seed=0 --wandb

# Smoke test: random-init policy, no checkpoint needed.
python pyscripts/evaluate.py --n-episodes 2 --max-steps 5 \
    arch=ppo model=dnn env=pendulum delay=none seed=42
```

Output: `results/eval_<identity>_n<N>.json` with per-episode returns, mean,
std, SEM, 95% CI, and timing.

### 4.4 Analysis script

**`scripts/analyze.py`** reads a results CSV (condition × metric), computes
per-condition mean/std, 95% bootstrap CI, Cohen's d, and paired/2-sample
t-tests, then writes a markdown report + bar figure. **Torch-free** (numpy +
scipy only) — runs on the login node or workstation.

```bash
# Smoke test: synthesize a tiny CSV and analyse it end-to-end.
python scripts/analyze.py --synthetic

# Analyse a real CSV (exported from wandb or evaluate.py summaries).
python scripts/analyze.py results/sweep_h1.csv \
    --out docs/plan/reports/h1.md --fig docs/plan/reports/h1.png
```

---

## 5. Hypotheses under test

Five hypotheses are being tested. See `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md`
for the full, approved schedule.

| ID | Claim | Minimal testing |
|----|-------|-----------------|
| **H1** | Recursive latent policies outperform standard feedforward policies (MLP baseline; transformer as stretch baseline). | TRM trunk vs DNN trunk on pendulum, then on MuJoCo envs, across ≥1 algo. |
| **H2** | More recursive depth improves long-horizon / delayed-reward retention. | Vary H_cycles/L_cycles × delay/staleness. |
| **H3** | Deep supervision (auxiliary loss at intermediate steps) stabilizes training. | Compare supervision off / final-only / intermediate. |
| **H4** | TRM policies transfer better to novel/unseen dynamics. | Env wrapper perturbing physics at eval; zero-shot + fine-tune. |
| **H5** | A persistent high-level latent (`h`-carry) with explicit backprop-of-h improves long-horizon / stale-input control vs stateless TRM, MLP, **and an equal-size LSTM recurrent baseline**. | `TRMCarryTrunk` + obs-staleness wrapper + LSTM baseline. Family-B ablations headlined-gated on `trm_carry ≥ lstm`. |

**Algorithm policy.** Use **PPO** for mechanism sweeps and the H1 headline cell;
use **SAC** (or DDPG/TD3) for heavy MuJoCo replication (~35× cheaper per run).
Before mixing algos, run the **B0 parity check** (same pendulum cell under both
PPO and SAC — see §10) to verify off-policy episode adequacy and calibrate the
cost model.

**Delay and staleness** are two independent dials for effective horizon:
**action staleness** (`delay=`) delays the action applied to the env;
**observation staleness** (`staleness=`) feeds the agent an older observation.
They compose independently.

---

## 6. The trunk contract

Every backbone trunk is consumed through the same interface as the shipped
`DNNTrunk`:

```python
instance = instantiate(model_cfg, in_dim=<obs_dim or obs_dim + act_dim>)
```

Returns a `nn.Module` with:
- `.out_dim` property (width consumed by the output head)
- `.forward(x) -> Tensor` (flat continuous input → flat latent)

Actor/Critic heads are built on top (`make_head` / `layer_init`). **Critics
concatenate `obs` and `act` before the trunk**, so a critic trunk's `in_dim =
obs_dim + act_dim`.

New backbones (transformer, LSTM, deep-supervision variants, persistent-carry
TRM, ...) must honor exactly this contract — **except H5's `TRMCarryTrunk`**,
which is a deliberate exception: it carries `h` across forward calls (reset at
episode boundaries by the rollout machinery), so it is stateful rather than
stateless.

Do not change `src/algos/`'s trunk-consumption code unless a change is inherent
to the experiment (e.g. deep supervision needs the trunk to expose intermediate
latents, or H5 needs per-transition `h` snapshots in an off-policy replay
buffer).

---

## 7. Ground rules

1. **One logical change per branch**; commit small, with descriptive messages.
2. **Leave the code runnable exactly as documented.** Every experiment must be
   launchable from a clean checkout using only the commands in this README.
3. **Do not silently change experiment protocol.** If a hyperparameter, budget,
   or metric must change, say so in the report; do not just tweak it.
4. **Do not invent data.** Never fabricate run results. If a run didn't
   complete, say so explicitly.
5. **Preserve the trunk contract and Hydra composition.** New backbones/configs
   extend the existing pattern; they do not bolt on a second, divergent
   framework.
6. **Match the existing style**: Hydra YAML for config, clean modules under
   `src/`, wandb for metrics, `gymnasium` for envs, plain float32 torch.

---

## 8. Adding a new model / env / experiment

1. **New backbone**: add `conf/model/<name>.yaml` + a class in `src/models/`
   honoring the trunk contract (`.out_dim` + `.forward(x)`). Exception: H5's
   `TRMCarryTrunk` is stateful and needs rollout/reset handling.
2. **New env** (e.g. MuJoCo): add `conf/env/<name>.yaml` and ensure the
   dependency is in `setup.sh`. Only `Box` (continuous) action spaces are
   currently supported by the actor heads.
3. **New delay/staleness mode**: add `conf/delay/<name>.yaml` or
   `conf/staleness/<name>.yaml` + the matching wrapper in `src/delays.py` /
   `src/staleness.py`. The two channels are independent.
4. Always run a **smoke test** on `env=pendulum` first, then launch the full
   matrix on the supercomputer.
5. Every addition gets a documented run command in this README (add it to §10
   below) and a cost estimate in H100-hours (see
   `/workspace/trm/docs/plan/EXPERIMENT_PLAN.md` §5.5).

---

## 9. Definition of done

A task in this repo is done when:

1. **Code** (extends existing pattern; trunk contract preserved; full config
   surface supported).
2. **Conf YAML** for any new architecture/env/delay/staleness/sweep dimension.
3. **A documented run command** in this README (§10 runbook).
4. **A metrics/eval plan**: mean return over N fixed-seed eval episodes (no
   exploration), std across ≥5 seeds, 95% bootstrap CI, paired t-test where
   meaningful.
5. **A cost estimate in H100-hours** for the experiment, plus a measured
   wall-clock / steps-per-sec number from any real run so the cost model stays
   calibrated.
6. **A short report** of what ran and what did not (no fabricated numbers).
7. **Nothing fabricated**; any unrun item is explicitly flagged.

---

## 10. Current project status & experiment runbook

> This section is updated as experiments are completed. Each entry gives the
> single-line bash command to run that experiment on the Pawsey quantum HPC.
> After obtaining an interactive node (`salloc -p quantum -N 1 --gres=gpu:1
> --time=01:00:00 -A pawsey1302-q`) and bootstrapping the environment (§3),
> copy and paste the command below.

### Legend

- ✅ **Complete** — results collected and report committed
- 🟡 **In progress** — running or pending results
- 📄 **Ready to run** — command is specified; human needs to run it on the
  supercomputer
- ⬜ **Not yet ready** — prerequisites incomplete (e.g. model not implemented)

---

### B0 — Calibration gate (pendulum, PPO + SAC, 5 seeds each)

**Status:** 📄 Ready to run

```bash
python scripts/sweep.py --arch "ppo sac" --model "dnn trm" --env pendulum --delay none --seed "0 1 2 3 4"
```

This submits 20 Slurm jobs (2 algos × 2 models × 5 seeds). After all jobs
complete, collect results and write the calibration report to
`docs/plan/reports/b0-calibration.md`. This gate finalises the SAC-vs-PPO
allocation for all MuJoCo experiments below.

---

### H1 — Recursive vs feedforward

#### H1a — Pendulum repro (PPO, 5 seeds)

**Status:** 📄 Ready to run

```bash
python scripts/sweep.py --arch ppo --model "dnn trm" --env pendulum --delay none --seed "0 1 2 3 4"
```

10 Slurm jobs. Confirm TRM ≥ MLP is reproducible on pendulum before MuJoCo.

#### H1b — MuJoCo main matrix (PPO + SAC, 3 models × 2 envs, 5 seeds)

**Status:** ⬜ Not yet ready (needs MuJoCo envs + transformer model)

```bash
python scripts/sweep.py --arch "ppo sac" --model "dnn trm transformer" --env "hopper halfcheetah" --delay none --seed "0 1 2 3 4"
```

~120 Slurm jobs. The H1 headline cell. SAC replication keeps the MuJoCo bill
at ~1.5 H100-hr (§5.5).

---

### H2 — Depth × delay/staleness matrix

**Status:** ⬜ Not yet ready (depends on H1b config + staleness wrapper complete)

```bash
python scripts/sweep.py --arch ppo --model "dnn trm" --env "hopper halfcheetah" --delay "none fixed stochastic" --seed "0 1 2 3 4"
```

Vary `H_cycles` / `L_cycles` via `--extra model.trm.H_cycles=N` for the depth
dimension. The full grid may use a YAML config.

---

### H3 — Deep-supervision ablation

**Status:** ⬜ Not yet ready (needs deep-supervision mechanism implemented)

```bash
python scripts/sweep.py --arch ppo --model trm --env "hopper halfcheetah" --delay none --seed "0 1 2 3 4" --extra "model.trm.deep_supervision=true"
```

Compare `deep_supervision=false` vs `true` across envs.

---

### H4 — Dynamics transfer

**Status:** ⬜ Not yet ready (needs dynamics wrapper implemented + MuJoCo envs)

```bash
python scripts/sweep.py --arch ppo --model "dnn trm" --env hopper --delay none --seed "0 1 2 3 4" --extra "dynamics=perturbed"
```

Zero-shot eval on perturbed dynamics, then fine-tune.

---

### H5 — Persistent-carry trunk

#### H5-core — `trm_carry` vs LSTM vs DNN, depth × delay × staleness

**Status:** ⬜ Not yet ready (needs carry trunk implemented + staleness wrapper)

```bash
python scripts/sweep.py --arch ppo --model "trm_carry dnn lstm" --env "hopper halfcheetah" --delay "none fixed stochastic" --seed "0 1 2 3 4" --extra "staleness=fixed"
```

#### H5 Family-B — Sibling ablations (depth, deep-supervision, transfer)

**Status:** ⬜ Gated (not scheduled until `trm_carry ≥ lstm` on pendulum)

```bash
python scripts/sweep.py --arch ppo --model trm_carry --env "hopper halfcheetah" --delay none --seed "0 1 2 3 4" --extra "model.trm_carry.depth=3 model.trm_carry.deep_supervision=true model.trm_carry.dynamics_transfer=true"
```

---

### Results

- **wandb** is the primary metric store (return, avg100_return, losses, steps).
- **`models/`** — `.pt` checkpoints (gitignored).
- **`results/`** — eval JSON / analysis CSV (gitignored).
- **`docs/plan/reports/`** — analysis reports (markdown tables + figures).
- Per-run **wall-clock / steps-per-sec** are logged to wandb by `trainer.py`
  so the H100-hour cost model stays calibrated.