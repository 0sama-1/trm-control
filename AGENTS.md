# trm-control — Orchestration Instructions

> These instructions are the standing contract for any **coding-manager sub-agent** that
> works on this repository. They encode the project's conventions, the computation
> environment it must target, and how to run every experiment. A human reviewer approves
> any experiment plan before a coding-manager is launched.

---

## 1. What this repo is

`trm-control` tests whether **Tiny Recursive Model (TRM) recursion** — iterative latent
refinement over multiple cycles — improves policy learning in **continuous control RL**.

- Reference: A. Jolicoeur-Martineau, *"Less is More: Recursive Reasoning with Tiny Networks"*, 2025, arXiv:2510.04871.
- Research question: Can TRM-style recursion replace a standard feedforward pass as a policy/critic backbone and (a) match or beat MLP/transformer baselines, and (b) better retain long-horizon / delayed-search knowledge?

The five hypotheses under test:

| ID | Claim |
|----|-------|
| **H1** | Recursive latent policies outperform standard feedforward policies (MLP baseline; transformer as stretch baseline). |
| **H2** | More recursive depth (H_cycles/L_cycles) improves long-horizon / delayed-reward retention. |
| **H3** | Deep supervision (auxiliary loss at intermediate recursive steps) stabilizes training. |
| **H4** | TRM policies transfer better to novel/unseen dynamics than feedforward baselines. |
| **H5** | A persistent high-level latent (`h`-carry) with the output attached to the fast latent (`y`-on-`l`) and explicit backprop-of-`h` mode improves long-horizon / stale-input control vs the stateless TRM, the MLP baseline, **and an equal-size LSTM recurrent baseline** (the latter required so a `trm_carry` win is not merely "memory helps"). |

Each hypothesis must be tested with a **shared backbone-selection mechanism** so the
comparison isolates the model/training mechanism and not the RL algorithm. The repo
already ships DDPG, TD3, SAC and PPO against a shared **trunk contract** (see §4).

---

## 2. Ground rules for the coding-manager

1. **Repository root behaves as a git repo.** All project code lives in this folder
   (`/projects/trm/trm_control_integration`). The parent `/projects/trm` is **not** a git
   repo — never commit anything there. Do all git work inside this folder.
2. **Never push to `main`/`master`; do not push to a remote without being explicitly told.**
   Work on local branches. This environment has no SSH keys installed and the upstream
   (`git@github.com:0sama-1/trm-control.git`) is unverified on this machine, so pushes are
   expected to fail — treat that as expected, not as a bug.
3. **One logical change per branch**; commit small, with descriptive messages.
4. **Leave the code runnable exactly as documented.** Every experiment you add must be
   runnable via the supercomputer template in §6 — a human must be able to launch it from
   a clean checkout using only the documented commands.
5. **Do not silently change experiment protocol.** If you believe a hyperparameter, budget,
   or metric must change, say so in your report; do not just tweak it.
6. **Do not invent data.** Never fabricate run results. If a run didn't complete, say so.
7. **Preserve the trunk contract and Hydra composition.** New backbones/configs extend the
   existing pattern; they do not bolt on a second, divergent framework.
8. **Match the existing style**: Hydra YAML for config, dataclass-free clean modules under
   `src/`, wandb for metrics, `gymnasium` for envs, plain float32 torch.

---

## 3. Repo layout

```
trm_control_integration/
├── AGENTS.md            # this file
├── README.md            # HPC setup + quickstart (human-facing)
├── notes.md             # author's TRM-port integration notes (read this first)
├── main.py              # Hydra entry point  (arch × model × env × delay)
├── conf/                # Hydra config groups
│   ├── config.yaml      # composition defaults + top-level fields (seed, folders, wandb)
│   ├── arch/            # ddpg | td3 | sac | ppo   (RL algorithm + its hyperparams)
│   ├── model/           # dnn | trm | trm_hier6 | trm_singlez | trm_carry | lstm   (backbone trunks)
│   ├── env/             # pendulum | cartpole   (+ any new MuJoCo envs you add)
│   ├── delay/           # none | fixed | stochastic | time_varying   (action staleness)
│   └── staleness/       # (planned) observation-staleness group, independent of delay
├── src/
│   ├── algos/           # ddpg.py td3.py sac.py ppo.py  (algo-specific Trainer + agent)
│   ├── models/          # dnn.py trm_control.py  trm/ (vendored reference, read-only)
│   │                    #   + planned trm_carry.py (H5 persistent-carry trunk)
│   ├── trainer.py       # RLTrainer / OffPolicyTrainer / OnPolicyTrainer
│   ├── delays.py        # delay models + DelayedEnv wrapper (action staleness)
│   ├── logger.py        # WandbLogger
│   ├── recorder.py      # VideoRecorder
│   └── utils.py         # seeds, init, soft_update, atanh
├── pyscripts/           # standalone single-file scripts (sanity checks, quick trials)
├── setup.sh             # HPC environment bootstrap (module load + venv + deps)
├── setup/               # HPC notes
├── models/              # checkpoint output dir (gitignored)
├── docs/
│   ├── RUNBOOK.md       # human-facing how-to-run for every experiment
│   ├── EXPERIMENT_PLAN.md# the full experiment plan (approved; source of truth)
│   ├── kickoff_prompt.md# coding-manager kick-off prompt (WITHHELD until plan approved)
│   └── reports/         # per-stage coding-manager reports (real numbers only)
└── pyproject.toml / .gitignore / .python-version
```

---

## 4. The trunk contract (authoritative)

Every backbone is consumed through the same hooks as the shipped `DNNTrunk`:

```
instantiate(model_cfg, in_dim=<obs_dim or obs_dim+act_dim>)
```

returns a module with:
- an `.out_dim` property (width consumed by the output head), and
- `.forward(x) -> Tensor` (flat continuous input → flat latent).

Actor/Critic heads are then built on top (`make_head` / `layer_init`). **Critics concatenate
`obs` and `act` before the trunk** — so a critic trunk's `in_dim = obs_dim + act_dim`.

New backbones (transformer, LSTM, deep-supervision variants, persistent-carry TRM, …) must
honor exactly this — **except H5's `TRMCarryTrunk`**, which is a deliberate exception: it
carries `h` **across forward calls** (reset at episode boundaries by the rollout machinery),
so it is stateful rather than stateless. That divergence must be documented and handled in the
trainer/buffer, never by forking the whole repo.

Do not change `ddpg.py`/`td3.py`/`sac.py`/`ppo.py`'s trunk-consumption code unless a change
is inherent to the experiment (e.g. deep supervision needs the trunk to expose intermediate
latents, a recurrent backbone needs a history window, or H5 needs per-transition `h` snapshots
in an off-policy replay buffer — config/data concerns handled by the trainer or a wrapper, not
a fork of the whole repo).

---

## 5. Serving the five hypotheses (what each needs)

Grounding for what the plan must cover — see `docs/EXPERIMENT_PLAN.md` for the concrete,
approved schedule.

| Hypothesis | Minimal viable implementation |
|-----------|-------------------------------|
| **H1** recursive > feedforward | TRM trunk vs DNN trunk on pendulum, then on a MuJoCo env, across ≥1 algo. All other knobs identical. |
| **H2** depth → long-horizon | The delay machinery already exists (`conf/delay/*`, `src/delays.py`). Vary H_cycles/L_cycles × delay/staleness (none/fixed/stochastic/time_varying). Also consider sparse envs. |
| **H3** deep supervision stabilizes | Extend a TRM trunk to return intermediate latents; add a weighted aux PPO loss. Compare supervision off / final-only / intermediate. |
| **H4** transfer to novel dynamics | Env wrapper that modifies physics/morphology at eval; train on defaults, evaluate zero-shot + fine-tune on variants. |
| **H5** persistent `h`-carry + backprop-of-h | `TRMCarryTrunk` (output on fast `l`, `h` mediated + carried across calls), `h_backprop: {full, one_step}` config, per-transition `h` plumbing for off-policy replay, an observation-staleness wrapper, an **equal-size LSTM recurrent baseline**, and a y-on-l vs y-on-h readout control. **Family-B sibling ablations (depth / deep-supervision / transfer) are headline-gated** — implemented behind config, scheduled only after a positive pendulum signal (`trm_carry ≥ lstm`); see `docs/EXPERIMENT_PLAN.md` §6 decision #9. |

**Delay and staleness are first-class constructs here.** The repo ships `DelayedEnv`
(action staleness). Long-horizon hypotheses (H2, H5) use **two independent dials** —
**action staleness** (the applied action was chosen `n` steps ago) and **observation
staleness** (the fed observation is `n` steps old) — as controlled proxies for "horizon" and
for input smoothing by the high-level latent.

**Algo policy.** **PPO is the mechanism + headline standard** (deep supervision and the
sequential `h`-carry fit its on-policy objective; aligns with the control-RL literature).
**SAC / other off-policy algos are the replication + cheap-MuJoCo lever** (~35× cheaper per
run). Use PPO for mechanism sweeps on pendulum and the H1 headline cell; use SAC for the
heavy MuJoCo matrices plus one short PPO confirm each. **Before any cross-algo comparison,
run Stage B0** (same pendulum cell under ppo and sac) to verify off-policy episode adequacy
and calibrate the cost model — see `docs/EXPERIMENT_PLAN.md` §§5.5, 6.

---

## 6. THE SUPERCOMPUTER TEMPLATE (must follow for every experiment)

All experiment code must be runnable on the Pawsey **quantum** HPC partition. Reuse the
exact patterns already in the repo (`setup.sh`, `setup/setup.md`, `README.md`). The general
pattern is: **module env → venv → launch `main.py` with Hydra overrides → wandb records**.

### 6.1 One-time environment bootstrap (mirrors `setup.sh`)

```bash
cd <repo-root>          # e.g. /scratch/... / $MYSCRATCH/trm-control
module purge
module load pawsey pawseytools pawseyenv/2026.01 PrgEnv-nvidia craype-arm-grace gcc-native-mixed/12.3
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

Everything a human needs is in `setup.sh`; do not invent a new install path.

### 6.2 Interactive node

```bash
salloc -p quantum -N 1 --gres=gpu:1 --time=01:00:00 -A <allocation>
```

### 6.3 Running an experiment (single config)

The entry point is always `main.py` with Hydra overrides. The default composition groups
are `arch`, `model`, `env`, `delay`, plus top-level `seed`. Examples from the repo:

```bash
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=dnn env=pendulum delay=none seed=42
torchrun --standalone --nproc_per_node=1 main.py arch=ddpg model=trm env=pendulum delay=stochastic seed=42
torchrun --standalone --nproc_per_node=1 main.py arch=sac  model=trm_hier6 env=pendulum delay=fixed       seed=0
```

**Single-process only.** `main.py` never calls `torch.distributed.init_process_group`;
`nproc_per_node` must stay `1`. Do not introduce multi-rank training unless you add the
proper distributed init.

### 6.4 Running a sweep (many seeds/configs)

Generate Slurm batch jobs. A **sweep controller** must be added (see experiment plan —
it is currently missing). Conventions:

- One Slurm job per (config × seed) combination; each job runs a single `main.py` process on one GPU.
- Batch script skeleton (Station to the alloc name for your experiment):

```bash
#!/bin/bash
#SBATCH --partition=quantum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=<allocation>
#SBATCH --job-name=trm-control
#SBATCH --output=logs/%x_%j.out
set -e
cd <repo-root>
source <venv-activate-script>      # module env + activate .venv
torchrun --standalone --nproc_per_node=1 main.py arch=${ARCH} model=${MODEL} env=${ENV} delay=${DELAY} seed=${SEED}
```

- Logs go under `logs/` (add to `.gitignore`); checkpoints under `models/`.
- Each run's wandb run name already encodes `env_algo_model_delay_seed` (see `logger.py`).

### 6.5 Where results live

- **wandb** is the primary metric store (`return`, `avg100_return`, losses, buffer_size).
- **`models/`** holds `.pt` checkpoints (gitignored).
- **`videos/`** holds recorded rollouts (gitignored).
- **Wall-clock / steps-per-sec must be logged per run** so the H100-hour cost model
  (`docs/EXPERIMENT_PLAN.md` §5.5) stays calibrated against real throughput.
- Any aggregation tables / figures / analysis must be written under `docs/` or a new
  `results/` dir with a human-readable markdown.

---

## 7. What every new experiment/feature must include

For each hypothesis deliverable, the coding-manager must produce *all* of:

1. **Code** (extends existing pattern; trunk contract preserved; full config surface
   supported so deferred choices are config-only — see `docs/EXPERIMENT_PLAN.md` §6).
2. **Conf YAML** for any new architecture/env/delay/staleness/sweep dimension.
3. **A documented run command** (or batch script) in `docs/RUNBOOK.md`.
4. **A metrics/eval plan**: how success is measured (mean return over N fixed-seed eval episodes, no exploration; std across ≥5 seeds; bootstrap CI; paired t-test where meaningful).
5. **A cost estimate in H100-hours** for the experiment (see `docs/EXPERIMENT_PLAN.md` §5.5), plus a measured wall-clock / steps-per-sec number from any real run so the cost model stays calibrated.
6. **A short report** of what ran and what did not (no fabricated numbers).

Unless the plan says otherwise, use **≥5 seeds per condition** and report **95% CI**.
Bootstrap CI, Cohen's d and paired tests are expected where the plan calls for them.

---

## 8. Sub-agent launch mechanics (for the orchestrator)

The orchestrator launches a coding-manager via the existing spawner:

```bash
python /projects/trm/scripts/spawn_subagent.py <task_text> --profile huazhen__deepseek-v4-flash --max-iters 200
```

- The spawner defaults to granting `file_editor`, `task_tracker`, `terminal`, `browser_tool_set`; `terminal` implies git access inside the repo.
- **Working directory** for the sub-agent must be the repo root
  `/projects/trm/trm_control_integration` (set via the spawner's `working_dir` / by running
  the spawner from that directory).
- The orchestrator **writes the plan and the kick-off prompt**; the coding-manager executes.
- The orchestrator reviews the coding-manager's diff/branch before anything is merged or run.
- **Do not send the kick-off prompt until a human has approved the experiment plan.**

---

## 9. Definition of done for the coding-manager

A task is done when:
- Code + config + runbook + report for the assigned hypothesis are present and consistent.
- The trunks follow the contract and Hydra composition resolves (`main.py` runs with the documented command, at least for a smoke config on `env=pendulum`).
- Nothing fabricated; any unrun item is explicitly flagged.
- Work is committed on a clearly named local branch.

---

*Last updated: 2026-07-23.*