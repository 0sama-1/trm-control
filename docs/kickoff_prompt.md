# Kick-off prompt — coding-manager (WITHHELD pending approval)

> **Status: NOT SENT.** This prompt must not be dispatched to the coding-manager until a
> human has approved `docs/EXPERIMENT_PLAN.md`. Do not launch the sub-agent with this text
> before approval.

---

## To the coding-manager sub-agent

You are the **coding-manager** for the `trm-control` repository at
`/projects/trm/trm_control_integration`. You will implement a staged experiment program that
tests whether Tiny Recursive Model (TRM) recursion improves policy learning in continuous
control RL.

Before doing anything else, read, in order:
1. `AGENTS.md` — this is your operative handle on how this repo works, the trunk contract,
   the supercomputer run template, and your ground rules. **Follow it exactly.**
2. `notes.md` — the TRM-to-RL port rationale (what was kept vs dropped, and why).
3. `docs/EXPERIMENT_PLAN.md` — the approved plan. **Implement it literally.**
4. `docs/RUNBOOK.md` — how humans run experiments on the Pawsey quantum HPC; keep it current.

## Your working contract

- **Git:** this folder is a self-contained git repo. Parent `/projects/trm` is NOT a git repo.
  Work on a clearly named local branch (one logical change per branch). Never push to
  `main`/`master`; pushes to the SSH upstream are expected to fail in this environment (no
  SSH keys) — that is expected, not a bug. Commit small and often.
- **Environment:** everything must run on the Pawsey **quantum** partition under the exact
  template in `AGENTS.md` §6 — module env → venv → single-process `torchrun main.py
  <overrides>` → wandb. Never set `nproc_per_node > 1`.
- **No fabrication. Ever.** Report what actually ran and what did not. If a run is missing or
  failed, say so explicitly. Do not invent numbers.
- **Algo split.** Follow plan §5.5: PPO for mechanism + headline; SAC for the heavy MuJoCo
  replication. Don't silently expand PPO-MuJoCo cells — that is the cost lever.
- **Preserve the trunk contract** (§4 AGENTS.md): `instantiate(model_cfg, in_dim=...)` →
  `.out_dim` + `.forward(x)`. New backbones extend this pattern; do not fork the repo.
- **Smoke-test on `env=pendulum` before any MuJoCo run.**

## Scope — implement the plan in order

**Stage A — automation backbone**
1. `scripts/sweep.py`: take a grid (arch×model×env×delay×staleness×seed) and either submit one
   single-process Slurm job per combo per the `AGENTS.md` §6.4 template, or run locally in
   sequence when GPUs are present. Follow the HPC module→venv→torchrun pattern.
2. `pyscripts/evaluate.py` (or a `main.py` eval mode): fixed-seed, exploration-off
   evaluation of a checkpoint → mean/std return over ≥10 episodes.
2a. **Observation-staleness wrapper** (independent of action delay): a wrapper that
   serves the agent an observation that is `n` steps old, active via a `conf/staleness/`
   group or `env.staleness_obs`. Action staleness already exists as `conf/delay/*`. Both
   are consumed by H2 and H5 (plan §3).
3. `scripts/analyze.py`: from wandb/CSV, compute per-condition mean/std, 95% bootstrap CI,
   Cohen's d and paired t-test where the plan calls for it; emit `docs/` markdown tables +
   figures. Ensure `logs/` and `results/` are gitignored.
3a. **Cost tracking**: log wall-clock / steps-per-sec per run and surface the measured
   H100-hours against the plan §5.5 estimates so the cost model stays calibrated.
4. Update `docs/RUNBOOK.md` and `docs/EXPERIMENT_PLAN.md` status as you go.

**Stage B0 — training-parity / cost calibration (gate)**
4a. Run the SAME pendulum cell under `arch=ppo` and `arch=sac` (5 seeds each). Record final
   return AND wall-clock; verify off-policy episode adequacy (200 episodes) is sufficient on
   pendulum, and later recheck on hopper/halfcheetah. This calibrates the §5.5 H100-hour model
   and decides which cells run in sac vs ppo.

**Stage B — H1 current-state repro**
5. `arch=ppo model=trm vs dnn, env=pendulum, delay=none`, 5 seeds each, via `sweep.py` and
   `evaluate.py`. Report mean/std/CI + a paired comparison.

**Stage C — MuJoCo**
6. Add `mujoco` + `gymnasium[mujoco]` to `setup.sh`; add
   `conf/env/{halfcheetah,hopper,walker2d,ant,humanoid}.yaml` (continuous `Box` actions only).

**Stage D — H1 transformer trunk**
7. `TransformerTrunk` honoring the trunk contract, with a stateless observed-history window
   wrapper (not a fork of the actor/critic wiring); separate actor/critic backbones;
   `conf/model/transformer.yaml` (num_layers, num_heads, embed_dim, context_length, dropout).
   Document the minimal wrapper changes needed for PPO's RolloutBuffer vs DDPG/TD3/SAC replay.

**Stage E — H1 main matrix**
8. `model=trm|dnn|transformer × env=hopper|halfcheetah × delay=none`, `arch=ppo` (add `sac`
   only if it runs comfortably), 5 seeds each. Report the H1 comparison with stats.

**Stage F — H2 depth × delay**
9. On `model=trm`, `env=pendulum` → `hopper`: sweep `model.trm.H_cycles ∈ {1,2,3}` ×
   `delay ∈ {none, fixed, stochastic, time_varying}` × `staleness ∈ {none, obs<n>}`, 5
   seeds. Then a MuJoCo follow-up with a tuned fixed delay. Optional strength: a sparse-reward
   env variant if cheap.

**Stage G — H3 deep supervision**
10. Extend `TRMTrunk` to optionally emit latents at selected recursion checkpoints (e.g.
   `{1,3,5}`), gated by config so default behavior is unchanged; add a weighted auxiliary
   PPO loss (full gradient). Ablate: off / every step / `{1,3,5}` w=`[0.1,0.3,0.6]` on
   `arch=ppo model=trm env=pendulum delay=none`, 5 seeds. Report final return, seed variance,
   convergence speed.

**Stage H — H4 dynamics transfer**
11. Dynamics wrapper (gravity/friction/morphology scaling) without touching the trainer.
    Train `trm` and `dnn` on `env=hopper delay=none` (5 seeds); evaluate zero-shot on gravity
    {0.5,1.0,2.0}×, friction {0.5,1.0,2.0}×, morphology {0.8,1.0,1.2}×; report % retention;
    fine-tune on the hardest variant (~100k steps) and report the adaptation curve.

**Stage I — H5 persistent-carry trunk + backprop-of-h + observation staleness**
12. `TRMCarryTrunk` (`conf/model/trm_carry.yaml`): output attached to the fast `l`
    (y-on-l), slow `h` mediating `l` and **persisting across forward calls** (reset at episode
    boundaries by the rollout machinery). This is a deliberate, documented exception to the
    stateless trunk contract (AGENTS.md §4).
13. `h_backprop: {full, one_step}` config — `full` = full BPTT through all cycles (TRM);
    `one_step` = hold `h` as context, backprop to `l` only (HRM/HRM-Agent). This resolves the
    HRM-vs-TRM backprop disagreement in control.
14. **Carry plumbing**: PPO (sequential) carries `h` per env step naturally; DDPG/TD3/SAC
    (random replay) must store per-transition `h` snapshots in the replay buffer and feed them
    through the trunk. Keep changes minimal and guarded by the model type.
15. H5 ablation (arch=ppo, env=pendulum → hopper), 5 seeds: stateless `trm` vs carry
    (one_step vs full) × staleness {none, obs stale} × {action fixed}. Report mean/std/CI +
    the full-vs-one_step comparison, and confirm `h` resets correctly across episodes.

## Reporting

At the end of each stage, commit your changes and write a short markdown report into
`docs/reports/` with: what ran, **real metrics**, **measured H100-hours (with the plan §5.5
estimate alongside)**, what did not run and why, and any requested plan deviations raised for
human review rather than silently applied.

Work in order A→I. Proceed without stopping between stages, but flag anything that blocks a
stage in the running report rather than guessing.

Begin by reading the four files in the order given above, then confirm to me that you have
internalized the trunk contract and the HPC template before you write any code.