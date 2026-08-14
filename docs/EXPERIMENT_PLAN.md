# EXPERIMENT PLAN — TRM-control

> **Source of truth for what the coding-manager must implement.** Approved by a human before
> any kick-off prompt is sent. Every experiment below is to be implemented following the
> supercomputer template in `AGENTS.md` §6 and documented in `docs/RUNBOOK.md`.

Version: 0.2 (draft for approval) · Last updated: 2026-07-23

---

## 0. Guiding principles

1. **Isolate the mechanism, not the algorithm.** All comparisons hold the RL algorithm
   (`arch`) and every training knob fixed and vary only the backbone (`model`) or the
   training mechanism under test. This is why the trunk contract (§4 in `AGENTS.md`) exists.
2. **Delay is the controlled "horizon" dial.** The repo ships `conf/delay/*` +
   `DelayedEnv`. Long-horizon hypotheses (H2, and the motivation behind H3/H4's "harder"
   cases) express "harder to search" as longer / stochastic delay.
3. **Staleness covers both observations and actions.** The env exposes two independent,
   configurable staleness channels: **observation staleness** (the observation fed to the
   agent is `n` steps old) and **action staleness** (the action applied to the env was
   chosen `n` steps ago, i.e. the existing `DelayedEnv`). Both are first-class environment
   parameters used by H2 and H5 to control effective horizon / input smoothing.
4. **Run on the HPC exactly as shipped** — module → venv → single-process `torchrun main.py
   <overrides>` → wandb. No custom framework.
5. **Every condition ≥5 seeds**, aggregations with **95% CI**. No fabricated numbers.
6. **Smoke-test on `env=pendulum` before any MuJoCo run.**
7. **Every experiment carries an H100-hour cost estimate.** See §Cost. Estimates are
   planning numbers to be refined with measured throughput from real sweep logs; the code
   must log per-run wall-clock/step-rate so the model stays calibrated.
8. **The code must support the full configuration surface.** Where a decision is deferred
   (see §6), the code still accepts every option (models, delays, staleness, backprop modes,
   envs, hyperparameters) so flipping a config later is trivial and requires no code change.
9. **Cost strategy — PPO is the mechanism standard, SAC the performance/replication lever.**
   Use **PPO** as the primary algorithm for mechanism hypotheses (H2, H3, H5 — deep
   supervision and the sequential `h`-carry fit PPO's on-policy objective naturally) and for
   the publishable H1 headline (aligned with the control-RL literature). Use **SAC** (and the
   other cheap off-policy algos) to *replicate/confirm* results and to run the heavy MuJoCo
   matrices at ~35× lower per-run cost. This split is the main lever for the v0 cost reduction
   (see §5.5).**

---

## 1. Current capabilities audit (what's already here)

| Capability | Status | Where |
|-----------|--------|-------|
| RL algos: DDPG, TD3, SAC, PPO | ✅ | `src/algos/*.py` |
| MLP baseline trunk (`DNNTrunk`) | ✅ | `src/models/dnn.py`, `conf/model/dnn.yaml` |
| TRM trunks: `trm`, `trm_hier6`, `trm_singlez` | ✅ | `src/models/trm_control.py` |
| **Persistent-carry TRM variant (H5: y-on-l, h-mediates)** | ❌ | needed — see H5; current trunks are stateless per call |
| **Backprop-of-h mode (`full`/`one_step`)** | ❌ | needed — TRM (full BPTT) vs HRM (1-step fixed-point) dispute unresolved in control |
| Delay wrapper (action staleness) + 4 delay modes | ✅ | `src/delays.py`, `conf/delay/*.yaml` |
| **Observation-staleness wrapper** | ❌ | needed — only action staleness implemented today |
| Wandb + checkpoints + resume + video | ✅ | `src/logger.py`, `trainer.py` |
| HPC bootstrap | ✅ | `setup.sh`, `setup/setup.md` |
| **Transformer baseline trunk** | ❌ | needed for H1 stretch |
| **MuJoCo envs** (Hopper/Cheetah/…) | ❌ | only `pendulum`, `cartpole`; actor heads need continuous `Box` action spaces |
| **Sparse-reward envs** | ❌ | needed for H2 strength |
| **Deep-supervision mechanism** | ❌ | trunk returns only final `z_H`; no intermediate exposures / aux loss |
| **Transfer/dynamics wrapper** | ❌ | needed for H4 |
| **Sweep controller (Slurm) + analysis scripts** | ❌ | needed for all multi-seed runs |
| **Eval protocol (fixed-seed, no-exploration eval)** | ❌ | training-loop `return` is not an eval protocol |

Gap framing: the repo is a **foundation + one baseline**. The plan's job is to add the
backbones, mechanisms, envs, and automation needed to test the **5** hypotheses rigorously
(H1–H4 described here; H5 adds the persistent-carry recursive variant).

---

## 2. Deliverables by hypothesis

### H1 — Recursive latent policies outperform standard feedforward policies

**Question:** With all else fixed, does `model=trm` beat `model=dnn` on the same algorithm,
env, delay, seed? (Transformer as a stretch baseline.)

**Tasks for the coding-manager:**

1. **(Re)verify current state**: run `arch=ppo model=trm vs dnn, env=pendulum, delay=none`
   for 5 seeds each; confirm TRM ≥ MLP baseline is at least reproducible on a toy task before
   spending HPC time on MuJoCo.
2. **Add a Transformer trunk** (`TransformerTrunk`) honoring the trunk contract:
   - `conf/model/transformer.yaml` with configurable `num_layers`, `num_heads`, `embed_dim`,
     `context_length`, `dropout`.
   - Recurrent-style history window: since the trunk takes a flat feature vector, implement
     a **stateless observed-history wrapper** (e.g. an `ObservationHistory` module that
     maintains the last `context_length` observations per env step) feeding the transformer,
     NOT a change to the actor/critic head wiring. Ensure it interacts correctly with
     `RolloutBuffer` for PPO and the replay-driven updates for DDPG/TD3/SAC (may require a
     per-algo wrapper update — document the smallest change).
   - Separate actor/critic backbones to avoid gradient conflict (per the literature).
3. **MuJoCo dependencies**: add `mujoco`, `gymnasium[mujoco]` to `setup.sh`; add
   `conf/env/{halfcheetah,hopper,walker2d,ant,humanoid}.yaml` (all continuous `Box`
   actions).
4. **Algo split for the main matrix** — `ppo` is the **headline/algo-attribute** vehicle
   (aligns with the HRM-Agent/Tiny-Recursive-Control literature); `sac` is the
   **replication/cheap-confirm** vehicle on the heavy MuJoCo cells (per principle #9):

   | model | env | delay | seeds | budget (algo) |
   |-------|-----|-------|-------|----------------|
   | `trm`, `dnn`, `transformer` | `pendulum` (smoke) | `none` | 5 | ppo |
   | `trm`, `dnn`, `transformer` | `hopper`, `halfcheetah` | `none` | 5 | sac (headline cell also run in ppo) |

   Metric: mean return over ≥10 fixed-seed eval episodes with exploration off; report std
   across seeds + 95% bootstrap CI + paired t-test (TRM vs each baseline) + Cohen's d.
5. **Training-parity calibration (Stage B0)** — before trusting any cross-algo comparison,
   run the SAME pendulum cell under both `ppo` and `sac` (5 seeds each) and record final
   return and wall-clock. This (a) verifies 200 episodes is sufficient for the off-policy
   algos (a known trap on MuJoCo — if not, the off-policy budget must be raised and the
   "cheap" math updated) and (b) calibrates the §5.5 per-run throughput numbers with a real
   measurement on this codebase. Recheck episode adequacy for hopper/halfcheetah similarly
   (a short sac run vs a reference return).

**Success:** TRM statistically ≥ both baselines on ≥1 MuJoCo env, or ≥ pendulum.

---

### H2 — Recursive depth improves long-horizon / delayed-search retention

**Question:** Holding the backbone type fixed, does more recursion (`H_cycles`/`L_cycles`)
help as the effective horizon grows (controlled by `delay`)?

**Tasks for the coding-manager:**

1. **Depth/delay matrix** on `model=trm`:

   | H_cycles | L_cycles | delay | env | seeds |
   |----------|----------|-------|-----|-------|
   | 1, 2, 3 | 6 | `none`, `fixed`, `stochastic`, `time_varying` | pendulum | 5 |

   (Overrides: `model.trm.H_cycles`, `model.trm.L_cycles`; e.g.
   `model=trm model.trm.H_cycles=1 delay=stochastic`.)
2. **Follow-up on MuJoCo** with a fixed moderate delay (e.g. `fixed`, value tuned in #1)
   comparing `trm` at H_cycles 1 vs 3 vs deeper, if #1 shows a monotonic-but-flattening trend.
3. **Optional strength:** sparse-reward env variant for pendulum/hopper wrapping dense
   reward into a sparse terminal/binary signal.

**Success:** Return rises with depth under delay and flattens (diminishing returns) without
delay, confirming depth buys long-horizon robustness rather than raw-control performance.

---

### H3 — Deep supervision during recursive rollout stabilizes policy learning

**Question:** Adding a weighted auxiliary loss at intermediate recursive steps improves
stability/final performance vs final-only supervision.

**Tasks for the coding-manager:**

1. **Expose intermediate latents** — extend `TRMTrunk` (additive, contract-compatible) to
   optionally emit the latent at selected recursion checkpoints (e.g. steps `{1,3,5}` of the
   final H-cycle), gated by a config flag so default behavior is unchanged.
2. **Aux loss in the trainer** — in the PPO update, compute a policy/value auxiliary loss at
   each supervised checkpoint with configurable weights (`conf/model/trm.yaml` additions:
   `deep_supervision: null | {steps: [...], weights: [...]}`). Full-gradient, not truncated.
   Guard so `model=dnn`/others are unaffected.
3. **Ablation matrix** on `arch=ppo model=trm env=pendulum delay=none`:

   | deep supervision | seeds |
   |------------------|-------|
   | off (final only) | 5 |
   | every step | 5 |
   | steps `{1,3,5}`, weights `[0.1,0.3,0.6]` | 5 |

   Metrics: final return, seed variance, convergence speed (steps to 80% of final).

**Success:** Intermediate supervision ≥ final-only on variance and/or speed, with no
significant regression in final return.

---

### H4 — TRM policies transfer better to novel dynamics

**Question:** A policy trained on the default dynamics maintains more performance when
dynamics change, compared to the MLP/transformer baseline.

**Tasks for the coding-manager:**

1. **Dynamics wrapper** — an env wrapper that applies controllable perturbation at
   reset/eval (gravity scale, friction scale, and optionally morphology), implemented in
   `src/` and activated via `conf/env/<name>.yaml` or a top-level `dynamics=` override.
   Do not modify the trainer.
2. **Train** `trm` and `dnn` (and `transformer` if it ships) on `env=hopper delay=none`,
   5 seeds each.
3. **Evaluate zero-shot** on variants: gravity ∈ {0.5×, 1.0×, 2.0×}, friction ∈ {0.5×, 1.0×,
   2.0×}, morphology (torso length) ∈ {0.8×, 1.0×, 1.2×}. Report return per variant, and %
   retention vs the default-dynamics eval.
4. **Fine-tune** each on the hardest variant for a short budget (e.g. 100k steps) and report
   the adaptation curve.

**Success:** TRM retains ≥70% of original performance on held-out variants and ≥15% better
retention than the best baseline.

---

### H5 — Persistent high-level latent (`h`-carry), `y` on the fast latent

**Origin:** collaborator's control-variant proposal (kept, adapted; the supervised/test-time
search-tree piece was **discarded** as out of scope for v0). Two concrete architectural ideas
that the current trunks do not express:

1. **Second recursive variant — `y` on the fast `l`, `h` mediates.** In puzzles `y` (output)
   sits on the slow state; in control, attach the output to the **fast `l`** state and let the
   slow `h` be a longer-horizon planner that mediates `l`. This is a distinct trunk from the
   shipped stateless `trm` (which reads out `z_H`).
2. **Persistent `h` carry + explicit backprop-of-`h` mode.** `h` persists across successive
   observations (acting as "smoothing" of a stale/noisy input), while `l` re-derives from the
   current `x` each tick. Because `h` is held as a "constant-with-context" (like HRM-Agent),
   and because `y` changes each `l` tick, we can backprop to `l` at every tick
   (`backprop_h: one_step`) vs running full BPTT through all cycles (`backprop_h: full`).
   This directly tests a published, unresolved disagreement (HRM: `h` converges → 1-step;
   TRM: full BPTT) in the continuous-control regime.

**Tasks for the coding-manager:**

1. **Persistent-carry trunk** — add `TRMCarryTrunk` (`conf/model/trm_carry.yaml`):
   - `.forward` reads out from the fast latent (y-on-l); `h` holds state **across calls**
     (carried in `hide_prev` and reset at episode boundaries / by the trainer), violating the
     current "stateless per call" convention — this must be documented and handled by the
     rollout machinery.
   - `h_backprop: {full, one_step}` config. `full` = full gradient through all cycles (TRM);
     `one_step` = hold `h` as context, backprop to `l` only (HRM/HRM-Agent).
2. **Carry plumbing across algorithms** (the hard part): persistent `h` interacts with
   batching.
   - **PPO** (`OnPolicyTrainer`): rollout is sequential — natural carry; maintain `h` per env
     step, reset on `done`.
   - **DDPG/TD3/SAC** (`OffPolicyTrainer`): replay samples **random minibatches**, so each
     stored transition must carry its own `h` snapshot (and likely its `prev_h`) in the buffer,
     then feed it through the trunk on replay. This is new data plumbing in all four algos and
     their `RolloutBuffer`/`ReplayBuffer`; keep the change minimal and guarded by the model type.
3. **Observation-staleness wrapper** (shared with H2 — see §3 shared automation): a wrapper
   that feeds the agent an observation that is `n` steps old, independent of action delay.
4. **Ablation matrix** (on `arch=ppo model=trm_carry env=pendulum`), isolating the two ideas.
   Split into **unconditional interpretability cells** and **headline-gated sibling ablations**
   (see §6 decision #9):

   **Unconditional (must-run regardless of outcome — interpretability controls; H5's results
   are meaningless without them):**

   | condition | h_backprop | staleness (obs/act) | notes |
   |-----------|-----------|---------------------|-------|
   | stateless `trm` | — | none | baseline (carry off) |
   | **LSTM recurrent baseline (matched size)** | — | none | **must be added** — without it a `trm_carry` win reads as "memory helps", which an equal-size LSTM would reproduce |
   | V1 y-on-l, persistent `h` | `one_step` | none | |
   | V1 y-on-l, persistent `h` | `full` | none | |
   | **V1 y-on-l vs y-on-h readout control** | `one_step` | none | isolate readout placement from the carry itself |
   | V1 y-on-l, persistent `h` | `one_step` | obs stale (tuned) | |
   | V1 y-on-l, persistent `h` | `one_step` | act stale (fixed/delay) | |
   | V1 y-on-l, persistent `h` | `full` | obs stale | |

   Plus: the rollout machinery must correctly reset `h` across episodes (checked in all cells).

   **Headline-gated (Family-B sibling ablations — implement behind config, but only schedule
   after the pendulum smoke cell shows `trm_carry` ≥ the LSTM baseline; see §6 decision #9):**
   - **H5-depth**: does recursion depth inside the carry trunk help under delay? (mirror of H2)
   - **H5-deep-supervision**: does the aux-loss mechanism transfer to the stateful trunk? (mirror of H3)
   - **H5-transfer**: does the persistent carry transfer better to novel dynamics? (mirror of H4)

   Re-run the key conditions on `hopper` once MuJoCo is up. Metrics as H1 (mean/std/CI), plus
   the rollout machinery must correctly reset `h` across episodes.

**Success:** The `h`-carry variant (esp. with `one_step` backprop and/or observation
staleness) matches or beats the stateless `trm`, the **LSTM baseline**, and the MLP baseline on
long-horizon/stale inputs; and `full` vs `one_step` differ measurably, resolving the
HRM-vs-TRM backprop question in control.

**Scope note:** the collaborator's supervised search-tree / "V ⇒ trajectory R" idea is
**explicitly deferred** to future work — v0 tests only the architectural/backprop claims.

---

## 3. Shared automation (needed by all experiments)

Implement once, use everywhere:

1. **Eval script** — fixed-seed, exploration-off evaluation of a checkpoint producing mean/
   std return over N episodes. Integrate as `pyscripts/evaluate.py` or a `main.py` eval mode.
2. **Sweep controller** — a small script (e.g. `scripts/sweep.py`) that takes a config grid
   (model×env×delay×seed) and emits + submits the Slurm batch jobs per the template in
   `AGENTS.md` §6.4, or runs locally in sequence when GPUs are available on the node.
3. **Analysis script** — pulls wandb/CSV results, computes per-condition mean/std, 95%
   bootstrap CI, Cohen's d, paired t-test where applicable, and writes `docs/` markdown
   tables + figures.
   Apply the same stats to H5's conditions (including the `full` vs `one_step` backprop
   comparison).
4. **Staleness wrappers** — a) **action staleness** (already shipped as `DelayedEnv`/
   `conf/delay/*`); b) **observation staleness**: a new wrapper that serves the agent an
   observation that is `n` steps old, independent of action delay. Both are independent env
   parameters consumed by H2 and H5; expose a `conf/delay/`-style group (e.g.
   `conf/staleness/*` or `env.staleness_obs`) so a config can set obs-stale and act-stale
   independently.
5. **Logs cleanup** — ensure `logs/`, `results/` are gitignored; `.agg` tables in `results/`.

All of these follow the HPC template (module → venv → single-process torchrun); none introduce
distributed training.

---

## 4. Suggested implementation order (for kick-off scoping)

| Order | Deliverable | Depends on |
|-------|-------------|-----------|
| A | Sweep controller + eval script + analysis script (automation backbone) | — |
| A2 | Observation-staleness wrapper (additive to existing action-delay) | A |
| B0 | **Training-parity / cost calibration** (same pendulum cell under ppo AND sac; wall-clock + return; verify off-policy episode adequacy) | A |
| B | H1 current-state repro on pendulum (trm vs dnn, ppo, 5 seeds) | B0 |
| C | MuJoCo envs + deps | B |
| D | H1 transformer trunk | B, C |
| E | H1 main matrix (ppo headline + sac replication) | C, D |
| F | H2 depth/delay matrix (pendulum → hopper) | C |
| G | H3 deep-supervision mechanism + ablation | C |
| H | H4 dynamics wrapper + train/eval/fine-tune | C, D |
| I | **H5** persistent-carry trunk + backprop-mode + carry plumbing → ablations (pendulum → hopper) | A2, C |

H5 depends on the staleness wrapper (A2) and MuJoCo (C). Each stage produces committed code +
runbook + report before moving on. **B0 gates every later stage** — its throughput/parity
measurements drive the actual cost decisions (which cells run in sac vs ppo). A human can
approve/skip stages between A and I as compute allows. Within H5, the **unconditional
interpretability cells** (incl. the LSTM baseline and y-on-h readout) always run; the
**Family-B sibling ablations** (depth / deep-supervision / transfer) are only scheduled after
a positive pendulum headline (see `model=trm_carry ≥ LSTM baseline`, §6 decision #9).

---

## 5. Hypotheses → experiments summary (one-liners for humans)

| H | Experiment | Env(s) | Compare | Budget (per cond) |
|---|-----------|--------|---------|-------------------|
| H1 | Backbone comparison | pendulum → hopper, halfcheetah | trm vs dnn vs transformer × ≥5 seeds | ppo headline + sac replication |
| H2 | Depth × delay/staleness | pendulum → hopper | H_cycles {1,2,3} × delay/staleness {none,fixed,stoch,tv} | 5 seeds |
| H3 | Deep-supervision ablation | pendulum | off vs every vs {1,3,5} | 5 seeds |
| H4 | Dynamics transfer | hopper | trm vs dnn (zero-shot + fine-tune) | 5 seeds |
| H5 | Persistent `h`-carry + backprop-of-h | pendulum → hopper | trm_carry (y-on-l) vs stateless trm; **vs LSTM baseline**; full vs one_step; y-on-l vs y-on-h readout | 5 seeds (Family-B sibling ablations: headline-gated) |

---

## 5.5 Cost estimates (H100-hours)

**Goal.** Every experiment carries a planning H100-hour budget so we can trade cost vs
value and decide later which configs/sweeps to keep or cut. These are **estimates to be
calibrated** with real wall-clock from the sweep logs (the code must log steps/sec and
wall-clock per run so this model stays current).

**Assumptions** (single quantum/GH200-class GPU ≈ one H100):
- Horizon: `pendulum` 200 steps; MuJoCo 1000 steps (default `-v2` truncation).
- Budgets: PPO **2000 episodes** (on-policy, needs more samples → 400k steps pend / 2M steps
  MuJoCo); DDPG/TD3/SAC **200 episodes** (40k / 200k steps).
- Throughput (env-steps/sec, conservative): off-policy ~4k (pend) / ~1.2k (MuJoCo); on-policy
  PPO ~1.5k (pend) / ~350 (MuJoCo), +15% for PPO's multiple update epochs.
- Per-run cost ≈ `steps / throughput / 3600` (H100-hr). Small models (DNN, TRM) are compute
  negligible vs env stepping; the transformer/H5-carry add FLOPs but not wall-clock-dominant.

**Per-run H100-hr (single seed):**

| Condition | steps/run | est. H100-hr/run |
|-----------|-----------|------------------|
| pendulum · PPO | 400k | ~0.09 |
| pendulum · DDPG/TD3/SAC | 40k | ~0.01 |
| hopper/halfcheetah · PPO | 2M | ~1.8 |
| hopper/halfcheetah · DDPG/TD3/SAC | 200k | ~0.05 |
| ant · PPO | 2M | ~2.2 |
| humanoid · PPO | 2M | ~2.8 |

**Full-matrix totals (5 seeds), carrying the cost of every v0 condition:**

| Experiment | runs (5 seeds) | env · algo | est. H100-hr |
|-----------|----------------|-----------|--------------|
| H1 smoke (trm|dnn|transformer × pend · PPO, 3 cond) | 15 | pend · PPO | **~1.4** |
| H1 main — SAC replication (trm|dnn|transformer × hopper|halfcheetah) | 30 | MuJoCo · SAC | **~1.5** |
| H1 main — PPO headline confirm (trm vs dnn × hopper) | 10 | MuJoCo · PPO | **~18** |
| H1 alt algo (sac, trm|dnn × hopper) | 10 | MuJoCo · SAC | **~0.5** |
| H2 (trm: H∈{1,2,3} × delay{4} × stal{2} × pend · PPO) | 120 | pend · PPO | **~11** |
| H2 follow-up (trm H∈{1,3} × 2 delays × hopper · SAC, +1 PPO confirm) | 20 | MuJoCo · SAC+PPO | **~2** |
| H3 (trm × 3 sup modes × pend · PPO) | 15 | pend · PPO | **~1.4** |
| H4 (trm|dnn × hopper · SAC train + eval + fine-tune, +PPO confirm) | ~20 | MuJoCo · SAC+PPO | **~2** |
| H5 (carry on/off × 2 bprop × 2 stale × pend · PPO) | 40 | pend · PPO | **~3.6** |
| H5 interpretability controls (LSTM baseline + y-on-h readout × pend · PPO) | 10 | pend · PPO | **~0.9** |
| H5-Family-B sibling ablations (depth / deep-supervision / transfer) | *gated* | — | **0 unless triggered** (headline-gated, §6 decision #9) |
| H5 follow-up (carry × 2 bprop × hopper · SAC replication, +1 PPO confirm) | 20 | MuJoCo · SAC+PPO | **~2** |

**Cost-reduction strategy (delivered).** PPO on MuJoCo (H1 main, H2/H4/H5 follow-ups) was the
dominant ~180–190 h line. Under principle #9 and the B0 calibration, we flip the **heavy MuJoCo
matrices to SAC replication** and keep PPO only for (a) the mechanism hypotheses' pendulum
sweeps (cheap), (b) the H1 headline cell, and (c) one short PPO confirm per headline result.
SAC on the same MuJoCo configs is ~35× cheaper per run, cutting the MuJoCo bill to a rounding
error; the delayed/stale sweeps remain in PPO on pendulum. **Revised v0 total ≈ 40–60
H100-hours** (dominated by PPO pendulum sweeps H2/H5 and the H4 transfer runs) — a ~70–75%
reduction. Exact split is finalized after B0 measures real throughput and parity.
The added H5 interpretability controls (LSTM baseline + y-on-h readout, ~0.9 H100-hr on
pendulum) are inside this range; the Family-B sibling ablations add **0 cost until/unless**
triggered by a positive H5 headline (§6 decision #9).

**Guarding the cut.** The SAC-replication route only saves money if off-policy 200-episode
runs are actually sufficient (verified in B0). If Hopper/HalfCheetah need more episodes, the
"savings" shrink — B0's parity check is the gate. The exact SAC vs PPO allocation is finalized after B0; see §6 (deferred decision 1).

---

## 6. Decisions: resolved vs deferred

**Decided (per collaborator review, 2026-07-23):**

- **H5 carry plumbing — in scope (OFF-POLICY INCLUDED).** `TRMCarryTrunk` must handle
  per-transition `h` snapshots in the DDPG/TD3/SAC replay buffers, not just PPO's sequential
  carry. (Was open decision #7.)
- **H5 `full` vs `one_step` backprop — included in code AND in the run schedule.** The
  ablation runs on both the stateless `trm` and the carry variant where informative.
  (Was open decision #8.)
- **H5 must include an LSTM recurrent baseline of matched size and a y-on-l vs y-on-h
  readout control** as unconditional interpretability cells (see §H5 #4). Without them,
  a `trm_carry` win would be indistinguishable from "memory helps" (which an equal-size
  LSTM would reproduce) and the readout-placement effect would be unmeasurable.
  (Decision #9, new.)
- **H5's Family-B sibling ablations (depth, deep-supervision, transfer — mirrors of H2–H4)
  are headline-gated.** Implement behind config flags (§6 principle), but only schedule
  runs after the pendulum smoke cell shows `trm_carry ≥ LSTM baseline`. Deferred until
  then to avoid spending compute interpreting a negative result. (Decision #9, new.)

**Deferred — all are parameter/configuration choices, not architecture decisions.** The code
must implement the full surface for each so switching is a config change, not a code change.
The cost estimates in §Cost exist to help decide these later:

1. **PPO vs SAC allocation** — guided by principle #9 and §5.5: PPO is the mechanism +
   headline standard; SAC (and other off-policy algos) replicate and carry the heavy MuJoCo
   cells. The exact per-cell split is finalized after Stage B0's parity/throughput
   measurement. Both algos' configs ship and every grid is algo-parameterizable.
2. **MuJoCo env budget** — `hopper` + `halfcheetah` are the default cheap pair; `walker2d`,
   `ant`, `humanoid` configs must ship and are one grid line away.
3. **Seed budget** — default **5**; raising to 10 doubles cost and is a grid param.
4. **Delay/staleness values for the "horizon" dial** — start from shipped defaults
   (`fixed=1`, `stochastic` uniform `[10,500]`, `time_varying` schedule as-shipped) and tune
   in H2 #1; values are pure config.
5. **Sparse-reward variants** — H2 #3 "optional strength"; implement the wrapper behind a
   flag so it can be included or cut by config.
6. **Transformer trunk** — in scope for H1 stretch; shipped behind `transformer.yaml` and can
   be left out of a grid without code change.

---

## 7. Definition of done (per stage)

Per stage the coding-manager must commit: (a) code, (b) conf YAML, (c) runbook entry
(`docs/RUNBOOK.md`), (d) report with real numbers and explicit "not run" flags, (e) cleanup
(`logs/`/`results/` gitignored). No fabricated data.

---

*End of plan v0.2. Awaiting human approval before kick-off.*