# Phase 4 Plan: Complexity Rung, Training Discipline, LR Sweep

*trm-delay-robust-scaling · pre-registration · frozen 2026-10-06, before any Phase 4 data*
*Decision rules are machine-readable in `../../experiments/phase4/gates.json`. Any later change is recorded in §10 as a dated amendment, before the data it affects.*

## 0. Why Phase 4

Phases 0–3 established three things, all on Pendulum-v1 with DDPG: the TRM deficit is architectural, not capacity; recursion depth is inert, because only 7 of 21 cell applications receive gradient; and carrying the latent adds nothing. Two questions remain open:

- **Complexity scaling (H1) is untested.** MountainCarContinuous and BipedalWalker were floor effects, so no rung above Pendulum was ever learned by the baseline.
- **TRM ran without its training discipline.** The port truncates gradient to the last outer cycle and supervises only the final readout.

Phase 4 has three jobs, in priority order:

| Priority | Stage | Question |
|---|---|---|
| 1 | **B** | Add one complexity rung that the MLP actually learns, then test whether TRM's relative deficit shrinks on it (H1′). |
| 2 | **A** | Does restoring the training discipline (full gradient, deep supervision) change the TRM result? |
| 3 | **C** | Rule out "TRM was never tuned" with a small LR sweep. |

Stage E (Lipschitz mechanism analysis on existing checkpoints) is exploratory. Stage D support (C-header export) is engineering only. Neither has a gate.

**Budget:** three weeks, 2026-10-06 to 2026-10-26. The author makes every gate decision; execution stops and reports at each gate, at the kill switch, and before every main matrix (B1, A), showing the job count and estimated walltime.

## 1. Ground rules (carried over from Phase 3)

- **Frozen code per stage.** Each stage runs from a frozen snapshot (`p4code/<stage>` on Katana). There is a job ledger (`~/jobs/phase4/ledger.jsonl`) and a live status JSON (`p4.py status`). Everything runs as PBS jobs on Katana GPU nodes, never on the login node.
- **No changes to existing code paths.** New behaviour sits behind new, default-off flags:
  - `arch.total_steps` (env-step budget);
  - `model.full_grad`, `model.deep_sup` (TRM trunk);
  - `arch.actor_grad_clip`, `arch.critic_grad_clip` (DDPG).

  Existing configs must stay bit-identical to the pre-Phase-4 code (§5).
- **Statistics.**
  - Pairs are matched by seed. Sign tests: one-sided where a gate has a direction, with two-sided p reported alongside.
  - Gates pass at ≥ 5/6 seeds, or ≥ 10/12 when pooled.
  - Battery returns are reported as IQM with a stratified bootstrap CI (Agarwal et al., 2021).
- **Primary metric.** Final avg-100 *training* return, as in Phases 1–3. Deterministic evaluation (the last 5 `eval_return`s, and the battery where it runs) is secondary.
- **Walltime.** Before any main matrix, per-run walltime is estimated from pilot timings. Any arm likely to exceed its tier is flagged to the author with options (H = 1, a budget change, a longer tier); this plan does not decide that.

## 2. Code added for Phase 4

| Flag / file | What it does | Default |
|---|---|---|
| `arch.total_steps` (`src/trainer.py::_episodes`) | Env-step budget. No new episode starts once it is reached; the last episode runs to its end. Needed because IDP episode length grows with skill (≈20 steps failing, 1000 solved). | unset → episode budget, unchanged |
| `model.full_grad` (`TRMTrunk.forward_cycles`) | No one-step truncation: backprop through all H × (L + 1) = 21 cell applications. | false |
| `model.deep_sup` (`Critic.forward_cycles`, `DDPGAgent.update`) | Critic readout after **every** outer cycle. Critic loss = mean over cycles of MSE(Q_k, y), against the same target-network TD target y (target critic's final readout). Requires `full_grad`. The actor uses the final cycle's action, and its loss uses the critic's final readout. | false |
| `arch.{actor,critic}_grad_clip` (`DDPGAgent._clip`) | `clip_grad_norm_` per network, before the optimiser step; pre-clip norms are logged per update. `inf` logs without clipping. | unset → no clipping, no logging |
| `conf/model/trm_fullgrad.yaml`, `trm_fullgrad_ds.yaml` | `trm.yaml` + the flags above (distinct `name`, so results files never collide). | — |
| `experiments/phase4/p4.py`, `sync.sh` | Job manager (manifests, submit, status, resubmit), frozen snapshots. | — |
| `experiments/phase4/verify_code.py` | Gradient-hook test, bit-identity suite. | — |

## 3. Stage B0 — complexity pilot (day 1, highest priority)

**Setting.** InvertedDoublePendulum-v5, TD3 (`conf/arch/td3.yaml`, unchanged). Condition C1: fixed d = 1 + pending-action queue, no previous-action input (as in Phase 3 Stage 1).

**Reference return.** **9100**, the `reward_threshold` Gymnasium registers for InvertedDoublePendulum-v5 (`gymnasium/envs/__init__.py`, Gymnasium 1.3.0; checked in the Katana venv). For context: the maximum achievable return is ≈ 9360 (alive bonus 10 × 1000 steps, minus small penalties), and Fujimoto et al. (2018) report 9337 for TD3.

**"Learned"** means final avg-100 training return ≥ **8190** (90% of 9100).

**Arms.**

| Arm | Budget | Seeds | Jobs |
|---|---|---|---:|
| MLP 135k (`dnn`) | 150k, 300k, 500k env steps | 1–3 | 9 |
| TRM 788k (`trm`, H = 3), timing only | 15k steps (10% of the smallest budget) | 1 | 1 |
| TRM H = 1 (`trm_h1`), timing only | 15k steps | 1 | 1 |

**Budget rationale.**
- TD3 learning curves on IDP (Fujimoto et al., 2018) reach the threshold at roughly 100–200k steps.
- 150k sits at the published solve point, 300k gives 2× margin for the delay and for the avg-100 window (the last 100 solved episodes alone are ≈ 100k steps), and 500k is the most a TRM arm could plausibly afford in B1.
- Hyperparameters do not depend on the budget, so the three budgets are nested prefixes of one another for a given seed. Comparing a 500k run's avg-100 at 150k with the 150k run's final value is a free GPU-reproducibility check (reported, not gated).

**Timing.** Updates start once the buffer holds one batch (256 transitions), not at the end of warmup. A 15k-step run therefore times ~15k updates, enough to extrapolate per-step cost to every budget.

> **Gate GB0.** The MLP is learned in ≥ 2/3 seeds at some budget. The B1 budget is the **smallest** budget that passes.
> Reported alongside: deterministic eval (last 5), steps to first avg-100 ≥ 8190, and extrapolated TRM / TRM H = 1 walltime at the chosen budget.

> **Kill switch KB0.** If GB0 has not passed by **2026-10-09** (day 4), repeat B0 on **Hopper-v5** + TD3 (same budgets and seeds). Hopper's reference is its registered `reward_threshold`, 3800, so "learned" means ≥ 3420 (the 90% rule unchanged). If that also fails, Stage B stops and is reported to the author. *Note:* Fujimoto et al. (2018) report ≈ 3564 for TD3 on Hopper after 1M steps. 3420 at ≤ 500k steps of noisy training return may be out of reach. The author may amend this reference before any Hopper data exists.

## 4. Stage C — LR sweep (runs alongside B0)

**Setting.** Pendulum-v1, DDPG, condition C1 as in Phase 3 Stage 1 (no previous-action input), so a flagged LR feeds directly into a rerun of the Stage 1 contrast. TRM 788k (`trm`, H = 3), 200 episodes.

**Grid.** Full 3 × 3: actor LR × {0.3, 1, 3} (default 1e-4, so 3e-5 / 1e-4 / 3e-4) × critic LR × {0.3, 1, 3} (default 1e-3, so 3e-4 / 1e-3 / 3e-3). Seeds 1–3. **27 jobs.** The ×1/×1 cell is rerun in the same snapshot rather than reusing Stage 1, so all nine cells share one code snapshot.

> **Gate GC.** For each of the 8 non-default cells, per seed s: Δ(s) = final(cell, s) − final(×1/×1, s). The cell is **flagged** if Δ(s) exceeds the pooled seed SD of the two cells in **all 3 seeds**.
> - If any cell is flagged, it is reported to the author, who decides whether to rerun the Stage 1 contrast TRM 788k − MLP 788k at that LR.
> - Otherwise the result is reported as "TRM insensitive to LR within ×0.3–×3".
>
> Reported for every cell: median Δ, k/3, late critic loss (median over the last quarter of episodes, as Phase 3), and deterministic eval (last 5).

**Caveat (stated now).** The gate tests eight cells at n = 3. Under the null, a cell passes the sign part alone with probability 1/8, so the magnitude condition carries the weight. A flag is grounds for a confirmatory rerun, not a finding.

## 5. Verification before any Stage A or B job

1. **Gradient-hook test** (`verify_code.py grad`). Hooks on every reasoning-cell application count those whose output receives non-zero gradient. Required: **7 of 21 without `full_grad`; 21 of 21 with it** (H = 3); also checked at H = 1 and H = 6.
2. **Bit-identity** (`verify_code.py run` + `compare`). Every existing configuration is trained for 2 Pendulum episodes (≈ 300 updates) on CPU with fixed seeds, under both the pre-Phase-4 snapshot and the Phase 4 tree. The configurations are:
   - DDPG and TD3 × {dnn, dnn_large, trm, trm_small, trm_h1, trm_h6, trm_hier6, trm_singlez, transformer, lstm, trm_carryz, lstm_carry} at fixed d = 1 + queue;
   - DDPG × {dnn, trm, trm_carryz} at no delay and at C4 + previous action.

   The pre-Phase-4 snapshot is identical to Katana's frozen `p3code/s2x`. Required: identical final-weight hashes and identical episode histories (returns, losses, eval returns) for every configuration.
3. **New paths run** (`verify_code.py run --which flags`):
   - clip = inf reproduces the unclipped run exactly;
   - `trm_fullgrad` and `trm_fullgrad_ds` train;
   - `total_steps` ends a run at its budget.
4. **GPU smoke job** (`p4.py submit smoke`). One short job on a GPU node covering IDP + TD3, the step budget, an LR override, and `trm_fullgrad_ds` with clipping at C4.

## 6. Stage A — training discipline (implement week 1, run week 2)

**Arms.** Pendulum-v1, DDPG, 200 episodes, conditions **C1** and **C4**, both with `prev_action_obs=true` (matching Phase 3 A-6), seeds 1–6:

| Arm | Config | Clipping |
|---|---|---|
| TRM-trunc (baseline) | `trm` | yes |
| TRM-fullgrad | `trm_fullgrad` | yes |
| TRM-fullgrad + deep_sup | `trm_fullgrad_ds` | yes |
| MLP 135k | `dnn` | no (not a TRM arm) |

**Clip norms (set before Stage A data).** A pilot (`p4.py submit a0`, 4 jobs, seed 1) runs TRM-trunc at C1 and C4 with clip = inf, which logs pre-clip norms without clipping. It also runs TRM-fullgrad and TRM-fullgrad + deep_sup at C1, for timing and norm diagnostics.
- The actor and critic clip norms are each set to the 99th percentile of TRM-trunc's pooled per-update pre-clip norms (C1 + C4), rounded to 2 significant figures.
- They are recorded in `gates.json` (`frozen_values.stage_a_clip`) by a dated amendment before the Stage A matrix is submitted.
- The same norms apply to all three TRM arms. Each arm's clip-active fraction is reported, because full-gradient arms may have larger norms and be clipped more often.

**Reuse of Phase 3 runs.**
- **TRM-trunc is rerun** (12 jobs), because clipping makes it a different configuration from Phase 3's Reset-Z.
- **The MLP arm reuses** Phase 3 Stage 2's `dnn` C1/C4 runs **only if** the code-path equivalence check passes for those exact configurations:
  - §5.2 must show bit-identity for DDPG + `dnn` at C1 and C4 with the previous action;
  - Stage 2's frozen snapshot `p3code/s2` must be identical to `p3code/s2x` on every file the MLP path imports;
  - the Hydra overrides must differ only in logging fields (`print_stat_every`, `record_every`, output folders).
- Otherwise the 12 MLP runs are rerun.

**Jobs.** 36 TRM (+12 MLP if reuse fails), 48 in total.

> **Gate GA (primary).** For D ∈ {TRM-fullgrad, TRM-fullgrad + deep_sup} and c ∈ {C1, C4}, per seed, Δ = final(D) − final(TRM-trunc). GA passes if **any** of the four contrasts is positive in ≥ 5/6 seeds.
> *Multiplicity:* four contrasts at one-sided p = 0.109 each. A pass names the contrast(s) that carried it.

> **Secondary (reported whatever GA's outcome).**
> - D − MLP per condition (k/6, median, two-sided p).
> - Late critic-loss ratios D / TRM-trunc and D / MLP, last quarter of episodes, final-cycle loss.
> - Deterministic eval (last 5).
> - Per-decision latency: inference is unchanged by these flags, which affect only training. This is verified by timing, not assumed.
> - Training cost as wall-clock within a GPU model.

## 7. Stage B1 — complexity rung main matrix (week 2, after GB0 and author approval)

**Arms.** GB0's environment and budget, TD3, C1:

| Arm | Config |
|---|---|
| MLP 135k | `dnn` |
| MLP 788k | `dnn_large` |
| TRM 788k (H = 3) | `trm` |
| TRM H = 1 | `trm_h1` |

Seeds 1–6, **24 jobs.** Walltime per arm comes from the B0 timing runs. Any arm over the 2 h tier is flagged to the author before submission (§1).

**Span normalisation.** On environment E:
- untrained_E = mean return of a uniform-random-action policy, 100 episodes, under the same C1 wrapper, computed before B1 data;
- best_E = the highest final avg-100 of any run in the comparison set. That set is Stage 1 fixed d = 1 + queue (all six arms) on Pendulum, and all B1 runs on the new rung;
- span_E = best_E − untrained_E.

> **Pre-registered scaling test H1′.** Per seed s, the normalised deficit is
> δ_E(s) = [final_MLP135k(s) − final_TRM788k(s)] / span_E
> (positive when TRM trails).
> - On Pendulum, this uses Phase 3 Stage 1 `dnn` and `trm` at fixed d = 1 + queue, seeds 1–6.
> - On the new rung, it uses B1's `dnn` and `trm`, seeds 1–6.
>
> **H1′ (TRM's relative deficit shrinks with complexity) is supported only if δ_new(s) < δ_Pendulum(s) in ≥ 5/6 seed pairs.**
> Secondary, the same statistic for: TRM 788k vs MLP 788k; TRM H = 1 vs MLP 135k.

**Threats recorded now.**
1. The rungs differ in algorithm (Stage 1 is DDPG, B1 is TD3) as well as in complexity. H1′ cannot separate the two.
2. Pairing seed s across environments is an arbitrary pairing. The sign test is then a test of whether one distribution sits below the other, not a within-unit contrast.

**Battery.** Every B1 checkpoint runs through the Phase 3 evaluation battery (`w0_eval.py`) where applicable:
- deterministic nominal evaluation (IQM + bootstrap CI);
- sensor noise;
- actuator dropout;
- delay mismatch;
- control-quality metrics.

Plant mass/length scaling is Pendulum-only and is skipped.

## 8. Stage E — exploratory mechanism analysis (week 3, no training)

For every Phase 3 Stage 1 checkpoint (72):
- Estimate the actor's local Lipschitz constant: the spectral norm of ∂a/∂s, by autograd, at every state visited in 20 deterministic evaluation episodes (the battery's unseen seeds). Report the median and 95th percentile per checkpoint.
- Correlate it (Spearman, across checkpoints, and within trunk family) with direction reversals/s and with the sensor-noise (σ = 0.05) return drop, both from the Phase 3 battery.

**Labelled exploratory throughout.** No gate.

## 9. Stage D support — embedded export (low priority)

A script will export the actor weights of MLP 135k, TRM 136k (H = 3) and TRM H = 1 to float32 C headers. It also writes 20 reference input/output vector pairs from PyTorch for an on-device equivalence test. For each actor it reports the parameter count and the FP32 / INT8 weight size against a 512 KB flash budget. The STM32 side is the author's.

## 9b. Stage T — G1 under TD3 (added 2026-10-07, amendment T-1, before data)

**Question.** Does Phase 3's central attribution (the TRM deficit is architectural, not capacity) hold under a second learning algorithm? Every Phase 1–3 contrast used DDPG.

**Design.** Mirrors Phase 3 Stage 1, with TD3 in place of DDPG:
- Pendulum-v1, `arch=td3` with `conf/arch/td3.yaml` unchanged (no tuning for any trunk), 200 episodes;
- conditions: no delay, and C1 (fixed d = 1 + queue, no previous-action input);
- trunks: `dnn` (135k), `dnn_large` (788k), `trm_small` (136k), `trm` (788k);
- seeds 1–6, so **48 jobs**.

Parameter labels are the DDPG-basis configs. Under TD3, each total includes twin critics.

> **Gate GT.** Pairs are (seed, condition), n = 12. G1 reproduces under TD3 if TRM 788k − MLP 788k (final avg-100) is negative in **≥ 10/12** pairs. The two-sided sign-test p is reported.
> *Secondary:* G1's other contrasts (TRM 136k − MLP 135k, MLP 788k − MLP 135k, TRM 788k − TRM 136k; "different" = ≥ 10/12 share a sign), late critic-loss ratios at matched capacity, eval last-5, and a side-by-side with Phase 3 Stage 1 (DDPG).

**Threat.** `td3.yaml`'s 10k-step warmup is 50 of the 200 episodes, which shortens learning time for every arm alike.

## 10. Amendments

All amendments are recorded in `gates.json` (`amendments`), each dated and made before the data it affects. In summary:

- **R-1 (2026-10-06).** The Stage A MLP-reuse check passed (bit-identical to Phase 3's `p3code/s2` snapshot), so the 12 MLP runs are reused from `phase3/results/results_p3s2`.
- **R-2 (2026-10-06).** untrained_E frozen for the H1′ span (uniform-random policy): Pendulum −1163.84, IDP 49.90, Hopper 16.32.
- **V-1 (2026-10-07).** §5 verification passed before any B0/C/A job.
- **A-1 (2026-10-07).** Stage A clip norms frozen from the pilot: actor 15, critic 2800. All earlier runs (Phases 1–3, B0, C) are unclipped, so GA compares clipped TRM arms with each other. Any comparison against the unclipped MLP or Phase 3 also carries that difference.
- **K-1 (2026-10-07).** GB0 failed (0/3 learned at every IDP budget). The author fired kill switch KB0 on day 2. The Hopper-v5 pilot runs as pre-registered, with its threshold unchanged at 3420.
- **K-2 (2026-10-07).** KB0 also failed. The Hopper rung is dropped. An IDP diagnostic runs instead: MLP, TD3, seeds 1–3, C1 at 1M steps and no delay at 500k. It is reported against the unchanged 8190 bar. Amendment K-3 (B1 confirmatory at 1M if it passes; otherwise exploratory with a pre-stated "clear of floor" criterion) goes to the author before any B1 data.
- **T-1 (2026-10-07).** Stage T added (§9b): the Stage 1 contrast under TD3, 48 jobs, gate GT.
- **K-3 (2026-10-08).** B1 is not run and H1′ is recorded as not run. Two reasons. First, no harder rung was learnable by the MLP baseline under the pre-registered conditions: IDP and Hopper both failed, and on IDP the one-step delay is the main obstacle. Second, under `td3.yaml` TRM 788k does not train even on Pendulum (Stage T: 2/12 seeds, critic divergence). The exploratory draft (`K-3_draft.md`) is superseded. H1 remains untested.

## 11. Timeline

| Week | Dates | Work |
|---|---|---|
| 1 | 10-06 → 10-12 | Verification; B0 + C submitted; A implemented and verified; A pilot; `gates.json` frozen (clip norms by amendment); GB0 (kill switch on 10-09) |
| 2 | 10-13 → 10-19 | A and B1 main matrices (each after author approval) |
| 3 | 10-20 → 10-26 | Reruns/buffer only, no new designs; analysis, figures, Stage E, export script |

**Deliverables per stage.** `papers/phase4/stage_X.md`, a short results markdown in the Phase 3 style. Each has tables, figures, the gate outcome and threats to validity. No thesis prose.

### References

Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS · Fujimoto, S., van Hoof, H. & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* ICML (TD3) · Jolicoeur-Martineau, A. (2025). *Less is More: Recursive Reasoning with Tiny Networks.* arXiv:2510.04871 · Lee, C.-Y. et al. (2015). *Deeply-Supervised Nets.* AISTATS · Towers, M. et al. (2024). *Gymnasium.* arXiv:2407.17032
