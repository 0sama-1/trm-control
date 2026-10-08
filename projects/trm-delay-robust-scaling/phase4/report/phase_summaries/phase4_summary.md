# Phase 4 Summary: Complexity Rung, Training Discipline, Tuning and a Second Algorithm

*trm-delay-robust-scaling · phase summary · 2026-10-06 to 10-08 · 154 Katana jobs (incl. 1 smoke job and 6 failed-and-rerun Stage E shards)*
*Labels are defined in `../glossary.md`. Pre-registration: `../../plan/plan.md`; rules and amendments: `../../../experiments/phase4/gates.json`.*

## Questions

Phase 3 left five things open. Phase 4 took them in priority order:

1. **Complexity (B):** is there a harder task the MLP learns, on which TRM's relative deficit shrinks (H1′)?
2. **Training discipline (A):** does restoring TRM's own training (no gradient truncation, per-cycle supervision) change the result?
3. **Tuning (C):** was TRM simply never tuned?
4. **Algorithm (T, added mid-phase):** does Phase 3's central attribution hold under TD3 as well as DDPG?
5. **Mechanism and deployment (E, D):** is TRM's noise sensitivity explained by a sharper input–output map? Do the actors fit a 512 KB microcontroller?

## Methodology

- **Pre-registration.** Every stage's arms and decision rule were written into `plan.md` and `gates.json` before its data. Later changes are dated amendments, each made before the data it affects: R-1, R-2, V-1, A-1, K-1, K-2, T-1, K-3. The author approved every main matrix after seeing a job count and walltime estimate.
- **Code changes behind default-off flags:**
  - `arch.total_steps`, an env-step budget;
  - `model.full_grad`, `model.deep_sup` on the TRM trunk;
  - `arch.{actor,critic}_grad_clip`, gradient clipping with norm logging.
- **Verification before any data (V-1):**
  - a gradient-hook test: 7/21 cell applications receive gradient truncated, 21/21 with `full_grad`;
  - all 30 existing trunk × algorithm × condition configurations bit-identical (weights and histories) to the pre-Phase-4 code;
  - Stage A's MLP reuse checked against the exact Phase 3 snapshot (R-1);
  - one GPU smoke job.
- **Execution.** Katana PBS, every job from a frozen per-stage code snapshot, logged in a ledger, with a live run board.
- **Statistics.**
  - Pairs matched by seed (and condition when pooled to n = 12).
  - Sign tests, with k/n and two-sided p.
  - Gates at ≥ 5/6, or ≥ 10/12 pooled.
  - Final avg-100 training return is primary; eval last-5 and the battery are secondary.

| Stage | Setting | Arms | Seeds / budget | Jobs |
|---|---|---|---|---:|
| B0 | InvertedDoublePendulum-v5, TD3 (`td3.yaml`), C1 | MLP 135k; TRM 788k and TRM H = 1 timing runs | 3 seeds × 150k / 300k / 500k steps | 11 |
| B0h | Hopper-v5, TD3, C1 | MLP 135k | 3 × 150k / 300k / 500k | 9 |
| B0d | IDP, TD3 | MLP 135k at C1 (1M steps) and no delay (500k) | 3 each | 6 |
| C | Pendulum, DDPG, C1 | TRM 788k, actor × critic LR × {0.3, 1, 3} | 3 per cell, 200 episodes | 27 |
| A | Pendulum, DDPG, C1 and C4 with previous action | TRM truncated / full gradient / full gradient + deep supervision (clip 15 / 2800); MLP reused from Phase 3 | 6, 200 episodes | 4 pilot + 36 |
| T | Pendulum, TD3 (`td3.yaml`), no delay and C1 | MLP 135k / 788k, TRM 136k / 788k | 6, 200 episodes | 48 |
| E | 72 Phase 3 Stage 1 checkpoints | Actor Jacobian spectral norm over 20 deterministic episodes | — | 6 + 6 rerun |
| D | 3 seed-1 Stage 1 actors | C-header export, 20 reference I/O pairs | — | local |

## Results

| Gate | Outcome | What it says |
|---|---|---|
| **GB0** | Fail | The MLP never reliably learns IDP with delay: avg-100 2032–4312 against 8190 at every budget. |
| **KB0** | Fail | Hopper is no better: best 2805 against 3420. Hopper dropped. |
| B0d | — | 1M steps doesn't help (0/3). Without delay every seed reaches 8190 at some point but doesn't hold it. The delay on an unstable plant is the main obstacle. |
| **GC** | Insensitive | No learning rate beats the default (×1/×1 median −215) by more than the pooled SD in 3/3 seeds. Raising the actor LR hurts. |
| **GA** | Fail | Full gradient ± deep supervision ties truncated TRM (best 4/6) and still trails the MLP (1/6 each). It costs 1.45–1.5× the training time. |
| **GT** | Pass, with caveat | TRM 788k − MLP 788k negative in 12/12 (p = 0.0005), but because TRM 788k fails to train under TD3's defaults (2/12 seeds learn; its critic diverges). |
| H1′ | Not run | B1 skipped by the author (K-3): no harder rung was learnable, and TRM 788k does not train under `td3.yaml`. |
| E | Exploratory | TRM actors are *less* sensitive locally than MLPs (Lipschitz 11–16 vs 29–34), yet lose 1.6–3× more return under sensor noise. |
| D | Done | MLP 135k and TRM 136k actors fit 512 KB in FP32 (~265 KB). TRM H = 1 fits only in INT8 (386 KB). |

### Stage A in numbers (final avg-100, median of 6)

| Arm | C1 | C4 | Δ vs truncated (C1 / C4) | Δ vs MLP (C1 / C4) |
|---|---:|---:|---|---|
| MLP 135k | −178 | −235 | — | — |
| TRM truncated | −209 | −340 | — | −30 / −105 |
| TRM full gradient | −218 | −345 | +2.5 (3/6) / +10.1 (4/6) | −40.5 (1/6) / −82.7 (1/6) |
| TRM full grad + deep sup. | −200 | −323 | +0.3 (3/6) / +28.4 (4/6) | −22.5 (1/6) / −88.8 (1/6) |

### Stage T in numbers (median of 12)

| Trunk | TD3 | DDPG (Phase 3) | Seeds that learned |
|---|---:|---:|---:|
| MLP 135k | −172 | −174 | 12/12 |
| MLP 788k | −176 | −177 | 12/12 |
| TRM 136k | −321 | −232 | 9/12 |
| TRM 788k | −1389 | −208 | 2/12 |

## Discussion

- **TRM was not untuned (GC).** The default learning rates are near the best of a 9-cell grid, and TRM's critic loss stays well above the MLP's in every cell. Phase 3's deficit is not a tuning artefact within ×0.3–×3.
- **The training discipline does not rescue it (GA).**
  - Giving gradient to all 21 cell applications, and supervising every cycle's critic readout, leaves return and value fit essentially where truncated TRM was.
  - That separates "untrained depth" from the deficit. Depth was inert in Phase 3 because the cell gets no new information between passes, not merely because it got no gradient.
  - Compare Tiny Recursive Control (Jain & Linares, 2025): there, every pass receives the error from simulating its current answer, so iteration has something to correct.
- **The direction of G1 survives a second algorithm (GT), but the magnitude does not replicate.**
  - Under TD3's standard settings the MLPs train as well as under DDPG.
  - TRM 788k mostly fails to train at all; its critic loss rises through training.
  - In practice this is evidence against TRM: it is harder to optimise and needs care the MLP does not. It is not a replication of "loses by about 30 units".
  - The likely contributor, `td3.yaml`'s 10× higher actor learning rate, is untested.
- **Complexity stays untested.** InvertedDoublePendulum is learnable without delay, but the one-step delay keeps the MLP unreliable even at 1M steps, and Hopper never reached its bar. No harder task gave a usable baseline, and with TRM untrainable under TD3 a B1 run would have measured that failure instead. This is a limitation of the evidence, not evidence against TRM.
- **Robustness is not explained by local sensitivity (E).** TRM's policies react *less* sharply to small input changes yet degrade more under finite sensor noise. Whatever makes TRM less robust is not captured by its local Jacobian.

## What Phase 4 changes about the Phase 0–3 conclusions

| Phase 3 conclusion | After Phase 4 |
|---|---|
| The deficit is architectural (DDPG) | Still holds. Not tuning (GC), not the training discipline (GA). Its direction also holds under TD3 (GT), via an optimisation failure. |
| Restoring the training discipline is the one remaining route to a positive result | Tested, no effect (GA). |
| Generalisation to TD3 and an unstable plant is open | TD3: tested (GT, with caveat). Unstable plant: the baseline can't learn it reliably with delay. |
| Complexity scaling is untested | Still untested. Two more rungs (IDP, Hopper) failed the baseline. |

**Bottom line:** within the settings tested, TRM's weight-tied recursion brings no consistent benefit as a model-free actor-critic trunk. Five ways of rescuing it have now failed: capacity, depth, memory, tuning and training discipline. Under a second algorithm it is harder still to train. **Use the MLP.**
