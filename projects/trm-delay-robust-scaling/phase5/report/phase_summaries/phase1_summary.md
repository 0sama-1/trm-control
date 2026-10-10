# Phase 1 Summary: Horizon and Sparsity

*trm-delay-robust-scaling · phase summary · 2026-09-08/09 · 48 runs*
*Labels are defined in `../glossary.md`.*

## Question

Does TRM's disadvantage shrink, or turn into an advantage, as the task gets harder?

**H1.** TRM's advantage over the MLP and Transformer is non-decreasing in task complexity. It becomes measurable once the horizon is about 5× Pendulum's and the reward is sparse, which MountainCarContinuous-v0 provides (999 steps, one sparse terminal reward).

## Design

- **Trunks:** MLP, Transformer, TRM, LSTM.
- **Environments:** Pendulum-v1 and MountainCarContinuous-v0.
- **Conditions:** no delay; fixed d = 1 + queue.
- **Seeds:** 3, so 48 runs, 200 episodes each, DDPG, Katana CPUs.
- **Statistic:** a pooled sign test, n = 6 per comparison (3 seeds × 2 conditions), so a clean sweep gives p = 0.016.
- The matrix was submitted three times: v1 lost 8 jobs to walltime, v2 added per-episode logging, and v3 added step histories and is canonical.

## Results

| Comparison | Environment | TRM wins | p |
|---|---|---:|---:|
| TRM − MLP | Pendulum-v1 | **0/6** | **0.016** (a significant deficit) |
| TRM − MLP | MountainCarContinuous-v0 | 3/6 | 0.66 |
| TRM − Transformer | Pendulum / MountainCar | 3/6, 4/6 | ≥ 0.34 |

- **Parity failed.** The MLP beat TRM in every Pendulum seed pair, by 37.1 reward units under fixed delay (−194.0 vs −231.1), about 3% of the untrained-to-best span.
- **H1's threshold was falsified.** The gap closed on MountainCar in the predicted direction but never approached significance.
- **Cost grew with complexity**, from 6.7× to 12.8× the MLP's wall-clock.
- **The LSTM never learned Pendulum** (≈ −1460) while learning other tasks. It was flagged as a tuning artefact.

## What it established

The pilot's anecdote became a controlled measurement, a small but perfectly consistent deficit on the one task every trunk learned. The obvious rescue ("harder tasks will need the extra computation") failed its first proper test.

## What was not yet known

Whether MountainCar was actually learned. The Phase 2 re-analysis showed it was not: it was a floor effect, so the closing gap was an artefact.
