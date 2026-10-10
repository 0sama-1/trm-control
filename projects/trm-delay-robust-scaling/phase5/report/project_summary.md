# Recursion Under Delay: Project Summary, Phases 0–5

*trm-delay-robust-scaling · summary document · 2026-10-10*
*A short account of the whole project for a supervisor: the question, what each phase did and found, the answer, and what remains open. The detailed record is `full_report.md`; Phase 5 is reported in `phase5_report.md` and `stage_T2.md`, Phase 4 in `phase4/report/phase4_report.md` and its stage notes; per-phase summaries, a Q&A and a glossary sit alongside them in `phase5/report/`. Decision rules and gate outcomes are in `experiments/phase4/gates.json` and `experiments/phase5/gates.json`.*

## The short version

- **The question.** Does weight-tied recursion (TRM), used as the actor-critic trunk of a model-free RL agent, benefit continuous control under action delay — and if not, why not?
- **The answer.** No, on every task that was learned. On Pendulum-v1 a TRM trunk trails a plain MLP, and the project removed, one by one, every explanation it could test for why: capacity, the latent reset, the absence of anything to infer, untuned learning rates, the truncated gradient, the missing per-cycle supervision, and the choice of learning algorithm. The recursion behaves like a harder-to-optimise feed-forward block whose extra iterations add nothing.
- **What stays open.** Whether TRM's standing improves on harder tasks (H1). Four candidate rungs above Pendulum were tried, and the MLP baseline learned none of them under the conditions used, so the question was never put at risk.
- **The recommendation.** Use the MLP. If a recursive trunk is required, use one outer cycle, and keep its actor learning rate low: TRM is far more sensitive to it than the MLP.

---

## Phase 0 — Groundwork

A pilot on Pendulum-v1 under a one-step delay had TRM and Transformer trunks failing where an MLP converged. The cause was the delay wrapper, not the trunks: the agent was never told which of its own actions was about to land. Appending the pending-action queue to the observation fixed it (verified exact for fixed delays), after which the pilot reported "comparable reward" at about 18× TRM's training time. The TRM port keeps TRM's shared cell, its two-latent recursion (21 cell applications per call) and its one-step gradient truncation, which trains only 7 of the 21. The project was scoped, reviewed against the literature, and given a pre-registered hypothesis, H1.

## Phase 1 — Does TRM catch up as the horizon grows? (48 runs)

H1 predicted that TRM's advantage grows with task complexity and becomes measurable on the continuous MountainCar task (MountainCarContinuous-v0: 5× Pendulum's horizon, sparse reward). Four trunks, two environments, two delay conditions, three seeds. On Pendulum the MLP beat TRM in 6/6 seed pairs (p = 0.016), by about 3% of the performance span: the pilot's parity did not survive. On MountainCar nothing separated, so the threshold prediction was falsified. TRM cost 6.7–12.8× the MLP's wall-clock.

## Phase 2 — A second complexity axis (54 runs)

BipedalWalker-v3, six trunks, with and without the queue, three seeds: no comparison reached significance. A post-hoc analysis of all 102 runs then showed that only Pendulum had been learned (MountainCar and BipedalWalker were floor effects), so H1 was untested, not disconfirmed. It also found TRM fits its value function worse, and named three confounds that could each explain the Pendulum deficit: TRM's 5.8× larger capacity, a latent reset at every call, and a task with nothing to infer.

## Phase 3 — Removing the confounds (198 runs, 300 checkpoint evaluations)

A pre-registered, gated design on Pendulum. **Capacity:** TRM loses to a parameter-matched MLP at both 135k (0/12) and 788k (2/12) parameters (actor + critic, excluding target networks), while a 5.8× larger MLP changes nothing; the deficit is architectural. **Depth:** one, three or six outer cycles give the same return. **Value fit:** TRM's is worse even at matched capacity. **Memory:** carrying the latent across steps changes nothing, and where memory had something to recover (hiding the queue under a stochastic delay), no trunk recovered it; the residual-uncertainty hypothesis H3 was rejected, as was a capacity account. **Control quality:** TRM's greedy policy nearly matches the MLP's return and commands much smoother torque, but it is less robust to noise, plant changes and dropout, and costs 65× the MLP's decision latency.

## Phase 4 — Training discipline, tuning, a second algorithm, a harder rung (141 runs)

- **A harder rung (Stage B0).** Neither InvertedDoublePendulum-v5 nor Hopper-v5 was learned by the MLP under TD3 with a one-step delay (0/3 at every budget up to 500k; IDP also 0/3 at 1M). Without the delay the same MLP does reach the IDP bar: the delay on an unstable plant is the obstacle. The complexity test (B1, H1′) was therefore not run.
- **Tuning (Stage C).** A 3 × 3 grid of actor and critic learning rates (×0.3–×3) found nothing better than the defaults; raising the actor's rate hurts.
- **Training discipline (Stage A).** Training all 21 cell applications, with or without a value loss at every outer cycle, does not improve TRM (best 4/6 seeds; the gate needed 5/6) and costs 1.45–1.5× the training time.
- **A second algorithm (Stage T).** Under TD3 as configured, both MLPs reproduce their DDPG results in every run, but the 788k TRM (about 1.18M parameters once TD3's twin critics are counted) fails to train in 10 of 12 runs. The pre-registered capacity contrast passes on sign (12/12), but by optimisation failure, not by replicating Phase 3's deficit. Phase 5 traced the failure to TD3's actor learning rate.
- **Mechanism (Stage E, exploratory).** TRM actors have a lower local gain than MLPs, yet lose more under sensor noise; gain does not explain the noise sensitivity.
- **Embedded export (Stage D).** At matched size TRM and MLP actors need the same flash (265.2 and 263.0 KiB in FP32); the 788k-configuration TRM fits 512 KiB only in INT8. The on-device test is the author's.

## Phase 5 — The actor learning rate under TD3 (36 runs)

Did TRM fail under TD3 because of the algorithm or because of one setting? `td3.yaml`'s actor learning rate is 1e-3, ten times DDPG's, and Stage C had shown that raising TRM's actor rate hurts it. A pre-registered stage (T2) reran Stage T's design with only the actor rate lowered (TRM 788k at 1e-4 and 3e-4; Stage T's 1e-3 runs reused; MLP 788k control at 1e-4). TRM learned (final avg-100 > −1000) in 12/12 runs at 1e-4, 11/12 at 3e-4 and 2/12 at 1e-3; the MLP learned 12/12 at both 1e-4 and 1e-3 (`experiments/analysis_p5_t2.json`).

**GT2: "LR explains"** (`experiments/phase5/gates.json`). At the matched rate TRM 788k still trails MLP 788k: median −13.0, 2/12 pairs positive (p = 0.039), the same direction and count as Phase 3 under DDPG (−27.1, 2/12), with a smaller median, and again a worse value fit (1.50×, 12/12). Phase 3's attribution now holds under two algorithms at matched, sensible learning rates, and the caveat on Stage T's gate is resolved. Exploratory: the MLP ignores a tenfold change of actor rate (paired −0.03); TRM goes from 12/12 learned to 2/12.

---

## The final answer

| Claim | Status |
|---|---|
| TRM trails the MLP on Pendulum-v1 | replicated (6/6, then 12/12), and under TD3 at a matched actor rate (−13.0, 2/12) |
| The deficit is architectural, not capacity | established |
| It is not an untuned learning rate | established within ×0.3–×3 |
| It is not the truncated gradient or missing per-cycle supervision | supported (GA fail) |
| Recursion depth adds cost, not return | established |
| Carrying the latent, or memory generally, helps | not supported (H3 rejected) |
| TRM gives smoother actuation | yes, but with less robustness and 65× latency |
| TRM is robust under a second algorithm | only at a low actor learning rate: the 788k TRM does not train under TD3's default (2/12 runs learn) but does at DDPG's rate (12/12) |
| The deficit is specific to DDPG | no: the same direction and sign count under TD3 at a matched rate |
| TRM's advantage grows with task complexity (H1) | **untested**: no harder rung was learned by the baseline |

**Why it fails, as far as the evidence goes.** Each pass of the recursion re-reads the same observation and receives no error signal of its own, so iterating adds depth but no information. In TRM's own domain each pass is anchored to a fixed answer; in model-based Tiny Recursive Control each pass sees the error of the current candidate under known dynamics. That contrast is context, not a proposal: the project asked whether TRM's architecture works as a model-free RL trunk.

**For practice.** Use the MLP for delay-conditioned model-free control at this scale. If a recursive trunk is required anyway, use `H_cycles = 1`: the same return at a third of the latency. Restoring full gradients is not worth its 1.45–1.5× training cost.

## Open questions

1. **Complexity scaling (H1).** Still untested. It needs a rung the baseline learns under delay; on IDP the delay, not the budget, is the obstacle.
2. **Why TRM is so sensitive to the actor learning rate.** Phase 5 showed the TD3 failure was the actor rate; the mechanism, and TRM's deficit at its best TD3 rate (nothing below 1e-4 was tried), are open.
3. **What makes TRM noise-sensitive,** now that local gain is excluded.
4. **A tuned recurrent baseline,** before any claim about memory in this setting.
5. **The on-device test** of the exported actors, and their accuracy after INT8 quantisation.
