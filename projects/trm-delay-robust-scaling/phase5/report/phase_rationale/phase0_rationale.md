# Phase 0 Rationale: Why Build the Test Bed First

*trm-delay-robust-scaling · phase rationale · up to 2026-09-08*
*Why the phase existed, what inspired its design, and what it contributed. Results are in `../phase_summaries/phase0_summary.md`; labels are in `../glossary.md`.*

**Research question (the whole project).** Does weight-tied recursion (TRM), used as the actor-critic trunk of a model-free RL agent, benefit continuous control under action delay, and if not, why not?

## In one paragraph

Phase 0 existed because a comparison between architectures is only worth something if the test bed is correct, the recursive trunk is a faithful port, and the success criterion is fixed before the data. An early pilot failed the first of those three tests: the delay wrapper was silently corrupting every training transition. Phase 0 fixed the wrapper, ported TRM, and turned an informal idea into a hypothesis with a pre-registered statistic. It produced no verdict on TRM. Every later verdict depends on it.

## 1. Where the project stood going in

- **The idea.** Tiny Recursive Models (Jolicoeur-Martineau, 2025) reach 45% on ARC-AGI-1 with about 7M parameters by applying one small, weight-tied cell many times rather than adding parameters. The claim underneath is about a *mechanism*: iteration can substitute for parameter count. If that holds anywhere outside puzzles, embedded control is the obvious place, because parameters, memory and latency are scarce there.
- **The codebase.** DDPG, PPO, TD3 and SAC were implemented, with MLP and Transformer trunks and a `DelayedEnv` wrapper.
- **The pilot.** On Pendulum-v1 under a one-step delay, the MLP converged and the TRM and Transformer trunks did not.

## 2. Why this phase was made

The pilot looked like a result ("TRM fails under delay"), but it was traced to a bug. `DelayedEnv` applied each action late without telling the agent which action was about to land, so the replay buffer paired every reward with an action that had not caused it. The MLP happened to tolerate the corrupted signal; the larger trunks did not.

That made three things urgent before any experiment could be trusted:

1. **A correct test bed.** The delay had to be real and the agent had to be given the standard sufficient statistic.
2. **A faithful trunk.** "TRM" had to mean TRM's mechanism, not a loose imitation, or a negative result would only be about the imitation.
3. **A falsifiable claim.** After the fix the pilot reported "comparable reward at about 18× the training cost". That was one anecdote. It needed to become a hypothesis with a test fixed in advance.

## 3. Design decisions and what inspired them

| Decision | Inspiration | Why it mattered |
|---|---|---|
| Action delay as the test bed | Delayed-MDP theory (Katsikopoulos & Engelbrecht, 2003): under a d-step delay the observation is no longer Markov; appending the pending-action queue restores a sufficient statistic in principle | The theorem says a sufficient statistic *exists*, not that a small network can fit the resulting value function. Iterative refinement, acting like a learned state estimator, was a plausible bias for exactly that. A harder task alone would vary difficulty, not the property recursion is supposed to exploit. |
| Queue augmentation as the baseline, verified exactly | The same theory, plus the pilot bug | Verified exact for fixed delays (5 delay values × 10 steps, zero mismatches). An audit also fixed a silent stochastic-delay bug. Without it the project would have measured a logging error. |
| Port TRM's mechanism, not its task machinery | TRM's own design | **Kept:** the shared SwiGLU/RMSNorm cell, the two latents z_H and z_L, 3 outer × 6 inner cycles, the one-step gradient truncation. **Dropped:** token attention (a flat observation is a sequence of length 1), per-puzzle embeddings (replay transitions are not identified instances), ACT halting and deep supervision (off-policy RL never revisits the same input across optimiser steps). |
| A drop-in trunk contract | TRM's cell is itself a small MLP applied repeatedly, i.e. a recurrent network | Every trunk plugs into the same actor and critic with nothing else changed, which makes a like-for-like comparison possible. |
| Model-free positioning | The two mandatory prior works, Tiny Recursive Control (Jain & Linares, 2025) and the J2-Lambert TRM (Wijayatunga & Armellin, 2026), are both *model-based* refiners against known dynamics | Defines the gap the project fills: recursion as implicit memory or state estimation in model-free RL, where no dynamics model or solver labels exist. |
| Run as a gated research epic | Pre-registration practice; three rounds of hypothesis review | Turned the anecdote into H1 with a threshold prediction. The reviewers capped significance at 3/5 because the claim rested on one sparse-reward environment, and that limitation was carried into the reports rather than hidden. |
| A pooled sign test instead of a regression | A power check: the regression's best possible p at n = 3 was 0.125 | Pooling 3 seeds × 2 conditions gives n = 6, so a clean sweep reaches p = 0.016. Sign tests make no distributional assumption, which suits heavy-tailed RL returns. |
| Cost as a first-class metric; a negative result as a valid outcome | The project brief (`project-config.yaml`) | Kept the project from chasing a positive result, and put the 18× cost on the same footing as reward. |

## 4. How it contributed to the research question

Phase 0 answered no part of the question directly. It made the question answerable:

- **"Under action delay"** became a correct, verified condition rather than a buggy one.
- **"TRM"** became a specific, documented port with two known simplifications.
- **"Benefit"** became measurable on three fixed axes: final return (pre-registered primary), cost, and later control quality.
- **The first hypothesis, H1**, was fixed: TRM's advantage grows with task complexity and becomes measurable on a long-horizon, sparse-reward task.

## 5. How it shaped the experiments that followed

| Phase 0 element | Where it went |
|---|---|
| "Comparable reward at 18× cost" | Phase 1 tested it under control and found a consistent deficit instead. |
| The latent re-initialised every call | Identified in Phase 2's re-analysis as confound C-2; tested in Phase 3 as Reset-Z versus Carry-Z (G3). |
| Only 7 of 21 cell applications get gradient | Phase 3 found depth inert; Phase 4 Stage A restored full gradient and per-cycle supervision (GA). |
| H1 and its threshold | Phase 1 (MountainCar), Phase 2 (BipedalWalker), Phase 4 (H1′, InvertedDoublePendulum and Hopper). |
| Sign tests, paired by seed | Used by every gate in Phases 1–5. |
| The model-free versus model-based distinction | Became the explanation of the final result: Tiny Recursive Control feeds new information into every pass; this port does not. |

## 6. In hindsight

- **The bug was the most consequential finding of the whole project's set-up.** Without it, the pilot's "TRM fails under delay" would have been reported as an architectural result.
- **The "comparable reward" anecdote was one seed.** It framed the project for a phase until Phase 1 overturned it, which is the argument for testing anecdotes before building on them.
- **Both port simplifications were faithful to TRM as published**, yet both became confounds to test. Documenting them at port time is what made that possible.

## Sources

- Jolicoeur-Martineau (2025), Tiny Recursive Models.
- Katsikopoulos & Engelbrecht (2003), Markov decision processes with delays.
- Jain & Linares (2025), Tiny Recursive Control; Wijayatunga & Armellin (2026), TRM for the J2-perturbed Lambert problem.
- Project files:
  - `phase0-literature/literature-review/review.md`
  - `phase0-literature/ideation/proposal-final.json`
  - `../full_report.md` §4
