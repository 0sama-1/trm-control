# Phase 1 Rationale: Testing the Obvious Rescue

*trm-delay-robust-scaling · phase rationale · 2026-09-08/09 · 48 runs*
*Why the phase existed, what inspired its design, and what it contributed. Results are in `../phase_summaries/phase1_summary.md`; labels are in `../glossary.md`.*

**Research question (the whole project).** Does weight-tied recursion (TRM), used as the actor-critic trunk of a model-free RL agent, benefit continuous control under action delay, and if not, why not?

## In one paragraph

Phase 0 ended with an anecdote, "comparable reward at about 18× the cost", and the rescue it invites: Pendulum is too easy, and harder tasks will reward the extra computation. Phase 1 existed to test both under control. It turned the anecdote into a measurement (TRM lost every Pendulum seed pair) and put the rescue, H1, to a falsifiable threshold test on a long-horizon, sparse-reward task. The threshold failed. Phase 2's re-analysis later showed the task had not been learned by any trunk, so H1 itself was never really at risk.

## 1. Where the project stood going in

- A verified delay test bed and a documented TRM port (Phase 0).
- An uncontrolled observation: with augmentation, MLP, Transformer and TRM reached "comparable reward" on Pendulum, with TRM about 18× slower to train.
- H1, pre-registered: TRM's advantage over the MLP and Transformer is non-decreasing in task complexity, and becomes measurable once the horizon is about 5× Pendulum's and the reward is sparse.

## 2. Why this phase was made

Two reasons, one defensive and one exploratory.

1. **The anecdote needed a control.** "Comparable" came from a pilot with no seeds paired and no statistic. If TRM was genuinely at parity on Pendulum, a harder task could tip it into an advantage. If it was already behind, the framing had to change.
2. **The rescue needed a test that could fail.** "Harder tasks will need the extra computation" is unfalsifiable unless "harder" and "measurable" are fixed in advance. H1 fixed both: horizon about 5× Pendulum's, sparse reward, significance at ≥ 5/6 pooled seed pairs.

## 3. Design decisions and what inspired them

| Decision | Inspiration | Why it mattered |
|---|---|---|
| MountainCarContinuous-v0 as the harder rung | H1's own threshold: 999-step horizon, one sparse terminal reward | It is the standard continuous-control task with both properties, so it tests the claim as written, not a convenient variant. |
| Four trunks: MLP, Transformer, TRM, LSTM | The trunks already in the codebase, plus an LSTM added in Phase 0 planning | The Transformer is a large non-recursive comparator; the LSTM is the classic recurrent baseline for partial observability. TRM's result could then be read against both "big" and "recurrent". |
| Two conditions: no delay (C0) and fixed d = 1 + queue (C1) | Phase 0's verified augmentation | Shows whether the delay changes the ranking, and doubles the pairs available for the sign test. |
| 3 seeds × 2 conditions pooled to n = 6 | Phase 0's power check | n = 6 is the smallest design in which a clean sweep is significant (p = 0.016). n = 3 alone cannot reach p < 0.125. |
| DDPG held fixed | The codebase's default algorithm | Changing only the trunk isolates the architecture. TD3 waited until Phase 4. |
| Wall-clock logged alongside reward | Cost as a first-class metric (project brief) | Turned "18× slower" into a measured multiple per environment (6.7× to 12.8×). |
| Re-submission when logging was found lacking (v1 → v3) | Operational lessons during the phase | v2 added per-episode logging and v3 step histories. That telemetry is what made Phase 2's re-analysis possible without new compute. |

## 4. How it contributed to the research question

| Before Phase 1 | After Phase 1 |
|---|---|
| "Comparable reward" on Pendulum (anecdote) | A consistent deficit: TRM − MLP 0/6 on Pendulum, p = 0.016; −37.1 under fixed delay, about 3% of the span |
| H1's threshold untested | Threshold falsified: TRM − MLP 3/6 on MountainCar, p = 0.66 |
| Cost "about 18×" | 6.7× to 12.8× the MLP's wall-clock, growing with task complexity |

For the "does it benefit" half of the question, Phase 1 gave the first controlled answer: no, on the one task every trunk learned. For the "why not" half it opened four competing explanations it could not separate: recursion itself, capacity (TRM had 5.8× the MLP's parameters), the latent reset, and a task with nothing for recursion to infer.

## 5. How it shaped the experiments that followed

- **Phase 2** added a second kind of difficulty (BipedalWalker's coupled dynamics), so a null could not be dismissed as "the wrong kind of hard", plus structure and augmentation ablations requested after Phase 1 review.
- **The Pendulum deficit** became the project's anchor contrast. Phase 3 reproduced it at n = 12 (−34.4, 0/12, p = 0.0005) and then took it apart.
- **The LSTM's failure on Pendulum** (≈ −1460) was flagged as a tuning artefact, which later fed Phase 3's decision to use previous-action input and stored-state replay for recurrent trunks (Ni et al., 2022; Kapturowski et al., 2019).

## 6. In hindsight

- **No budget pilot.** Nobody checked that MountainCar was learnable by the baseline at 200 episodes. It was not: the median run never reached the goal. The narrowing gap was all trunks converging on the same floor. Phase 4 made a budget pilot with a pre-registered "learned" bar mandatory before any complexity matrix.
- **The anecdote was overturned cleanly** because the statistic was fixed before the data. That is the main methodological win of the phase.

## Sources

- Matheron et al. (2019), the DDPG + OU-noise exploration failure (identified post hoc in Phase 2).
- Ni et al. (2022), recurrent model-free RL as a POMDP baseline.
- Project files:
  - `phase0-literature/ideation/proposal-final.json` (H1)
  - `../full_report.md` §5
  - `experiments/analysis_v3_final.json`
