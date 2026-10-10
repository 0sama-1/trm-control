# Phase 5 Summary: Does the Actor Learning Rate Explain TRM's TD3 Failure?

*trm-delay-robust-scaling · phase summary · 2026-10-10 · 36 Katana jobs (+ 12 reused Stage T runs)*
*Labels are defined in `../glossary.md`. Pre-registration: `../../plan/plan.md`; rules and amendment: `../../../experiments/phase5/gates.json`; analysis: `../../../experiments/analysis_p5_t2.json`.*

## Question

Phase 4's Stage T reran Phase 3's capacity contrast under TD3 with its standard settings (`td3.yaml`). GT passed, TRM 788k − MLP 788k negative in 12/12, but by an optimisation failure:
- TRM 788k stayed at random-policy level in 10/12 runs;
- its critic loss rose through training;
- both MLPs learned in every run.

The suspected cause was the actor learning rate. `td3.yaml` uses 1e-3, 10× the DDPG actor rate (1e-4) under which TRM trained normally in Phases 1–4, and Phase 4's LR sweep showed that a higher actor LR hurts TRM. Phase 5 tests that one explanation.

## Methodology

- **Pre-registration.** The plan and gate GT2 were written into `plan.md` and `gates.json` before any data, under the same procedure as Phase 4 (frozen code snapshot `p5code/t2`, its own ledger, stop at the gate). Amendment P5-1, made before data with the author's approval, reuses Stage T's 12 TRM runs as the actor-LR 1e-3 arm.
- **Code.** No changes. `src/` and `conf/` are byte-identical to Phase 4's frozen Stage T snapshot.
- **Setting.** Pendulum-v1, TD3 (`td3.yaml`), with only `arch.actor_lr` overridden; the critic LR stays 1e-3. 200 episodes, conditions C0 (no delay) and C1 (fixed d = 1 + queue, no previous action), seeds 1–6.

| Arm | Actor LR | Runs | Source |
|---|---|---:|---|
| TRM 788k | 1e-4 | 12 | new |
| TRM 788k | 3e-4 | 12 | new |
| TRM 788k | 1e-3 | 12 | Stage T (P5-1) |
| MLP 788k (control) | 1e-4 | 12 | new |

- **Rule (GT2).**
  - "Learned" = final avg-100 > −1000.
  - "LR explains the TD3 failure" if TRM 788k learns in ≥ 10/12 runs at some actor LR below 1e-3.
  - If so, TRM 788k − MLP 788k at that LR is reported (n = 12 pairs, sign test). The LR-matched 1e-4 contrast is primary.

## Results

| Arm | Learned | Median final avg-100 | Late critic loss | Eval last-5 |
|---|---:|---:|---:|---:|
| TRM 788k, actor LR 1e-4 | **12/12** | −187.7 | 36.0 | −173.1 |
| TRM 788k, actor LR 3e-4 | 11/12 | −234.6 | 47.4 | −200.7 |
| TRM 788k, actor LR 1e-3 (Stage T) | 2/12 | −1389.3 | 560.9 | −1415.1 |
| MLP 788k, actor LR 1e-4 | 12/12 | −176.1 | 25.4 | −168.0 |
| MLP 788k, actor LR 1e-3 (Stage T) | 12/12 | −175.9 | 23.7 | −168.3 |

| Contrast (n = 12) | Median | Pairs positive | Two-sided p |
|---|---:|---:|---:|
| **TRM 788k − MLP 788k, both at actor LR 1e-4 (primary)** | **−13.0** | **2/12** | **0.039** |
| TRM 788k at 3e-4 − MLP 788k at 1e-4 (unequal LRs) | −53.6 | 0/12 | 0.0005 |

**Gate GT2: "LR explains".** TRM 788k learns in 12/12 runs at actor LR 1e-4, against 2/12 at TD3's default of 1e-3.

## Discussion

- **Stage T's failure was an actor-learning-rate effect, not something specific to TD3.**
  - At the DDPG-like actor LR, TRM trains under TD3 in every run.
  - At 3e-4 it trains in all but one run (C0 seed 1 stays near random).
  - At 1e-3 it mostly fails.
  - The critic-loss trend matches: 36 → 47 → 561 as the actor LR rises.
- **With the LR fixed, Phase 3's attribution replicates under a second algorithm.** TRM 788k − MLP 788k is −13.0 under TD3 (2/12 pairs, p = 0.039). Phase 3's G1 under DDPG gave −27.1 (2/12). The direction and sign count are the same; the median is about half the size, with almost the same seed spread. TRM's value fit is again worse (median late critic loss 36.0 vs 25.4, about 1.4×; the median per-pair ratio is 1.5×, in line with Phase 3's 1.45–1.89×).
- **TRM is far more sensitive to the actor learning rate than the MLP.** The MLP is unaffected by a 10× change (−176.1 at 1e-4, −175.9 at 1e-3). TRM goes from 12/12 learned to 2/12. This is a practical cost of the recursive trunk: a standard algorithm configuration that suits an MLP can break it.
- **This resolves Phase 4's GT caveat.** At matched, sensible learning rates, TRM shows the same modest, consistent deficit under DDPG and TD3, rather than an algorithm-specific failure.

## Threats

- The "learned" cut (−1000) is coarse. It separates learning from staying near random (≈ −1164), not good from mediocre runs.
- Three learning rates only. 1e-4 is not shown to be TRM's best TD3 setting.
- The MLP control runs at 1e-4 only, so the 3e-4 contrast mixes architecture with LR and is secondary.
- The 1e-3 arm is reused from Stage T. The configuration and code are identical, but runs are not bit-reproducible across GPU models.

## What Phase 5 changes

| Before Phase 5 | After Phase 5 |
|---|---|
| G1's direction holds under TD3, but via an optimisation failure (GT caveat) | G1 replicates under TD3 at a matched actor LR: −13.0, 2/12, p = 0.039 |
| Likely cause of the TD3 failure (actor LR) untested | Tested and confirmed (GT2) |
| — | New: TRM is markedly more sensitive to the actor learning rate than the MLP |

**Bottom line:** under both DDPG and TD3, with sensible learning rates, TRM 788k trails a parameter-matched MLP by a small, consistent margin and fits its value function worse. It is also much more fragile to the actor learning rate.
