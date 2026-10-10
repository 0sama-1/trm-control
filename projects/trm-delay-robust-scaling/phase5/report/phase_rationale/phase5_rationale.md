# Phase 5 Rationale: Settling the TD3 Caveat

*trm-delay-robust-scaling · phase rationale · 2026-10-10 · 36 Katana jobs (+ 12 reused Stage T runs)*
*Why the phase existed, what inspired its design, and what it contributed. Results are in `../phase_summaries/phase5_summary.md`; labels are in `../glossary.md`.*

**Research question (the whole project).** Does weight-tied recursion (TRM), used as the actor-critic trunk of a model-free RL agent, benefit continuous control under action delay, and if not, why not?

## In one paragraph

Phase 4's Stage T asked whether Phase 3's central result holds under a second algorithm. The gate passed, but for the wrong reason: under TD3's standard settings TRM 788k mostly failed to train at all, so the pass measured an optimisation failure rather than a replication. That left a loose end in the project's strongest claim. Phase 5 existed to tie it off with one small, pre-registered test of the single most likely cause, TD3's actor learning rate. The cause was confirmed, and at a matched rate Phase 3's attribution replicated under TD3.

## 1. Where the project stood going in

- Under DDPG (Phases 1–4): a consistent, architectural deficit, with five rescues tested and failed.
- Under TD3 with `td3.yaml` unchanged (Stage T): both MLPs learned in every run; TRM 788k stayed at random-policy level in 10/12 runs and its critic loss rose through training. GT passed 12/12 with a recorded caveat.
- Two clues to the cause:
  - `td3.yaml` sets the actor learning rate to 1e-3, ten times the DDPG rate (1e-4) under which TRM trained normally in every earlier phase;
  - Phase 4's Stage C had shown, under DDPG, that raising TRM's actor rate hurts it.

## 2. Why this phase was made

1. **To protect the headline claim.** "The deficit is architectural" is the project's central finding. If it held under DDPG only, or held under TD3 only by breaking TRM, an examiner could call it algorithm-specific.
2. **To turn a caveat into a finding.** "TRM fails under TD3's defaults" has two readings: TD3 is bad for recursion, or one hyperparameter is. Only the second is cheap to test, and it predicts something specific.
3. **To keep scope tight.** The author's framing is to assess TRM as it is, without redesigns. Phase 5 changes one existing hyperparameter and nothing else, so it stays inside that framing.

## 3. Design decisions and what inspired them

| Decision | Inspiration | Why it mattered |
|---|---|---|
| Vary only the actor LR; leave the critic LR and the rest of `td3.yaml` unchanged | Stage C's finding that the actor rate is what hurts TRM; the 10× gap between `td3.yaml` and DDPG | One factor, so the outcome is attributable. |
| Actor LR ∈ {1e-4, 3e-4, 1e-3} | 1e-4 is the DDPG rate TRM trained under; 3e-4 is the midpoint on a log scale; 1e-3 is TD3's default | Brackets the failure and shows whether the effect is graded. |
| Reuse Stage T's 1e-3 runs (amendment P5-1, before data) | Compute economy, with identical code and configuration | Saved 12 jobs. Code byte-identical to Phase 4's Stage T snapshot was checked before submission. |
| An MLP 788k control at actor LR 1e-4 | The need for an LR-matched comparison | Without it, any TRM − MLP contrast would mix architecture with learning rate. The 1e-4 contrast was made primary. |
| Same conditions (C0, C1) and seeds (1–6) as Stage T | G1's and GT's pooled design | Keeps n = 12 pairs, so GT2's contrast is directly comparable with G1 and GT. |
| "Learned" = final avg-100 > −1000 | The random-policy level on Pendulum (≈ −1164) | A coarse but pre-registered cut that separates learning from not learning, fixed before data. |
| Gate GT2: ≥ 10/12 learned at some LR below 1e-3 | The ≥ 10/12 rule used for G1 and GT | Same standard of consistency as the rest of the project. |
| Same procedure as Phase 4: pre-registration, frozen snapshot, ledger, stop at the gate | Phases 3–4 ground rules | The answer carries the same evidential weight as earlier gates. |

## 4. How it contributed to the research question

| Before Phase 5 | After Phase 5 |
|---|---|
| G1's direction holds under TD3, but via an optimisation failure | **Replicated at a matched actor LR:** TRM 788k − MLP 788k −13.0, 2/12 pairs, p = 0.039 (Phase 3 under DDPG: −27.1, 2/12) |
| Cause of TRM's TD3 failure untested | **"LR explains" (GT2):** TRM 788k learns 12/12 at 1e-4, 11/12 at 3e-4, 2/12 at 1e-3 |
| Value fit worse under DDPG | Also worse under TD3: late critic loss 36.0 vs 25.4 (about 1.4× by medians) |
| — | **New:** TRM is far more sensitive to the actor LR than the MLP. The MLP is unaffected by a 10× change (−176.1 vs −175.9); TRM goes from 12/12 learned to 2/12. |

For the "does it benefit" half of the question, Phase 5 extends the answer to a second algorithm: no, under DDPG and under TD3 at sensible learning rates, by a small and consistent margin. For the "why not" half it adds a practical cost: the recursive trunk is fragile to a hyperparameter that a plain MLP ignores, so a standard configuration can break it.

## 5. How it shaped what follows

Phase 5 closes the project's planned experiments. It moves two items on the open list:

- **Resolved:** the GT caveat. The attribution now holds under two algorithms.
- **Opened:** how far TRM's learning-rate fragility extends (critic LR, batch size, other algorithms), and whether the smaller TRM is less LR-sensitive.

Still open from earlier phases: complexity scaling on a task the baseline learns, a tuned recurrent baseline, and why TRM is less robust despite a lower local gain.

## 6. In hindsight

- **One pilot run would have caught this.** A single TRM run under `td3.yaml` before Stage T would have shown the failure, and the fix could have been part of Stage T's design.
- **A narrow phase was the right size.** One factor, one gate, 36 jobs: the question was specific, so the test could be too.
- **The magnitude differs from Phase 3.** The median deficit is about half Phase 3's (−13.0 vs −27.1), with the same sign count. The claim that replicates is the direction and consistency, not the exact size.

## Sources

- Fujimoto et al. (2018), TD3.
- Project files:
  - `phase5/plan/plan.md`
  - `experiments/phase5/gates.json` (GT2, P5-1)
  - `experiments/analysis_p5_t2.json`
  - `../phase5_report.md`
  - `../stage_T2.md`
