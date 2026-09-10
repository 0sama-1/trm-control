# Phase 3 Proposal: Follow-Up Design

*Drafted after Phase 2 (BipedalWalker-v3) completion. Not yet submitted as PBS jobs — this is a proposal for review, consistent with `execution.constraints` (all training runs are batch jobs, minimum 3 seeds/condition) and the project's autonomy mode; new compute spend awaits author sign-off.*

## 1. Where the evidence actually stands after Phase 1 + Phase 2

- **Two independent complexity axes tested (horizon/sparsity via Pendulum→MountainCarContinuous; dimensionality/coupling via BipedalWalker), three environments, six architecture families.** Across every pairwise comparison run in both phases, there is exactly **one** statistically significant result — TRM significantly *worse* than DNN on Pendulum-v1 (p=0.016) — and **zero** significant TRM advantages anywhere.
- **TRM's wall-clock overhead over DNN has never been below ~5x** and reached 14x on MountainCarContinuous-v0. Cost has not been recovered by any observed benefit at any rung.
- **The TRM-variant ablation (hier6, singlez) found no structural difference from base TRM anywhere** — not a near-miss, but genuinely sign-inconsistent, noise-dominated deltas (Phase 2 report §3, Table 2).
- **The one candidate real effect in the entire two-phase dataset is the augmentation ablation split**: TRM-hier6 gained from observation-augmentation in all 3 seeds with tightly-clustered magnitude (+11.1/+9.6/+11.3), and LSTM lost from it in all 3 seeds (−3.1/−3.6/−8.6) — both hit the n=3 significance ceiling (p=0.125) without being genuinely testable at that sample size.

**Reading this honestly (per the project's own research boundary that a rigorous negative result is a valid, reportable outcome, not something to route around):** the general complexity-scaling hypothesis (H1: TRM's advantage grows with task complexity) has now failed to produce a positive result on two independently-varied axes. Continuing to add complexity rungs on the same n=3, single-shot design has a low prior of suddenly producing a significant advantage, and would repeat the exact statistical-power mistake Phase 1's own methodology section already diagnosed and fixed once (switching from n=3 to a pooled n=6 design) — Phase 2 regressed back to n=3 and inherited the same ceiling.

## 2. Proposed Phase 3 priorities, in order

### Priority 1 (recommended first): adequately-powered augmentation-ablation follow-up

**Rationale.** This is the only pattern in the two-phase dataset with internally consistent direction *and* magnitude across all seeds observed so far — exactly the signature a real effect would leave and noise usually doesn't. It is also comparatively cheap: it needs only 2 architectures (TRM-hier6, LSTM) × 2 conditions (fixed+aug, fixed+noaug) rather than the full 6×3 matrix.

**Design.** Re-run TRM-hier6 and LSTM on BipedalWalker-v3, fixed-delay, aug vs. no-aug, at **6 seeds instead of 3** (bringing each cell to the same n=6 pooling power Phase 1 used, reaching p≤0.0313 on a clean 6/6 or 5/6 sweep) = **2 architectures × 2 conditions × 6 seeds = 24 jobs**, all GPU (`mechgpu12`, consistent with Phase 2 infra — `conf/env/bipedalwalker.yaml`, `experiments/phase2_submit.sh` as the template). No new environment or architecture code required.

**Decision rule.** If the augmentation effect replicates at n=6 (5/6 or 6/6 agreement, both architectures), it becomes the first genuine, adequately-powered positive finding in this project and is worth its own short report section explaining *why* augmentation might interact differently with a deeper recursive trunk (hier6) versus a stateless recurrent one (LSTM) — plausibly, hier6's extra recursive depth has more "room" to exploit the augmented action-queue signal, while LSTM's single-call sequence processing may be more sensitive to the extra tokens as noise. If it does not replicate, that closes the last open thread from Phase 2 cleanly and cheaply (24 jobs, not another 54-job matrix), before committing to Priority 2.

### Priority 2 (contingent): the originally-proposed third complexity axis (MuJoCo)

**Rationale.** The original three-axis ladder (dimensionality, horizon, dynamics coupling) has now had two axes tested; a MuJoCo environment (e.g. `HalfCheetah-v4` or `Ant-v4` — higher-dimensional than BipedalWalker, harder dynamics coupling) would complete the originally-scoped ladder and is the natural G3 continuation per `EPIC.md`. This was flagged in both prior reports as license/dependency-gated and not yet resolved.

**Design, if pursued.** Reuse the Phase 2 6-architecture × 3-condition structure, but at **5+ seeds per cell from the start** (not 3), given the lesson from both prior phases — this is 90 jobs at 5 seeds, a meaningful compute commitment that should be sized against the actual prior probability of a positive result, which the evidence above suggests is low. **Recommendation: gate this on (a) resolving the MuJoCo license/dependency check, and (b) the outcome of Priority 1** — if Priority 1 finds nothing, the marginal value of a third complexity rung under the same negative pattern is low, and the stronger recommendation becomes closing the empirical arc and moving to final project synthesis (Phase E) instead.

### Priority 3 (not recommended to run, recorded for completeness): a fourth TRM variant or algorithm swap

Raised only to explicitly decline it with reasoning: neither the TRM-variant ablation (Table 2) nor the primary comparison gives any signal that a *different* recursive structure or a different off-policy algorithm (TD3/SAC in place of DDPG) would change the picture — there is no observed trend to extrapolate from, only flat noise. Pursuing this without a new hypothesis for *why* it would differ would be adding compute without a falsifiable question attached, which the project's research boundaries caution against.

## 3. What this proposal is not

This is not a request to keep running experiments until a positive result appears — Priority 1 is scoped specifically to either confirm or kill the one candidate signal in the data at proper statistical power, cheaply, and Priority 2 is explicitly conditioned on that outcome rather than assumed. If Priority 1 fails to replicate, the recommended next step is writing the final combined-phases synthesis (Phase E) reporting the complexity-scaling hypothesis as disconfirmed on the two axes tested, not opening a Phase 4.

## 4. Next action

Awaiting author decision: (a) approve Priority 1 (24 jobs, ~cheap, high information value) to run next, (b) approve both Priority 1 and Priority 2 in sequence, or (c) treat Phase 2 as the last empirical phase and move straight to final synthesis.
