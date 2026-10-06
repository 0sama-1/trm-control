# Phase 3 Plan: Does Recursion Have Anything Left To Infer?

*Experiment plan for `trm-delay-robust-scaling`, drafted 2026-09-14 against the Phase 1 + Phase 2 evidence base (102 completed runs; all 102 checkpoints verified present on Katana). Nothing here has been submitted; the compute in §4 awaits sign-off.*

## 0. The one-paragraph version

Phases 1 and 2 measured a recursive trunk under the one delay condition where the theory says recursion is unnecessary. Fixed, known, 1-step delay plus action-queue augmentation restores an exact sufficient statistic — our own `augmentation.md` §5–6 proves and verifies it for this wrapper — so there was no residual uncertainty for iterative refinement to resolve, and our trunk could not have resolved it anyway because it re-initialises its latent on every forward call. Under those conditions a null result is the *predicted* outcome, not evidence against recursive computation. Phase 3 therefore stops adding complexity rungs and instead (a) removes the three confounds that make the current negative uninterpretable, (b) tests recursion where the conditioning genuinely leaves something to infer, and (c) opens a second evaluation axis the project has never measured — the control-engineering quality of the resulting controller, which is what decides whether a trunk is usable in a robot at all. Stage 0 costs 18 training jobs plus one zero-training evaluation pass over checkpoints we already own, and can kill the whole phase for about four CPU-hours.

---

## 1. What the evidence actually supports

### 1.1 The ledger

| Established | Strength | Source |
|---|---|---|
| On the one environment where every trunk learned, TRM loses to a 5.8× smaller MLP in 6/6 seed-pairs by 37.1 reward units | p = 0.016; ≈3% of the untrained→best span | report §6.1 |
| Recursion costs 5–14× the MLP's training wall-clock on every environment and both hardware configurations | invariant in direction | report §6.4 |
| The recursive trunks fit the value function worse at matched budget (7/7 cells early, median 2.72×) | p = 0.008 | report §7.1 |
| That value-fit deficit is **not** recursion-specific — the 797k Transformer shows it too, the 167k LSTM does not | direct contradiction of the convenient reading | report §7.3 |
| Learning speed and seed stability are unaffected | episodes-to-90% 117 vs 115 | report §7.2 |
| Two of three environments never left the failure floor | best BipedalWalker return −81.1 vs +300 solved | report §6.2 |

### 1.2 The three confounds, restated as design faults rather than caveats

**C-1 — Capacity is perfectly confounded with architecture.** Every recursive trunk in the study holds ≈788k parameters and every MLP holds ≈135k. No cell of the design varies one while holding the other. Because the 797k Transformer reproduces the value-fit deficit and the 167k LSTM does not, capacity is currently the *better-supported* explanation of the headline result, and it runs in the direction opposite to what the original design assumed. Until this is crossed, the Phase 1 headline is not a measurement of recursion.

**C-2 — The port implements the weaker of two known latent-initialisation variants.** `TRMTrunk.forward` expands `H_init`/`L_init` fresh on every call, so nothing persists between consecutive environment steps. HRM-Agent (Dang & Rawlinson, 2025) compare exactly this choice and find carrying the latent converges faster in 4 of 5 runs. A trunk with no cross-step state cannot perform state estimation, which is the mechanism the project's own framing claims to be testing.

**C-3 — The delay condition leaves nothing to infer.** Under `FixedDelay(1)` with `augment_obs=true`, the pending-action queue is exact — verified in `augmentation.md` §6 against a recorded action sequence, all 5 delay values × 10 steps, zero mismatches. The augmented observation is a sufficient statistic for the delayed MDP (Katsikopoulos & Engelbrecht, 2003). An architecture whose hypothesised advantage is *inferring missing state* was given a task with no missing state.

The project's own documentation already names the regime where this stops being true. `augmentation.md` §5, "Known limitation: exact for fixed delay, approximate for stochastic/time-varying":

> For `StochasticDelay`/`TimeVaryingDelay`, whose per-step `d` can be *less* than `max_delay`, the actually-applied action isn't always the oldest one in the buffer, so the exposed "pending queue" is a reasonable proxy … rather than a proven sufficient statistic. Extending §3's guarantee to a resampled-every-step delay isn't a small change (**the agent would need to reason about a delay it can't observe in advance**) …

That last clause is an inference problem, written into the codebase before the experiment that failed to test it. `conf/delay/stochastic.yaml` and `conf/delay/time_varying.yaml` have existed since the beginning and **have never been run**: all 102 runs used `delay=none` or `delay=fixed value=1`.

**A fourth gap, for completeness.** `project-config.yaml` names the delay axis as "observation delay, actuation delay, and jitter". `DelayedEnv` implements **actuation delay only** — it queues the agent's action and applies an older one; the observation returned is always current. Observation delay and jitter have no implementation and no config group. They are listed in the exploration inventory (§6, items B7–B8) rather than folded into the critical path, but the project should stop describing them as tested axes.

### 1.3 What this means for the current report

The report's §10 programme (E1–E5) is sound but under-specified in two places, and Phase 3 supersedes it:

- **E1 (capacity-matched MLP) is only half the fix.** Growing the MLP to 788k leaves the comparison at one capacity level. Shrinking the recursive trunk to ≈135k as well turns it into a 2×2 factorial and is the only version that can attribute the deficit. §4.2 gives verified configurations for both.
- **E2 (Carry-Z) is not a "contained change".** Off-policy replay samples i.i.d. transitions with no latent attached. Carrying a latent at acting time but resetting it at update time is inconsistent and would confound the result. §7 specifies the R2D2-style stored-state fix and books real engineering time for it.

---

## 2. The Phase 3 hypothesis

**H3 — Residual-uncertainty hypothesis.** Weight-tied recursive computation earns its cost in model-free continuous control only where two conditions hold jointly: **(i)** the agent's conditioning leaves residual state uncertainty that iteration could resolve, and **(ii)** the trunk can carry information across decision steps so that the iteration has something to accumulate. Phases 1–2 satisfied neither. Their null is therefore the outcome H3 predicts, and Phase 3's job is to construct the conditions under which H3 makes a *different* prediction from the alternatives, then check.

The alternative H3 is tested against is not "recursion works" but the two deflationary accounts the current data actually support:

- **H_cap — capacity.** The deficits track parameter count under a fixed sample budget; recursion is irrelevant either way (Ota et al., 2021; Klein et al., 2024).
- **H_null — no transfer.** Recursive refinement is a property of supervised, revisitable, compositionally-correctable problems and does not transfer to model-free continuous control under any conditioning; the cost is always net negative.

### 2.1 Sub-hypotheses and discriminating predictions

| # | Claim | H3 predicts | H_cap predicts | H_null predicts | Stage |
|---|---|---|---|---|---|
| H3.1 | The measured deficit is a capacity artefact | small-TRM ≈ small-MLP, large-MLP ≈ large-TRM | same | large-MLP ≈ small-MLP, TRM worse at both sizes | 1 |
| H3.2 | **Uncertainty × memory interaction** | Carry-Z − MLP ≤ 0 under fixed-known delay, **> 0** under stochastic delay | no interaction (same nets both sides) | ≤ 0 everywhere | 2 |
| H3.3 | Depth pays only when there is something to infer | reward flat in `H_cycles` at fixed delay, increasing at stochastic delay | flat everywhere | flat-to-worse everywhere, cost linear in depth | 1, 2 |
| H3.4 | Recursion trades return for command regularity | recursive trunks smoother at matched return | regularity tracks architecture class, not recursion | no consistent difference | 0 |
| H3.5 | Recursion degrades more gracefully off-nominal | flatter degradation vs delay mismatch / obs noise | no difference | no difference or worse | 0 |
| H3.6 | Recursion is real-time feasible at robot rates | quantify: latency per decision vs control period | — | — | 0 |

H3.2 is the load-bearing row, and it is deliberately framed as an **interaction, not a main effect**. A capacity explanation cannot produce a sign flip across delay conditions using the identical networks, and neither can a tuning artefact that applies uniformly. A predicted change of sign across conditions is the strongest thing this design can ask for at the seed counts available.

### 2.2 Why this hypothesis is consistent with the data rather than a rescue of it

H3 does not require any Phase 1/2 number to be wrong. It reinterprets them:

- 0/6 seed-pairs at fixed-known delay is what H3 predicts (condition (i) violated, cost still paid).
- Value-fit deficit shared with the Transformer is what H3 predicts (nothing to infer ⇒ recursion contributes capacity and nothing else, and capacity is a liability at this budget).
- No difference between `trm`, `trm_hier6` and `trm_singlez` is what H3 predicts (varying the recursive *structure* changes nothing when the recursion has no signal to work on).
- 5–14× cost with no benefit is what H3 predicts for a mechanism running with its input information already complete.

H3 also makes a prediction the project can be held to: **if the interaction in H3.2 does not appear, H3 is dead** and the conclusion becomes H_null with all three confounds removed — a considerably stronger negative result than the one currently drafted, and one that closes the question rather than leaving it open.

---

## 3. The missing axis: what makes a trunk usable in a control system

The project scores trunks on return, training wall-clock, and parameter count. A controls or robotics reader needs three things none of those supply, and all three are cheap to obtain.

**3.1 Command quality.** A controller that achieves the same return while commanding twice the actuator rate is a worse controller: actuator wear, thermal load, gearbox backlash, and vibration all scale with command rate, not with reward. Standard measures — control effort ∫u²dt, command rate |Δu|, jerk |Δ²u|, saturation fraction, sign-reversal rate, and the fraction of command power above a cutoff frequency — are computable directly from rollouts. §5 shows a candidate signal already in the existing data.

**3.2 Robustness envelope, not point performance.** Every number in the project so far was measured under the exact conditions the policy trained in. A controller is judged on its margin: how far the plant can drift, how much the delay can vary, how noisy the sensor can get before performance collapses. Because delay-robustness is this project's founding axis, the absence of any off-nominal measurement is the largest single gap in the evidence base.

**3.3 Real-time feasibility.** Training wall-clock is the wrong cost for deployment. The deployment cost is per-decision inference latency against the control period. At `H_cycles=3, L_cycles=6` the TRM trunk performs **21 applications of the shared cell per forward pass** (18 inner updates plus 3 outer), each two SwiGLU blocks, versus the MLP's two matrix multiplies — and both the actor *and* critic pay it during training while only the actor pays at deployment. Whether that is 0.4 ms or 4 ms at batch 1 on a CPU decides whether the trunk is admissible at 100 Hz, 500 Hz or not at all. **The project has never measured it.** It is a one-hour experiment.

These three, plus return and parameters, form the **deployment scorecard** that Phase 3 reports for every trunk.

---

## 4. The stage plan

Six stages. Each names its entry condition, exactly what runs, what is measured, the pre-registered statistic, and the gate that decides whether the next stage happens. Only Stage 0 is committed at sign-off.

### 4.0 Stage map

```
Stage 0  headroom probe (18) + checkpoint battery (0 training)      ≈30 CPU-h
   │
   ├─ G0 fails ──▶  no headroom: no architecture can win here.
   │                Write up Stage 0 + Stage 1. Phase ends.
   ▼
Stage 1  capacity × depth factorial (72)               ≈60 CPU-h + 25 GPU-h
   │     (not gated by G0 — its question stands either way)
   │
   ├─ G1 ──▶  attribution: capacity or architecture? Both branches inform.
   ▼          Sets the capacity level Stage 2 runs at.
Stage 2  uncertainty × memory (72, +24 optional)       ≈70 CPU-h + 40 GPU-h
   │
   ├─ G2 fails ──▶  no interaction: report H_null with all three
   │                confounds removed. Stronger than the current draft.
   ▼
Stage 3  does it generalise? TD3 (24) + unstable plant (9 + 48)   ≈75 GPU-h
   │
   ├─ G3 fails ──▶  stop; P1 loses its constructive test.
   ▼
Stage 4  restore the training discipline (36)                      ≈45 GPU-h
   │
   ▼
Stage 5  synthesis — report revision, no compute
```

### 4.1 Stage 0 — Is there a question, and what has already been paid for?

**Entry:** none. This is the only stage committed at sign-off.

#### S0.1 — Headroom probe (18 jobs)

Before asking whether recursion can exploit residual uncertainty, establish that residual uncertainty *costs anything*. Three arms, MLP only, Pendulum-v1, DDPG, 200 episodes, seeds 1–6:

| Arm | Condition | Hydra overrides |
|---|---|---|
| A1 | fixed d=1 + augmentation (the Phase 1 condition; sufficient statistic) | `delay=fixed delay.value=1 augment_obs=true` |
| A2 | stochastic d ~ U{0..4} + augmentation (approximate) | `delay=stochastic delay.min=0 delay.max=4 augment_obs=true` |
| A3 | A2 + **delay oracle**: realised `d_t` appended to the observation | `... delay.expose_realised=true` |

`info["delay_sampled_steps"]` already carries the realised delay; exposing it in the observation is ~10 lines in `DelayedEnv` (engineering item E-4). Arm A3 is an *information upper bound*: the best any architecture could do by resolving the delay, handed over for free rather than inferred.

```bash
python3 main.py arch=ddpg model=dnn env=pendulum \
  delay=stochastic delay.min=0 delay.max=4 augment_obs=true \
  arch.total_episodes=200 seed=$S wandb.enabled=false \
  print_stat_every=20 record_every=1000 eval_every=10 \
  results_folder=results_p3s0 track_step_history=true
```

PBS: `select=1:ncpus=2:mem=8gb`, `walltime=04:00:00`, queue `mech12`. ≈4 CPU-hours total.

**Measured:** final avg100 return; the `eval_return` curve at 10-episode resolution; per-seed paired differences A3 − A2 and A2 − A1.

**Statistic:** one-sided sign test on per-seed A3 − A2 (n = 6), plus the effect size in reward units against the pooled across-seed standard deviation.

> **Gate G0.** A3 must beat A2 by more than the pooled seed spread, sign-consistent in ≥5 of 6 seeds. If it does not, there is no headroom for inference to recover, **no architecture can win at this condition**, Stage 2 is void, and the phase ends after Stage 1. Cost of learning this: about four CPU-hours.

#### S0.2 — Checkpoint battery (0 training jobs)

204 checkpoints totalling 1.6 GB sit in `/srv/scratch/z5308482/trm-control/models`. Each run's `model_id` in its results JSON maps to one file; **all 102 Phase 1–2 runs were confirmed present** (the remaining 102 files are earlier archived runs). Every trained policy from both phases can be re-evaluated without retraining anything.

Per checkpoint, deterministic policy, no exploration noise:

| Probe | Grid | Yields |
|---|---|---|
| Nominal performance | 20 episodes, unseen seeds | IQM return + stratified bootstrap CI (Agarwal et al., 2021) |
| Command quality | same rollouts | ∫u²/T, mean\|Δu\|, mean\|Δ²u\|, saturation %, reversals/s, HF power fraction, mechanical work ∫\|u·ω\| |
| Delay mismatch | test-time d ∈ {0,1,2,3,4,5} against a policy trained at d=1 | degradation curve, empirical delay margin |
| Sensor noise | Gaussian σ ∈ {0, 0.01, 0.05, 0.10} × obs scale | noise-rejection curve |
| Plant perturbation | Pendulum m, l ∈ {0.7, 0.85, 1.0, 1.15, 1.3}× | model-mismatch curve |
| Actuator dropout | p ∈ {0, 0.05, 0.10} | fault tolerance |
| Inference latency | CPU 1 thread batch 1; GPU batch 1; batch 256 | p50/p99 ms per decision, max control rate |

PBS: `select=1:ncpus=2:mem=8gb`, `walltime=00:30:00`, queue `mech12`, ~102 jobs. The latency benchmark runs as two dedicated jobs with `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` so the batch-1 timing is not contaminated by thread scheduling. ≈25 CPU-hours total.

**Deliverable:** `analysis_w0.json`; the deployment scorecard; return-versus-effort Pareto front; degradation curves; the latency-versus-control-rate table. This tests H3.4–H3.6 and produces the whole of §3 before any new training job is submitted.

**Gate:** none. Stage 0.2 runs regardless of G0 and its output survives every branch of the plan.

### 4.2 Stage 1 — Attribution: capacity or recursion?

**Entry:** none. Not gated by G0 — its question stands whatever the headroom probe says, and its answer is needed to configure Stage 2.

Pendulum-v1, DDPG, 200 episodes (matching Phase 1 exactly so the existing runs remain comparable), conditions {no-delay, fixed d=1 + aug}, seeds 1–6. Six arms forming a 2×2 in architecture × capacity plus two depth arms at identical parameter count:

| Arm | Config | Params (Pendulum, aug) | Role |
|---|---|---:|---|
| `dnn` | `[256,256]` | 134,914 | existing small MLP |
| `trm_small` | `hidden_size=44, L_layers=2` | **135,654** (+0.55%) | recursion at MLP capacity |
| `dnn_large` | `[624,624]` | 788,114 (+0.03%) | MLP at recursive capacity |
| `trm` | `hidden_size=128` | 787,842 | existing recursive trunk |
| `trm_h1` | `trm`, `H_cycles=1` | 787,842 | **recursion off, parameters identical** |
| `trm_h6` | `trm`, `H_cycles=6` | 787,842 | double depth, parameters identical |

Parameter counts are derived from the trunk's exact construction (two SwiGLU blocks at the source's inner width, plus input projection and head) and the formula is verified exactly against two measured runs — TRM on Pendulum (787,842) and on BipedalWalker with augmentation (794,757). `dnn_large.yaml` already exists with the same verification; `trm_small.yaml` is one file. Under stochastic delay (`max_delay=4`, obs dim 7) the matching holds: 135,918 vs 136,450 and 788,610 vs 791,858.

`trm_h1` is the cleanest control in the entire project: it removes the outer recursion while changing *nothing* about parameter count, optimiser, or input. `H_cycles` also sets the compute directly — 7 cell applications per forward pass at `H_cycles=1`, 21 at 3, 42 at 6, with the gradient reaching only 7 in every case.

**Jobs:** 6 arms × 2 conditions × 6 seeds = **72**. MLP arms `mech12` (`ncpus=2:mem=8gb`, 04:00:00); TRM arms `mechgpu12` (`ncpus=2:mem=16gb:ngpus=1`, 06:00:00). ≈60 CPU-h + 25 GPU-h.

**Measured:** final avg100 return; the full critic/actor-loss telemetry re-run through the existing `analyze_mechanism.py` pipeline at each capacity level; wall-clock; per-decision latency for each arm (folded in from the S0.2 benchmark).

**Statistic:** paired sign tests within each capacity level (n = 12 per contrast, pooling both delay conditions), plus the 2×2 interaction of architecture × capacity on median return.

> **Gate G1 — attribution.** No kill branch; both outcomes are informative and both change the report.
> - `dnn_large` degrades toward `trm` **and** `trm_small` improves toward `dnn` ⇒ the Phase 1 headline is largely a capacity artefact, must be restated in the report, and Stage 2 runs at matched capacity only.
> - Both MLPs equal and both TRMs equal-and-worse ⇒ the deficit is architecture-family; H_cap weakens and the recursion-specific account survives into Stage 2.
> - `trm_h1` ≈ `trm` ⇒ depth is inert at this delay condition, reproducing Roye-Azar et al.'s "shallow effective recursion" outside its original domain.

### 4.3 Stage 2 — The decisive test: uncertainty × memory

**Entry:** G0 passed, and G1 has fixed the capacity level.

Four trunks:

| Trunk | What it isolates |
|---|---|
| `dnn` at the capacity G1 identifies as fair | the memoryless baseline |
| `trm` (Reset-Z, the current port) | recursion without cross-step memory |
| `trm_carryz` | recursion *with* cross-step memory — the C-2 fix |
| `lstm`, hidden state genuinely carried, tuned per Ni et al. (2022) | the honest memory baseline; a beaten recurrent baseline that was never tuned proves nothing |

Three conditions, ordered by residual uncertainty:

| | Condition | Sufficient statistic? | Overrides |
|---|---|---|---|
| C1 | fixed d=1 + aug | yes, proven and verified | `delay=fixed delay.value=1 augment_obs=true` |
| C3 | stochastic d ~ U{0..4} + aug | no — documented approximation | `delay=stochastic delay.min=0 delay.max=4 augment_obs=true` |
| C4 | stochastic d ~ U{0..4}, no aug | no — information withheld entirely | `... augment_obs=false` |

Seeds 1–6. **4 × 3 × 6 = 72 jobs.** C0 (no delay) and C2 (fixed, no aug) are already covered by Stage 1 and Phase 2 respectively. ≈70 CPU-h + 40 GPU-h.

**Pre-registered primary statistic**, fixed before submission: the interaction contrast

> Δ = [median(TRM_carryz) − median(DNN)] at C3 − [median(TRM_carryz) − median(DNN)] at C1

tested by a one-sided sign test on per-seed paired differences, n = 6. **Secondary:** Carry-Z vs Reset-Z at each condition, which isolates C-2 independently of the delay axis; and the C4 replication of the same contrast.

**Optional C5 (+24 jobs).** `delay=time_varying` with the existing schedule makes the delay *non-stationary* rather than merely random — the regime in which Derman et al. (2021) prove stationary policies over augmented states are sub-optimal in general. This is the sharpest version of the test and the most defensible place in the project for a recursive trunk to win. Recommended if C3 shows any interaction at all.

> **Gate G2 — interaction.** The contrast must flip sign from C1 to C3, sign-consistent in ≥5 of 6 seeds. If it does not, **H3 is rejected**: report H_null with all three confounds removed, which is a stronger and more final negative than the current draft. Stages 3 and 4 do not run.

### 4.4 Stage 3 — Does the finding generalise?

**Entry:** G2 shows an interaction.

**S3.1 — Algorithm cross-check (24 jobs).** DDPG is documented as brittle on two of our three environments, and its OU exploration is itself a confound on the command-regularity measure of §5. `src/algos/td3.py` is implemented and has never been used by this project. Re-run the decisive contrast — 2 trunks (`dnn`, `trm_carryz`) × {C1, C3} × 6 seeds — under `arch=td3`. If the interaction reproduces, the finding is not DDPG-specific; if it inverts, that is a finding in itself. ≈25 GPU-h.

**S3.2 — A second rung chosen for control-theoretic reasons (9 pilot + 48).** Replace BipedalWalker-v3 with **InvertedDoublePendulum-v5**. The rationale is specific rather than "harder": it is an *open-loop unstable* plant with a short time constant, so delay directly consumes stability margin — delay does not merely obscure the state, it destabilises. It gives natural control metrics (time-to-fall, RMS pole angle, control effort), trains fast, and MuJoCo has been Apache-2.0 and pip-installable for years, so the licence gating recorded in earlier phases is stale and should be rechecked rather than inherited (engineering item E-7).

*Pilot first:* 1 trunk × 3 seeds × budget ∈ {200, 600, 1500} episodes, to find the budget at which the baseline reaches the solved threshold. Phase 2's floor effect was caused precisely by skipping this step; no full matrix is submitted until a pilot shows the environment is learnable within budget. Then 4 trunks × {C1, C3} × 6 seeds = 48. ≈50 GPU-h.

**S3.3 — Retire the floor-effect environments.** MountainCarContinuous-v0 and BipedalWalker-v3 stay in the report as documented failure modes (Matheron et al., 2019) but are not re-run at higher budget: a third environment adds nothing until the mechanism question is settled on a rung that works.

> **Gate G3 — continuity.** Carry-Z must beat Reset-Z somewhere (evaluated on Stage 2 data, re-checked on Stage 3). If it does not, Stage 4 does not run and precondition P1 loses its constructive test.

### 4.5 Stage 4 — Restore the training discipline

**Entry:** G3 passed — refinement continuity has traction.

Three variants of the recursive trunk at fixed parameters, {C1, C3}, 6 seeds = **36 jobs**, ≈45 GPU-h:

- **(a) No truncation.** Backpropagate through all outer cycles. Currently 14 of 21 cell applications receive no gradient at all.
- **(b) Per-cycle supervision.** Apply the critic loss to every outer cycle's readout — a deep-supervision surrogate (Lee et al., 2015) for the mechanism ARC Prize (2025) identify as HRM's true driver, and the closest available analogue to TRM's deep supervision over `halt_max_steps`.
- **(c) Fixed-point / DEQ readout.** Iterate to convergence and differentiate implicitly (Bai et al., 2019), removing truncation entirely at constant memory. The most scientifically interesting and the most speculative.

This is the only stage that can produce a *constructive* positive result — the mechanism transfers when its training discipline is ported with it — rather than a conditional one.

### 4.6 Stage 5 — Synthesis

No compute. Report revisions: §6.1 restated per G1; a new command-quality and robustness section built on the Stage 0 scorecard; the cost ledger reported three ways (matched episodes, matched wall-clock, matched inference latency); §10's E1–E5 superseded by whichever stages ran; and the threats-to-validity section updated to record which of the three confounds were actually removed.

### 4.7 Gate summary

| Gate | Test | Evaluated on | If it fails |
|---|---|---|---|
| **G0 — headroom** | oracle-delay MLP > augmented MLP at stochastic delay, > seed spread, ≥5/6 seeds | Stage 0 | Stage 2 void. Phase ends after Stage 1; write up. |
| **G1 — attribution** | 2×2 architecture × capacity contrast | Stage 1 | No failure branch; both outcomes restate report §6.1 and set Stage 2's capacity. |
| **G2 — interaction** | (Carry-Z − MLP) flips sign C1→C3, ≥5/6 seeds | Stage 2 | H3 rejected. Report H_null with all three confounds removed. Stages 3–4 do not run. |
| **G3 — continuity** | Carry-Z > Reset-Z at any condition | Stage 2/3 | Stage 4 does not run; P1 loses its constructive test. |
| **G4 — practicality** | some trunk sits favourably on the return/effort/latency front | Stage 0 | Report the trade as unfavourable; the deployment recommendation stands with the MLP. |

### 4.8 Budget rollup

| Stage | Contents | Jobs | Est. cost | Committed? |
|---|---|---:|---|---|
| 0 | headroom probe + checkpoint battery | 18 + ~102 | ≈30 CPU-h | **yes, at sign-off** |
| 1 | capacity × depth factorial | 72 | ≈60 CPU-h + 25 GPU-h | on Stage 0 completion |
| 2 | uncertainty × memory (+24 optional) | 72 (+24) | ≈70 CPU-h + 40 GPU-h | G0 |
| 3 | TD3 cross-check + unstable-plant rung | 24 + 9 + 48 | ≈75 GPU-h | G2 |
| 4 | training discipline | 36 | ≈45 GPU-h | G3 |
| 5 | synthesis | 0 | — | always |
| | **total if every gate passes** | **279 training** (+24 optional) **+ ~102 eval** | ≈290 CPU-h + 185 GPU-h | |

All jobs are PBS submissions on Katana under group `MECH`, ≤12 h walltime, following `experiments/phase2_submit.sh` as the template. **Committed at sign-off: Stage 0 only — 120 jobs, ≈30 CPU-hours, no GPU allocation.** Everything after it is conditional on a decision rule stated here in advance. For comparison, Phases 1 and 2 together committed 102 jobs unconditionally.

**Seeds.** 6 per cell throughout, pooled where the design permits, giving a floor of p = 0.0156 on a clean sweep. Agarwal et al. (2021) show 5 runs per task yields Type I error above 50% for point estimates; IQM-with-bootstrap-CI reporting is the mitigation where seed counts cannot rise further, and no Phase 3 comparison runs at n = 3.

**What would make Phase 3 a success.** Not a positive result. Either of these is a complete outcome: (a) recursion wins where residual uncertainty exists and memory is carried, with a quantified cost ledger and control-quality scorecard saying whether the win is worth buying; or (b) it does not, with capacity crossed, the latent carried, the delay genuinely uncertain, and the environments validated — at which point the negative result is about recursive computation in continuous control rather than about our port of it.

---

## 5. Preliminary signal already in the banked data

Running the command-quality measure of §3.1 over the existing `step_history` — 102 runs, no new compute (`experiments/analyze_control_quality.py`) — turns up a pattern the project has not reported.

**Every non-MLP trunk commands a markedly smoother action sequence than the MLP.** Paired per seed and condition across Pendulum-v1 and BipedalWalker-v3 (MountainCarContinuous excluded as a floor effect), median ratio of mean |Δa| against the DNN baseline:

| Trunk | Smoother than DNN | Median ratio | one-sided p |
|---|---:|---:|---:|
| TRM | 13/15 | 0.42× | 0.004 |
| TRM-hier6 | 9/9 | 0.39× | 0.002 |
| TRM-singlez | 7/9 | 0.54× | 0.090 |
| Transformer | 13/15 | 0.10× | 0.004 |
| LSTM | 14/15 | 0.16× | 0.0005 |

On Pendulum-v1 alone, TRM commands 0.37× the DNN's action rate (5/6 seed-pairs, p = 0.109) while losing 37 reward units — roughly a **2.7× reduction in actuator activity for 3% of the performance span**. That is the shape of trade a controls engineer makes deliberately.

**Three reasons this is a candidate signal and not yet a finding**, all of which Stage 0.2 is designed to settle:

1. **The actions are exploration-noisy.** DDPG adds OU noise inside `_select_action`, so the statistic mixes policy behaviour with a common noise process. Stage 0.2 measures deterministic rollouts.
2. **A dead policy is maximally smooth.** The LSTM scores smoothest on Pendulum while returning −1460, i.e. untrained. Regularity is only interpretable jointly with return, which is why Stage 0.2 reports a Pareto front rather than a ranking.
3. **It is not recursion-specific either.** The LSTM and Transformer are smoother than TRM. The honest reading is that *the plain MLP is the outlier in command regularity*, and the interesting question is whether that costs it anything in a real actuator — not whether TRM is special. This is the same confound shape as §7.3 of the report, and it is reported that way rather than dressed up.

### 5.1 The cost ledger, restated the way a practitioner reads it

The report's headline deficit of 37 reward units is the generous reading, obtained by giving both trunks the same number of *episodes*. Giving them the same number of *minutes* — the resource a practitioner actually allocates — reads very differently. On Pendulum-v1, at the wall-clock moment the MLP's run has finished, the recursive trunk has completed 6–44 of its 200 episodes and sits at −488 to −1390, against an untrained policy's ≈ −1460:

| Reading | TRM − DNN on Pendulum-v1 | As a share of the untrained→best span |
|---|---:|---:|
| Matched episodes (as reported) | −37 | ≈3% |
| Matched wall-clock | ≈ −846 | ≈58% |

Both numbers are true and they answer different questions. Phase 3 reports both, on every comparison, and adds the third — matched *inference* latency — which is the only one that survives to deployment.

---

## 6. Exploration inventory

Everything the project could pursue, grouped by what it would settle. Status is one of **stage N** (on the critical path), **candidate** (worth doing, not scheduled), **done** (already answered), or **declined** (with the reason).

### A. Mechanism — what does recursion actually contribute?

| # | Question | What would answer it | Cost | Status |
|---|---|---|---|---|
| A1 | Is the measured deficit capacity or recursion? | 2×2 architecture × capacity at matched parameters | 48 jobs | **stage 1** |
| A2 | Does recursion depth help at fixed capacity? | `H_cycles` ∈ {1,3,6}; `H_cycles=1` removes recursion with parameters unchanged | 24 jobs | **stage 1** |
| A3 | Does carrying the latent across steps matter? | Carry-Z vs Reset-Z, both delay regimes | in stage 2 | **stage 2** |
| A4 | Does the gradient truncation hurt? | backprop through all outer cycles (currently 14 of 21 cell applications are untrained) | 12 jobs | **stage 4** |
| A5 | Does per-cycle supervision recover the benefit? | critic loss on every outer cycle's readout | 12 jobs | **stage 4** |
| A6 | Is there a principled alternative to truncation? | DEQ fixed-point readout with implicit differentiation | 12 jobs | **stage 4** |
| A7 | Can adaptive computation kill the cost premium? | ACT/PonderNet halting; if most decisions need one cycle, the 5–14× multiple collapses | ~24 jobs | candidate |
| A8 | Does per-instance conditioning transfer (precondition P3)? | under domain randomisation each episode *is* an instance; a learned per-episode embedding is system identification by another name | ~36 jobs | candidate |
| A9 | Does the recursive *structure* matter (hier6, singlez)? | already run: sign-inconsistent, noise-dominated deltas | — | **done** → declined |

### B. Conditioning — what is the agent told about the delay?

| # | Condition | Sufficient statistic? | Status |
|---|---|---|---|
| B1 | fixed d=1 + augmentation | yes — proven and verified in `augmentation.md` | **done** (Phases 1–2) |
| B2 | fixed d=1, no augmentation | no | **done** (Phase 2, BipedalWalker only) |
| B3 | stochastic d ~ U{0..4} + augmentation | no — documented approximation | **stage 0, stage 2** |
| B4 | stochastic d ~ U{0..4}, no augmentation | no — withheld entirely | **stage 2** |
| B5 | stochastic + delay oracle (realised `d_t` exposed) | yes, by construction — an information upper bound | **stage 0** |
| B6 | time-varying (non-stationary) delay | no; Derman et al. prove stationary augmented policies sub-optimal here | **stage 2, optional** |
| B7 | **observation delay** | not implemented — `DelayedEnv` delays actions only, though `project-config.yaml` names it as an axis | candidate, needs code |
| B8 | **jitter** | not implemented; no config group exists | candidate, needs code |
| B9 | longer delays, d ∈ {2, 4, 8} | augmentation's cost is exponential in the delay (Derman et al.); the regime where it should break down | candidate, ~36 jobs |
| B10 | test-time delay mismatch (train d=1, test d ∈ 0…5) | zero-cost from banked checkpoints | **stage 0** |

### C. Control practicality — is any of this a usable controller?

| # | Question | Measure | Status |
|---|---|---|---|
| C1 | How hard does each trunk work the actuator? | ∫u²/T, mean\|Δu\|, jerk, saturation %, reversals/s, HF power fraction | **stage 0** |
| C2 | How far can the plant drift before it fails? | mass/length sweep, sensor noise sweep, actuator dropout | **stage 0** |
| C3 | Can it close a real control loop? | per-decision latency at batch 1 on one CPU thread, p50/p99, against 100 Hz / 500 Hz / 1 kHz periods | **stage 0** |
| C4 | What is the actual trade? | return-versus-effort-versus-latency Pareto front, not a ranking | **stage 0** |
| C5 | What is the *delay margin*, as control theory defines it? | linearise the policy by finite-differencing its Jacobian at the operating point, close the loop with the linearised plant, compute gain/phase margin and the critical delay | candidate — highest-value item on this list |
| C6 | What do you get per hour of compute, not per episode? | reward at matched wall-clock, reported on every comparison | reporting mode, all stages |
| C7 | What does it cost in energy? | mechanical work ∫\|u·ω\| from the same rollouts | **stage 0** (cheap add) |
| C8 | Does it generalise off-distribution? | train under domain randomisation, evaluate on held-out parameters | candidate, nearly free once C2's wrappers exist |

### D. Task selection

| # | Environment | Why | Status |
|---|---|---|---|
| D1 | Pendulum-v1 | the only validated rung; every trunk demonstrably learns | **all stages** |
| D2 | InvertedDoublePendulum-v5 | open-loop unstable, short time constant: delay consumes stability margin rather than merely obscuring state | **stage 3** |
| D3 | budget pilot before any new rung | Phase 2's floor effect came from skipping exactly this | **stage 3** |
| D4 | Reacher-v5 (multi-joint tracking), HalfCheetah-v5 (high-dimensional) | natural tracking-error metrics; higher dimensionality | candidate |
| D5 | MountainCarContinuous-v0, BipedalWalker-v3 | floor effects at the budget run; retained in the report as documented failure modes | **decided** — retired from comparison |

### E. Learner

| # | Question | Status |
|---|---|---|
| E1 | Is the finding DDPG-specific? — TD3 cross-check on the decisive contrast (`src/algos/td3.py` exists, never used) | **stage 3** |
| E2 | Does a maximum-entropy learner change the picture? — SAC (`src/algos/sac.py` exists, never used) | candidate |
| E3 | Does exploration fix MountainCar? — parameter-space noise (Plappert et al.) instead of OU | declined — environment retired |
| E4 | Is a *tuned* recurrent baseline the real comparator? — Ni et al. show a tuned recurrent model-free agent matches specialised POMDP methods on 18 of 21 environments | folded into **stage 2** |

### F. Methodology

| # | Commitment | Status |
|---|---|---|
| F1 | 6 seeds minimum per cell; IQM + stratified bootstrap CI for evaluation reporting | all stages |
| F2 | One pre-registered primary statistic per stage, fixed before submission | all stages |
| F3 | Kill criteria written before compute is spent | this document |
| F4 | Every cost claim reported three ways: matched episodes, matched wall-clock, matched inference latency | all stages |
| F5 | Commit-after-each-unit discipline for code, per the epic's dispatch rules | all stages |

### G. Out of scope

| # | Item | Reason |
|---|---|---|
| G1 | Embedded deployment, quantisation, flash/SRAM/energy budgets | excluded by `project-config.yaml`; Stage 0's latency benchmark is the bridge if that scope reopens |
| G2 | Hardware-in-the-loop, real robot | same |
| G3 | TRM as a model-based MPC corrector (Jain et al.; Wijayatunga & Armellin) | a different project — this one is model-free by construction |
| G4 | A third arbitrary complexity rung | two axes have already failed to produce a signal; adds compute without a falsifiable question attached |

---

## 7. Engineering required before submission

| # | Work | Where | Risk | Est. | Needed by |
|---|---|---|---|---|---|
| E-1 | Evaluation harness: checkpoint loader (rebuild cfg from results JSON + `model_id`), deterministic rollouts, metric suite, perturbation grids | new `src/eval.py`, `experiments/w0_eval.py` | low | 2 d | stage 0 |
| E-2 | **Carry-Z with correct off-policy semantics** | `trm_control.py`, `ddpg.py`/`td3.py`, replay buffer | **high** | 3 d | stage 2 |
| E-3 | Perturbation wrappers: obs noise, plant-parameter scaling, actuator dropout | new `src/perturb.py` | low | 0.5 d | stage 0 |
| E-4 | Delay-oracle arm: append realised `d_t` to the observation | `delays.py` | low | 2 h | stage 0 |
| E-5 | Configs: `trm_small`, `trm_h1`, `trm_h6`, `trm_carryz`, `delay/stochastic_u4` | `conf/` | low | 1 h | stages 1–2 |
| E-6 | Latency benchmark (CPU 1-thread batch 1, GPU, p50/p99) | `experiments/bench_latency.py` | low | 0.5 d | stage 0 |
| E-7 | MuJoCo dependency check on Katana + `conf/env/invdoublependulum.yaml` | `setup/`, `conf/env/` | medium | 0.5 d | stage 3 |
| E-8 | Deep-supervision and no-truncation variants | `trm_control.py`, algo update step | medium | 2 d | stage 4 |
| E-9 | *(optional)* observation-delay and jitter wrappers (inventory B7–B8) | `delays.py` | medium | 1 d | unscheduled |

**E-2 is the plan's main technical risk and deserves its own paragraph.** Carrying a latent across environment steps is easy at acting time and wrong at update time: off-policy replay samples i.i.d. transitions with no latent attached, so the update would refine from a cold latent while the behaviour policy refined from a warm one. Three options, in descending preference:

1. **Stored state (R2D2-style).** Write the carry into the replay buffer alongside the transition and use the stored, slightly stale carry during updates. Cheapest, well-precedented, introduces representational drift as the trunk's weights move.
2. **Burn-in.** Store short sequences (k ≈ 8) and recompute the carry by rolling the trunk forward over the prefix under `no_grad` before the update. More faithful, roughly k× the update cost, requires converting `ReplayBuffer` to a sequence buffer.
3. **Carry at act, reset at update.** Rejected: it makes the acting and learning policies structurally different and would confound exactly the comparison Stage 2 exists to make.

Recommendation: implement (1), with (2) behind a config flag so the choice itself can be ablated if the result is close. Whichever is chosen must be recorded in the report as a design decision on the same footing as Reset-Z was — that is how C-2 happened in the first place.

---

## 8. Deliverables

1. `experiments/analysis_w0.json` + deployment scorecard, Pareto front, robustness-envelope figures, latency table (Stage 0).
2. `experiments/analysis_s1.json` + the 2×2 capacity attribution and depth sweep (Stage 1).
3. `experiments/analysis_s2.json` + the interaction contrast, pre-registered (Stage 2).
4. `experiments/analysis_s3.json`, `analysis_s4.json` if those stages run.
5. Report revisions per §4.6.
6. Code: `src/eval.py`, `src/perturb.py`, Carry-Z trunk and buffer changes, new configs, submission scripts — each committed as it lands, per the epic's commit-after-each-unit discipline.

## 9. Immediate next action

Approve or amend **Stage 0** — the 18-job headroom probe plus the zero-training checkpoint battery, ≈30 CPU-hours in total, no GPU allocation, no new training matrix. It is the cheapest possible test of whether the rest of the phase has a question to answer, and the battery delivers the entire control-practicality axis regardless of how the gate resolves.

---

## 10. Amendments before submission (2026-09-21)

Recorded at sign-off, before any Stage 0 job ran. Every pre-registered arm and statistic above still runs unchanged; these add to it.

**A-1 — Theory check on the headroom probe.** Under i.i.d. per-step delay d ~ U{0..4}, with a fully observed plant (Pendulum-v1 observes cos θ, sin θ and θ̇) and the four-slot pending queue, the augmented observation is already Markov. Every action that can reach the plant next step is either in the exposed queue or is the one about to be chosen, and d_{t+1} does not depend on the history. So `augmentation.md` §5 labels this case "approximate" too cautiously for this wrapper, and arm A3's realised *past* delay adds no decision-relevant information. Theory therefore predicts A3 ≈ A2, which would fail G0 by construction. The residual that inference *can* recover lives where the queue is withheld (C4: a memory-carrying trunk can rebuild its own recent actions) and where the delay is non-stationary (C5).

**A-2 — Arm A4 and an amended G0.** S0.1 gains arm A4: MLP, stochastic d ~ U{0..4}, **no** augmentation, seeds 1–6 (+6 jobs, 24 in total). G0 passes if *either* A3 − A2 (the original oracle headroom) *or* A2 − A4 (memory headroom: the information a perfect memory could recover at C4) meets the original criterion — above the pooled seed spread and sign-consistent in ≥5 of 6 seeds. Both contrasts are reported whatever the outcome.

**A-3 — Stage 2 primary contrast.** The C3 interaction stays pre-registered as written. Given A-1, the C4 replication is promoted to co-primary, and G2 passes on either condition under the same ≥5/6 criterion. Both are reported.

**A-4 — Exact discrete-uniform delay.** `delay=stochastic_u4` draws from the exact discrete uniform (`uniform_int`). The existing `uniform` distribution rounds a continuous U(0, 4) draw, which gives 0 and 4 half the mass of the interior values.

**A-5 — Compute placement.** At the author's direction, every Phase 3 job runs on a GPU node (`ngpus=1`), not on the CPU queues §4.1 names. The checkpoint battery is bundled into 12 shard jobs instead of ~102 single-checkpoint jobs. Latency runs as one GPU job that measures both CPU (one thread) and GPU. Each result records its device, so training wall-clock can be stratified by GPU model.

**A-6 — Stage 2 input design, fixed now.** Every Stage 2 trunk, the MLP included, receives the previous executed action appended to its observation (Ni et al., 2022). The trunks then differ only in architecture and memory. Under C1/C3 this input duplicates what the queue already exposes; under C4 it gives the memoryless MLP one step of the history that the memory trunks can integrate over many.

---

### References

Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS. arXiv:2108.13264 · ARC Prize (2025). *The Hidden Drivers of HRM's Performance on ARC-AGI.* · Bai, S., Kolter, J. Z. & Koltun, V. (2019). *Deep Equilibrium Models.* NeurIPS. arXiv:1909.01377 · Banino, A., Balaguer, J. & Blundell, C. (2021). *PonderNet.* arXiv:2107.05407 · Dang, L. H. & Rawlinson, D. (2025). *HRM-Agent.* arXiv:2510.22832 · Derman, E., Dalal, G. & Mannor, S. (2021). *Acting in Delayed Environments with Non-Stationary Markov Policies.* ICLR. arXiv:2101.11992 · Kapturowski, S. et al. (2019). *Recurrent Experience Replay in Distributed RL (R2D2).* ICLR · Katsikopoulos, K. V. & Engelbrecht, S. E. (2003). Markov decision processes with delays. *IEEE TAC* 48(4) · Klein, T. et al. (2024). *Plasticity Loss in Deep RL: A Survey.* arXiv:2411.04832 · Lee, C.-Y. et al. (2015). *Deeply-Supervised Nets.* AISTATS · Matheron, G., Perrin, N. & Sigaud, O. (2019). *The problem with DDPG.* arXiv:1911.11679 · Ni, T., Eysenbach, B. & Salakhutdinov, R. (2022). *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.* ICML. arXiv:2110.05038 · Ota, K., Jha, D. K. & Kanezaki, A. (2021). *Training Larger Networks for Deep RL.* arXiv:2102.07920 · Plappert, M. et al. (2018). *Parameter Space Noise for Exploration.* ICLR. arXiv:1706.01905 · Roye-Azar, A. et al. (2026). *Tiny Recursive Models on ARC-AGI-1.* arXiv:2512.11847
