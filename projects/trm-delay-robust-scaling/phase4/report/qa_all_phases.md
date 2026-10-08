# Recursion Under Delay: Design Considerations, Inspirations and Questions, Phases 0–4

*trm-delay-robust-scaling · Q&A across the whole project · 2026-10-08*
*Part I tells the story of each phase and why each design decision was made. Part II maps every design choice to what inspired it. Part III is a question bank with answers and evidence. Labels are defined in `glossary.md`.*

<!-- toc -->

# Part I — The story, phase by phase, with the design reasoning

## 1. Where the idea came from

**Inspiration: TRM itself.** Tiny Recursive Models reached 45% on ARC-AGI-1 with about 7M parameters by applying one small weight-tied cell many times instead of adding parameters (Jolicoeur-Martineau, 2025). The claim underneath is about a *mechanism*, not about puzzles: iteration with tied weights can substitute for parameter count. That would matter most where parameters, memory and latency are scarce, which is embedded control.

**Why it looked transferable.** TRM is made of parts control already uses. Its cell is a small feed-forward network (an MLP), and applying it repeatedly to a latent makes it behave like a recurrent network. Those are the standard trunks of a model-free controller. So TRM could be swapped in as the actor and critic trunk with nothing else changed, which is what makes a clean like-for-like test possible.

**Design consideration: what to keep in the port.**
- **Kept:** the shared SwiGLU/RMSNorm cell, the two latents z_H and z_L, the 3 outer × 6 inner cycle structure, and the one-step gradient truncation.
- **Dropped:** token attention (a flat observation is a sequence of length 1, so attention degenerates), per-puzzle embeddings (replay transitions are not identified instances), and ACT halting with deep supervision (off-policy RL never revisits the same input across optimiser steps).

Two properties of the port turned out to matter all the way through: only 7 of 21 cell applications get gradient, and the latent resets on every call.

## 2. Why action delay is the test bed

**Inspiration: delayed-MDP theory** (Katsikopoulos & Engelbrecht, 2003). Under a d-step action delay the observation stops being a sufficient statistic. Appending the pending-action queue restores sufficiency in principle. But the theorem only says a sufficient statistic *exists*, not that a small network can *fit* the resulting value function efficiently.

**Design consideration.** If you want to give iterative refinement a fair chance, the task must contain something to refine. Delay supplies a principled reason: refinement could act like a learned state estimator over the queue. A harder locomotion task alone would only vary difficulty, not the property recursion is supposed to exploit.

## 3. Phase 0 — Make the comparison trustworthy

- **The pilot bug.** A Pendulum run had the MLP converging and TRM failing under a one-step delay. The cause was the wrapper: it delayed actions but never told the agent which one was about to land. Every stored transition was wrong. The queue-augmentation fix was verified exact for fixed delays, and an audit fixed a silent stochastic-delay bug.
- **Design consideration:** without that fix the project would have measured a logging bug, not an architecture. Every later phase re-verified that the wrapper's behaviour was unchanged.
- **The anecdote to test:** with the fix, all trunks reached "comparable reward", with TRM 18× slower to train.
- **Process inspiration: pre-registration.** The project ran as a gated research epic: literature review, three rounds of hypothesis review, then a plan with a fixed statistic. A regression design was rejected because its best possible p at n = 3 was 0.125.

## 4. Phase 1 — Test the obvious rescue

**Design consideration.** "Comparable reward" invited the rescue *"Pendulum is too easy; harder tasks will reward the extra computation."* Phase 1 made that falsifiable (H1) with a threshold:
- the horizon is 5× Pendulum's and the reward is sparse, which MountainCarContinuous provides;
- 4 trunks × 2 environments × 2 conditions × 3 seeds = 48 runs;
- a pooled sign test at n = 6, so a clean sweep reaches p = 0.016.

**Outcome:** the MLP won every Pendulum pair (p = 0.016) and the MountainCar gap closed without significance. Later, MountainCar turned out to be a floor effect.

## 5. Phase 2 — A second axis, and ablations

**Design considerations:**
- **A different kind of difficulty:** BipedalWalker's 24-D coupled dynamics. If only one axis had been tested, a null could always be explained as "the wrong kind of hard".
- **Structure ablations** (six parallel latents; a single latent), requested after Phase 1 review, ask whether a null concerns recursion in general or this recursive structure.
- **An augmentation on/off ablation** asks whether the delay fix matters more as tasks get harder.

**Outcome:** nothing significant in any direction.

## 6. The re-analysis that changed the project

Before writing Phase 2 up as a verdict, all 102 runs were re-analysed using telemetry already logged.

- **Two of three environments were floors.** MountainCar was the documented DDPG + OU-noise exploration failure (Matheron et al., 2019); no BipedalWalker run walked. The complexity question had never been put at risk.
- **TRM fit its critic worse** (7/7 cells, p = 0.008). So did a similar-sized Transformer, which pointed at capacity.
- **Three confounds:** C-1 capacity, C-2 latent reset, C-3 nothing to infer.
- **Inspiration for C-2: HRM-Agent** (Dang & Rawlinson, 2025). It compared re-initialising the latent with carrying it, in RL, and found carrying better. The port had implemented the weaker variant.
- **Inspiration for P1–P3: in-domain analyses of TRM and HRM.**
  - ARC Prize (2025): training-time refinement, not inference depth, drives HRM.
  - Roye-Azar et al. (2026): most accuracy arrives at the first recursion step, and removing the puzzle ID collapses accuracy.

  These suggested what the port lacks: refinement continuity (P1), a correctable structured answer (P2) and per-instance capacity (P3).

## 7. Phase 3 — Remove the confounds, decisions fixed in advance

**H3:** recursion pays only where there is something to infer *and* the trunk carries information across steps. Its key prediction was an *interaction*, chosen because no capacity story can flip the sign of a contrast between identical networks across two conditions.

**Design considerations:**
- **Gates instead of a big matrix.** Phases 1–2 committed 102 jobs and then found most could not answer the question. Phase 3 chained stages, each cheap enough to stop the phase.
- **Headroom first (S0.1):** before asking whether recursion exploits uncertainty, check that uncertainty costs anything.
- **The control-quality battery (S0.2).** Inspired by control engineering rather than RL: smoothness, robustness margins, latency. A controller is judged on its envelope, not just its return.
- **A theory check before submission (A-1).** With a delay redrawn every step, the queue already contains every action that can land, so the observation is Markov. An "oracle" arm could only be null. A condition that *hides* the queue (C4) was added instead.
- **The 2×2 (Stage 1).** Growing the MLP alone would only move the comparison to another single capacity; shrinking TRM too allows attribution. The H = 1 arm removes the outer recursion while changing nothing else.
- **Carry-Z with stored-state replay (Stage 2).** Inspired by R2D2 (Kapturowski et al., 2019). Carrying a latent at acting time but resetting it at update time would have silently confounded the test. Replay was verified to reproduce the behaviour policy exactly.
- **Previous-action input (A-6).** Inspired by Ni et al. (2022), so that all trunks differ only in architecture and memory.
- **The latent-delay test (G2s).** Inspired by an independent plan review: a delay drawn once per episode, which memory can identify. It was run after G2 failed, with its rule fixed before its data.

**Outcome:**
- Architecture, not capacity (G1).
- Depth inert.
- Carrying the latent does nothing (G3).
- Memory recovers nothing (G2, G2s).
- The final greedy policies nearly tie.
- TRM is smoother but less robust, and 65× slower per decision.

## 8. Phase 4 — Close the remaining doors

Phase 3 left five open questions. Phase 4's design took each in priority order.

- **Complexity, again (B).** InvertedDoublePendulum was chosen for control-theoretic reasons: it is open-loop unstable, so delay eats stability margin. Hopper was the fallback.
  - **Design consideration:** Phase 2's floors came from skipping a budget pilot, so B0 piloted the MLP first, with a "learned" bar fixed in advance at 90% of the solved return. A kill switch was written before any data.
  - Neither environment passed. A diagnostic then showed the one-step delay, not the budget, keeps the MLP unreliable on IDP.
- **Training discipline (A).**
  - **Inspiration:** TRM's own training, deep supervision across refinement steps (Lee et al.'s deeply-supervised nets as the RL surrogate) and no truncation.
  - **Design considerations:**
    - each flag is behind a default-off switch, verified bit-identical when off;
    - a gradient-hook test proved 21/21 applications get gradient with `full_grad`;
    - gradient clipping was set from a pilot's 99th percentile, so it only catches spikes;
    - the MLP runs were reused only after an exact code-equivalence check.
- **Tuning (C).** The first objection to any "architecture X loses" result is that X was never tuned. A 3×3 learning-rate grid answers it cheaply.
- **A second algorithm (T).** Phase 3 ran everything under DDPG. Repeating Stage 1 under TD3, with both conditions so the gate pools 12 pairs exactly like G1, tests whether the attribution is algorithm-specific.
- **Mechanism and deployment (E, D).**
  - **Stage E:** a Lipschitz analysis asks whether TRM's noise sensitivity comes from a sharper input–output map. Inspired by control robustness analysis.
  - **Stage D:** exports actors for the author's STM32. Inspired by the embedded-control motivation of TRM and TRC.

**Outcome:**
- TRM is insensitive to learning rate (GC).
- The training discipline does not help (GA fail).
- Under TD3, TRM 788k fails to train, so GT passes by optimisation failure.
- Complexity stays untested (GB0, KB0 fail; B1 skipped).
- Local sensitivity does not explain the robustness gap (E).

## 9. Why the recursion does nothing here: the TRC contrast

**Inspiration and contrast: Tiny Recursive Control** (Jain & Linares, 2025). TRC uses the same two-level latent recursion for optimal control, but on every pass it:
- simulates the known dynamics from its current control sequence;
- measures the tracking error;
- feeds that error back to correct the sequence;
- trains on exact solver labels, with full backpropagation and per-step improvement rewards.

Each pass gets new information. In this project's port, every pass sees the same observation, and the target is a moving TD estimate. Iteration with nothing new to integrate behaves like an expensive deep network whose extra depth goes unused. That fits every observation: depth is inert, value fit is worse, LR doesn't help, and full gradient plus deep supervision don't help. The author's research question deliberately assesses TRM's architecture *as it is*, so this is the explanation of the negative result, not a proposed redesign.

# Part II — Design choices and their inspirations

| Design choice | Inspiration / source | Phase |
|---|---|---|
| Weight-tied recursive trunk | TRM (Jolicoeur-Martineau, 2025); HRM | 0 |
| Action delay as the test bed; queue augmentation | Katsikopoulos & Engelbrecht (2003) | 0 |
| Dropping attention, ACT, puzzle embeddings | Analysis of what has no analogue in flat-observation off-policy RL | 0 |
| Positioning as model-free (vs model-based TRC, TRM-PL) | Jain & Linares (2025); Wijayatunga & Armellin (2026) | 0 |
| Pre-registration, sign tests, n = 6 floor | Research-epic review process; low-power concerns | 0–1 |
| MountainCar as the horizon/sparsity rung | H1's "5× horizon, sparse reward" threshold | 1 |
| BipedalWalker; structure and augmentation ablations | Supervisor feedback after Phase 1 | 2 |
| Floor-effect diagnosis | Matheron et al. (2019), DDPG exploration failure | 2 |
| Carry-Z vs Reset-Z | HRM-Agent (Dang & Rawlinson, 2025) | 3 |
| P1–P3 preconditions | ARC Prize (2025); Roye-Azar et al. (2026) | 2–3 |
| Stored-state replay for carried latents | R2D2 (Kapturowski et al., 2019) | 3 |
| Previous-action input | Ni et al. (2022) | 3 |
| Non-stationary delay as the hardest case | Derman et al. (2021) | 3 (C5, not run) |
| Control-quality battery, latency, robustness envelope | Control engineering practice | 3 |
| IQM + stratified bootstrap | Agarwal et al. (2021) | 3–4 |
| Latent per-episode delay (G2s) | Independent plan review | 3 |
| Deep supervision, no truncation | TRM's training; Lee et al. (2015); DEQ (Bai et al., 2019) as an unrun alternative | 4 |
| Budget pilot before any complexity matrix | Phase 2's floor effects | 4 |
| InvertedDoublePendulum as an unstable plant | Delay consumes stability margin | 4 |
| TD3 as the second algorithm | Fujimoto et al. (2018) | 4 |
| Lipschitz analysis | Control robustness / gain analysis | 4 |
| C-header export for STM32 | Embedded-control motivation (TRM, TRC) | 4 |

# Part III — Questions a supervisor or examiner might ask

## A. Motivation and framing

**Q1. Why TRM at all? Why would a puzzle model be relevant to control?**
Because its claim is architectural: iteration with tied weights substituting for parameters. That matters most in embedded control. TRM is also built from a small MLP cell applied repeatedly, so it swaps in as a trunk with nothing else changed.

**Q2. Why delayed control rather than a harder locomotion task?**
Delay breaks the Markov property, and augmentation restores it only in principle. Iterative refinement is a plausible inductive bias for fitting the resulting value function. A harder locomotion task varies difficulty, not the property recursion is supposed to exploit. (Phases 1, 2 and 4 also tried harder tasks; none was learnable by the baseline.)

**Q3. Isn't this just a negative result about your own port?**
After Phases 1–2 that was fair. Phases 3–4 removed each port-specific explanation in turn:
- capacity matched;
- latent carried, with verified replay;
- conditions with something to infer;
- learning rate swept;
- gradient truncation removed and every cycle supervised;
- a second algorithm.

The result survives all of them. What remains is TRM's *supervised* setting (exact labels, per-instance embeddings, test-time refinement), which off-policy RL cannot supply.

**Q4. HRM-Agent trained a recursive model with RL and it worked. Doesn't that contradict you?**
No. It is the closest precedent and the source of Carry-Z. It succeeded on discrete mazes, where multi-step planning over a revisited map rewards refinement. Ours is low-dimensional continuous control, where a two-layer MLP reaches the same greedy policy. Both can be true.

**Q5. How does your work relate to Tiny Recursive Control?**
TRC is model-based: each refinement pass simulates known dynamics, measures the error and corrects, trained on optimal-solver labels. Ours is model-free: no dynamics, no labels, the same observation every pass. TRC shows what recursion needs in order to help (new information per pass). Our results show what happens without it: depth inert, value fit worse. TRC also never compares against a same-size MLP trained on the same labels, so it doesn't yet show recursion beating a plain network on its own task.

**Q6. What is the contribution in one sentence?**
A confound-controlled negative: with capacity matched, the latent carried, inferable conditions, a learning-rate sweep, TRM's training discipline restored, and a second algorithm, weight-tied recursion brings no consistent benefit as a model-free actor-critic trunk on the task where every trunk learns. The deficit is architectural and visible in value fit.

## B. Design and method

**Q7. Why DDPG, and why add TD3 only in Phase 4?**
DDPG was the codebase's algorithm, and holding the learner fixed isolates the trunk. TD3 was planned as Phase 3's Stage 3 but gated on G2, which failed. Phase 4 ran it as Stage T because a single-algorithm result was the most obvious remaining threat.

**Q8. Why only Pendulum for the decisive tests?**
It is the only environment where every trunk demonstrably learns. MountainCar, BipedalWalker, InvertedDoublePendulum with delay, and Hopper were all unlearnable by the MLP baseline at the budgets tried. Running a decisive test on a floor would repeat Phase 2's mistake. The cost is external validity, stated as the first limitation.

**Q9. Six seeds is not many.**
Agreed. The gates detect *consistent* effects, not small ones:
- 5/6 gives one-sided p = 0.109;
- 12/12 pairs gives p = 0.0005.

Pooling conditions to n = 12 was used wherever the design allowed (G1, GT), and the battery evaluates 20 episodes per checkpoint.

**Q10. Why sign tests?**
They make no distributional assumption, which suits RL returns with heavy seed effects, and they were pre-registered in Phase 1. Medians, per-seed differences and bootstrap intervals are reported too.

**Q11. Your primary metric includes exploration noise.**
It was pre-registered for continuity, and the gates use it. Deterministic evaluation is reported alongside and tells a different story: the trunks nearly tie greedily. That localises the deficit to training, which is itself a finding.

**Q12. Did you tune TRM?**
Phase 4 did, within limits. A 3×3 actor × critic learning-rate grid (×0.3 to ×3) found no setting that beats the defaults consistently (GC), and raising the actor rate hurts. Depth (Phase 3) and capacity (both directions) are also inert. What has not been swept: batch size, schedules, and the TD3 settings that break TRM 788k.

**Q13. Why clip gradients in Stage A when no earlier run was clipped?**
Full-gradient training backpropagates through 21 applications of one tied cell, like an unrolled RNN, which is prone to exploding gradients. Clipping protects the comparison from a seed blowing up. The norms were set at the 99th percentile of truncated TRM's unclipped norms, so clipping fires on only 0.6–8.6% of updates. All three TRM arms were clipped the same way, so the GA contrasts compare like with like. Comparisons against the unclipped MLP carry that difference, and it is disclosed.

**Q14. How do you know the new Phase 4 code did not change earlier results?**
Every new behaviour sits behind a flag that is off by default. With the flags off, all 30 existing trunk × algorithm × condition configurations trained bit-identically (same weights, same histories) to the pre-Phase-4 code. A gradient-hook test confirmed `full_grad` actually reaches all 21 applications.

**Q15. How do you know Carry-Z was implemented correctly?**
Feeding the stored carries back reproduces the behaviour policy's actions and next carries with zero error, for actor and critic. Its parameter count equals Reset-Z's.

**Q16. Why did the complexity pilot use a "90% of solved" bar?**
"Learned" had to be fixed before data, or any failure could be redefined after the fact. 90% of the registered solved return is a conventional, external reference. When the pilot failed, the bar was not changed. A diagnostic ran instead, and the author chose not to run an exploratory version with a post-hoc bar.

## C. Results and interpretation

**Q17. What is the headline number?**
On Pendulum, TRM trails a parameter-matched MLP at both sizes: −50.2 at 135k (0/12) and −27.1 at 788k (2/12). Against the original small MLP it is −34.4 (0/12, p = 0.0005), reproducing Phase 1.

**Q18. Is that practically meaningful?**
On its own it is about 3% of the untrained-to-best span. But it is perfectly consistent, it comes with 11–13× training cost and 65× decision latency, and at matched wall-clock the gap is most of the span.

**Q19. Why is the value fit worse?**
It is not capacity: a 5.8× bigger MLP fits slightly better. It is not truncated gradient either: full gradient doesn't fix it (GA). And it is not learning rate (GC). The most consistent account is that the recursion receives no new information between passes. Extra passes are extra transformations of the same input, harder to optimise and not more informative.

**Q20. Depth is inert. Did you implement TRM wrongly?**
Phase 4 rules out the obvious implementation explanation: with gradient through all 21 applications and per-cycle supervision, depth still buys nothing. In-domain analyses of TRM and HRM also find most accuracy at the first recursion step. The pieces that make depth useful there (exact labels, per-instance embeddings, revisiting the same input) are what model-free RL lacks.

**Q21. Did restoring the training discipline do anything at all?**
Nothing consistent:
- full gradient − truncated: +2.5 (3/6) at C1 and +10.1 (4/6) at C4;
- with deep supervision: +0.3 (3/6) and +28.4 (4/6).

At C4 the discipline arms fit the critic about 12–15% better than truncated TRM, in 4/6 seeds, which is too inconsistent to claim. All arms still trail the MLP in 5/6 seeds, and training costs 1.45–1.5× more.

**Q22. GT passed. Does that mean Phase 3 replicated under TD3?**
The direction did, the magnitude did not. Under TD3's standard settings the MLPs train as well as under DDPG (−172 vs −174), but TRM 788k fails to train in 10/12 seeds: its critic loss rises through training, from about 100 to about 700. So the 12/12 pass is an optimisation failure. That counts against TRM in practice, but it is not "loses by about 30 units". The likely cause, TD3's 10× higher actor learning rate, is untested.

**Q23. Why did the larger TRM do worse than the smaller one under TD3?**
TRM 136k learned in 9/12 seeds and TRM 788k in 2/12 (contrast −669, 3/12). Under DDPG, size made no difference to TRM. A plausible but untested reading: the wider trunk is more sensitive to TD3's higher actor learning rate, consistent with Stage C's finding that raising TRM's actor rate hurts it.

**Q24. TRM is smoother. Is it a better controller for actuator wear?**
It is a trade:
- **Gains:** about 0.4 torque reversals per second against the MLP's 9 (11/12 pairs, p = 0.006), at a final return only 3.6 lower.
- **Costs:** it is less robust to sensor noise (−75.8, 2/12), a heavier plant and actuator dropout, and it is 65× slower per decision.

**Q25. Why is TRM less robust if it is smoother?**
Stage E asked whether TRM reacts more sharply to its inputs. It does the opposite: its local Lipschitz constant is 11–16 against the MLP's 29–34, yet it loses 1.6–3× more return under sensor noise. Local sensitivity does not explain the robustness gap. A finite perturbation passes through 21 nonlinear, normalised steps, or noise moves the plant into states the policy never trained on. Both are untested.

**Q26. What did the complexity experiments actually show?**
That no harder rung was usable:
- MountainCar and BipedalWalker (Phases 1–2) were floors.
- On InvertedDoublePendulum with a one-step delay, the MLP plateaus at about 30–45% fully solved episodes, even at 1M steps. Without delay, every seed reaches the bar at some point but does not hold it.
- Hopper peaked at 2805 against 3420.

H1 is therefore *untested*, a limitation, not a finding against TRM.

**Q27. Why was B1 skipped instead of run with a relaxed bar?**
- Any relaxed bar would have been chosen after seeing the data, so B1 could only be exploratory.
- IDP mixes complexity with delay sensitivity on an unstable plant.
- Stage T had just shown TRM 788k does not train under the same TD3 settings, so B1 would most likely have measured that failure again.
- Its planned Pendulum reference (Stage T) would have made "the deficit shrinks" pass almost automatically.

The author chose to record H1′ as not run (K-3).

## D. Validity, rigour and process

**Q28. How do I know you didn't fish for results?**
Every rule was written before its data and is reported verbatim with its outcome. Failures were kept as failures:
- G0's original contrast failed and only the amended one passed;
- G4 passed degenerately;
- GB0 and KB0 failed and their bars were never moved.

The one later bar proposal, "clear of floor" for B1, was drafted, labelled exploratory, and then not used.

**Q29. You amended the plans several times. Isn't that moving the goalposts?**
Every amendment is dated and was recorded before the data it affects, and none removed a pre-registered arm. Phase 4's amendments either:
- recorded checks (R-1, V-1);
- froze values from pilots (R-2, A-1);
- recorded the author's decisions after a gate (K-1, K-2, K-3);
- added a stage (T-1).

**Q30. Phase 3 stopped at Stage 2; Phase 4 ran what Phase 3 gated off. Isn't that inconsistent?**
Phase 3's gates kept *confirmatory* claims clean. Phase 4 was a new pre-registered phase, with its own rules fixed before its data, answering questions the Phase 3 write-ups named as open. Its results are labelled by their own gates, not by Phase 3's.

**Q31. Multiple comparisons?**
Each stage has one pre-registered primary statistic. GA has four contrasts at one-sided p = 0.109 each, disclosed in advance as a multiplicity risk; it failed anyway. GC tests eight cells at n = 3 with a magnitude condition; it flagged nothing. Secondary contrasts are labelled secondary.

**Q32. Is it reproducible?**
- Every stage ran from a frozen code snapshot.
- Every result embeds its full configuration and GPU model.
- Warmup actions have been seeded since Phase 3.
- Phase 4 found that runs are not bit-reproducible *across GPU models*: the same seed on an L40S and an A100 diverges. Comparisons are therefore paired by seed, and costs are compared within one GPU model.

**Q33. What went wrong operationally?**
- **Phase 3:** an unsupported GPU node, a queue-tier resubmission, and a battery filename collision.
- **Phase 4:**
  - two jobs got PBS exit −19 after finishing cleanly (counted done, results verified complete);
  - the first Stage E submission failed on a snapshot path bug and was rerun;
  - SSH to Katana briefly refused the key.

None produced silent wrong numbers.

## E. Bigger picture

**Q34. What is the single most important thing you learned?**
That the question moved from "does recursion beat an MLP?" to "what makes recursion useful?". The evidence says it is not forward-pass depth, nor gradient through that depth, nor supervising each cycle. Recursion needs something new to integrate on each pass, which TRM's supervised setting and TRC's dynamics feedback provide, and a model-free trunk does not.

**Q35. What would you do differently?**
- Pilot every environment's budget before committing any matrix.
- Read the plan review before executing.
- Treat the previous-action input as a factor.
- Check each algorithm's default settings against every trunk with a short pilot before a full stage. Stage T's TD3 failure would have surfaced in one run.

**Q36. What would falsify your conclusion?**
Any one of these:
- a tuned TRM beating a tuned MLP at matched capacity on a validated environment;
- a TRM variant recovering the value-fit gap under DDPG and TD3;
- a carried-latent trunk exploiting delay uncertainty that a memoryless one provably cannot.

The last two were tested and failed.

**Q37. What remains open?**
1. Complexity scaling on an environment the baseline actually learns: delay-free IDP, or a better-tuned learner.
2. TRM under TD3 with an actor learning rate matched to DDPG's.
3. A tuned recurrent baseline.
4. Why TRM is less robust despite a smaller local gain.

**Q38. Is anything here publishable on its own?**
- The confound-controlled negative.
- The depth-inertness and value-fit result, now with the training discipline ruled out.
- The smoothness-versus-robustness trade, a control-engineering measurement the RL-architecture literature rarely makes.

---

## Appendix — Numbers worth memorising

| Fact | Number |
|---|---|
| Phase 1 headline (TRM − MLP, Pendulum) | −37.1; 0/6; p = 0.016 |
| Same contrast in Phase 3 Stage 1 | −34.4; 0/12; p = 0.0005 |
| TRM vs matched MLP (135k / 788k) | −50.2 (0/12) / −27.1 (2/12) |
| MLP capacity effect | −3.8; 5/12 |
| Depth (H = 1 / H = 6 vs H = 3) | −3.3 (5/12) / −10.9 (4/12) |
| Value fit at matched capacity | 1.45× / 1.89×; 12/12 each |
| Deterministic final-policy gap | −3.6; 2/12; p = 0.039 |
| Memory headroom (queue hidden) | 70.8; 6/6 |
| G2 interaction at C4 | −40.0; 1/6 |
| Carry-Z − Reset-Z (C1 / C3 / C4 / C6) | +4.8 / −22.0 / +16.7 / −4.9 |
| Torque reversals/s (TRM vs MLP) | 0.38 vs 9.06; 11/12 |
| Decision latency, 1 CPU thread p99 | 4.13 vs 0.064 ms (65×) |
| LR sweep (GC) | default −215; best other cell −209; none flagged |
| Training discipline (GA) | best 4/6 vs truncated; 1/6 vs MLP |
| TD3 (GT) | TRM 788k −1389 vs MLP 788k −176; 12/12; TRM 788k learns 2/12 |
| IDP MLP, C1 | 0/3 learned at 150k–1M; no delay: all seeds reach 8190 transiently |
| Lipschitz (TRM vs MLP) | 11–16 vs 29–34; noise drop 157–257 vs 81–100 |
| Flash (FP32 / INT8) | MLP 135k 263 / 66 KB; TRM H = 1 1540 / 386 KB |
| Totals | about 441 training runs (300 through Phase 3, 141 in Phase 4); Phase 3: 231 jobs; Phase 4: 154 jobs |
