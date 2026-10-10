# Phase 3 Rationale: Removing the Confounds

*trm-delay-robust-scaling · phase rationale · 2026-09-14 to 09-23 · 231 jobs*
*Why the phase existed, what inspired its design, and what it contributed. Results are in `../phase_summaries/phase3_summary.md`; labels are in `../glossary.md`.*

**Research question (the whole project).** Does weight-tied recursion (TRM), used as the actor-critic trunk of a model-free RL agent, benefit continuous control under action delay, and if not, why not?

## In one paragraph

After Phase 2 the honest position was not "recursion does not work" but "the experiment that would tell us has not been run". The Pendulum deficit could be capacity, the latent reset, or a task with nothing to infer. Phase 3 was designed to remove all three, and to give recursion its best fair chance: a hypothesis (H3) under which TRM *should* win, with a prediction no capacity story could fake. It also added the measurements a control engineer would ask for. The result was a confound-controlled negative: the deficit is architectural, memory and depth recover nothing, and TRM's one advantage, smoother torque, comes with worse robustness and 65× the latency.

## 1. Where the project stood going in

- A consistent Pendulum deficit (0/6) with a value-fit deficit (7/7 cells, p = 0.008).
- Three confounds: C-1 capacity, C-2 latent reset, C-3 nothing to infer.
- Preconditions P1–P3 for TRM's in-domain success, which the port lacks.
- 102 banked checkpoints from Phases 1–2 never evaluated as controllers.
- Phases 1–2 had committed 102 jobs before discovering that most could not answer the question.

## 2. Why this phase was made

1. **To attribute the deficit.** A deficit that might be parameter count says nothing about recursion. Matching capacity in both directions settles it.
2. **To give recursion its strongest case.** If TRM is a learned state estimator, it should help where (i) the input leaves something to infer and (ii) the trunk carries information across steps. Phases 1–2 satisfied neither. H3 states that, and its failure would leave only H_null.
3. **To judge TRM as a controller, not just a return.** A controls reader needs smoothness, robustness margins and real-time feasibility. None had been measured.
4. **To spend compute only when it could answer something.** Gated stages, each cheap enough to stop the phase.

## 3. Design decisions and what inspired them

| Decision | Inspiration | Why it mattered |
|---|---|---|
| H3 with an *interaction* as its load-bearing prediction (H3.2) | Standard logic of confound control: a capacity story cannot flip the sign of a contrast between identical networks across conditions | Carry-Z − MLP ≤ 0 with nothing to infer, > 0 with something to infer. A main effect could be capacity; a sign flip could not. |
| Competing accounts H_cap and H_null stated up front | Pre-registration practice | Every outcome had a named reading before the data. |
| Six gated stages with rules written before compute | Phases 1–2's sunk 102 jobs | Each stage could stop the phase; Stages 3–4 were in fact gated off when G2 failed. |
| Headroom first (S0.1, G0) | "Before asking whether recursion exploits uncertainty, check that uncertainty costs anything" | Hiding the queue cost the MLP 70.8 (6/6), so memory had something to recover. |
| The 2×2 capacity design plus H = 1 and H = 6 (Stage 1) | Growing the MLP alone only moves the comparison; shrinking TRM as well allows attribution | G1: TRM loses at 136k and 788k; MLP capacity inert. Depth inert too, since H = 1 removes the outer recursion with nothing else changed. |
| Carry-Z with stored-state replay | HRM-Agent (Dang & Rawlinson, 2025) for carrying the latent; R2D2 (Kapturowski et al., 2019) for replaying it | Carrying the latent at acting time but resetting it at update time would silently confound the test. Replay was verified to reproduce the behaviour policy exactly. |
| A carried LSTM arm | The standard recurrent POMDP baseline; Ni et al. (2022) | Separates "memory helps" from "TRM's memory helps". |
| Previous-action input for all trunks (A-6) | Ni et al. (2022) | Trunks then differ only in architecture and memory. |
| Amendment A-1: hide the queue (C4) | A theory check before submission: with a delay redrawn every step, the queue already contains every action that can land | An oracle arm could only be null; hiding the queue creates something genuine to infer. |
| Latent per-episode delay (C6, G2s) | An independent plan review (2026-09-17), read after execution began | A delay drawn once per episode is something memory can identify. Run after G2 failed, with its rule fixed before its data, labelled supplementary. |
| Control-quality battery and latency benchmark (S0.2) | Control-engineering practice: a controller is judged on its envelope, not its return | Measured torque reversals, robustness to noise, mass and dropout, and per-decision latency against control periods. |
| IQM and stratified bootstrap alongside sign tests | Agarwal et al. (2021) | Reporting standards for few-seed RL. |
| Frozen code snapshots, ledgers, verification before data | Lessons from Phase 2's filename collision | Every result traceable to exact code. Carried into Phases 4 and 5. |

## 4. How it contributed to the research question

| Hypothesis | Outcome | What it means for the question |
|---|---|---|
| H_cap, the deficit is capacity (G1) | **Rejected.** TRM 136k − MLP 135k −50.2 (0/12); TRM 788k − MLP 788k −27.1 (2/12); MLP 788k − 135k −3.8 (5/12) | The deficit is architectural. |
| H3.2, uncertainty × memory (G2) | **Rejected.** Wrong sign at C4 (−40.0, 1/6) | Recursion does not exploit residual uncertainty here. |
| Carrying the latent (G3, G2s) | **Fail.** Carry-Z never beats Reset-Z in ≥ 5/6 | Confound C-2 removed; nothing recovered. |
| Depth | Inert: H = 1 and H = 6 vs H = 3 give −3.3 (5/12) and −10.9 (4/12) | Extra recursion buys nothing. |
| Value fit | TRM's critic fits worse at matched capacity (1.45×, 1.89×, 12/12 each) | The mechanism of the deficit. |
| H3.4 smoothness | **Supported**: 0.4 vs 9 torque reversals/s (11/12, p = 0.006) | TRM's one advantage. |
| H3.5 graceful degradation | **Contradicted**: worse under sensor noise, a heavier plant, dropout | |
| H3.6 real-time feasibility | 4.13 vs 0.064 ms p99 per decision (65×); 100 Hz feasible, 500 Hz not | A deployment cost. |

Both halves of the question received their main answer here. **Does it benefit?** No, with capacity matched, the latent carried, and conditions built to leave something to infer. **Why not?** Not capacity, not memory, not depth; the deficit sits in value fitting and mostly in training (the final greedy policies differ by only −3.6, 2/12).

## 5. How it shaped the experiments that followed

Phase 3's gates kept its confirmatory claims clean, which left five questions open. Phase 4 took all five under its own pre-registration:

| Left open by Phase 3 | Phase 4 stage |
|---|---|
| TRM's training discipline (full gradient, per-cycle supervision) | Stage A (GA) |
| Was TRM ever tuned? | Stage C (GC) |
| Is the attribution DDPG-specific? | Stage T (GT), then Phase 5 |
| An unstable plant, and complexity | Stage B0 on InvertedDoublePendulum, Hopper fallback |
| Why is TRM smoother yet less robust? | Stage E (Lipschitz analysis) |

## 6. In hindsight

- **The plan review should have been read before execution.** Execution reached its central point independently (A-1) but not its recommended fix, which then had to run as a supplementary test.
- **The interaction design worked as intended:** it gave a result no capacity argument can explain away.
- **The battery changed the story's texture.** "TRM loses on return" became "TRM is a smoother, slower, more fragile controller", which is what an engineer needs to know.

## Sources

- Dang & Rawlinson (2025), HRM-Agent; Kapturowski et al. (2019), R2D2; Ni et al. (2022), recurrent model-free RL; Agarwal et al. (2021), RL evaluation statistics; Derman et al. (2021), non-stationary delays (planned C5, not run).
- Project files:
  - `phase3/plan/plan.md` (§§2–4, §6 exploration inventory, §10 amendments)
  - `phase3/report/`
  - `../full_report.md` §§8–9
