# Phase 2 Rationale: A Second Axis, Ablations, and the Re-analysis

*trm-delay-robust-scaling · phase rationale · 2026-09-09/10 · 54 runs plus a re-analysis of all 102*
*Why the phase existed, what inspired its design, and what it contributed. Results are in `../phase_summaries/phase2_summary.md`; labels are in `../glossary.md`.*

**Research question (the whole project).** Does weight-tied recursion (TRM), used as the actor-critic trunk of a model-free RL agent, benefit continuous control under action delay, and if not, why not?

## In one paragraph

Phase 1 left TRM behind on Pendulum and level on MountainCar. Two objections remained: maybe horizon and sparsity were the wrong kind of difficulty, and maybe a null said something about this particular recursive structure rather than recursion in general. Phase 2 tested both on BipedalWalker with structural and augmentation ablations, and found nothing significant. Its lasting contribution came afterwards: a re-analysis of all 102 Phase 1–2 runs showed that two of the three environments were floors, that TRM fit its value function worse, and that three confounds could each explain the Pendulum deficit. That reframed the project from "does TRM catch up on harder tasks" to "is the deficit about recursion at all".

## 1. Where the project stood going in

- TRM − MLP: 0/6 on Pendulum (p = 0.016), 3/6 on MountainCar (p = 0.66).
- H1's threshold falsified, but the MountainCar gap had narrowed in the predicted direction.
- Four candidate explanations for the Pendulum deficit, none separated: recursion, capacity, the latent reset, nothing to infer.

## 2. Why this phase was made

1. **One axis is not enough.** If complexity is only varied along horizon and sparsity, any null can be explained away as "the wrong kind of hard". High-dimensional coupled dynamics is a different kind of difficulty, and one where iterative refinement of a joint latent is at least plausible.
2. **Recursion in general, or this recursion?** If TRM's two-latent structure was simply a poor fit, a different recursive structure might do better. Ablations answer that cheaply.
3. **Does augmentation matter more on harder tasks?** If the queue becomes harder to use as dimensionality grows, a trunk that can refine over it might gain.

The re-analysis was not planned. It was done before writing Phase 2 up as a verdict, because the results did not look like a clean null and the logged telemetry could be checked without new compute.

## 3. Design decisions and what inspired them

| Decision | Inspiration | Why it mattered |
|---|---|---|
| BipedalWalker-v3 as the second rung | A different difficulty axis from Phase 1: 24-D observation, 4-D action, coupled underactuated dynamics | Separates "harder" from "longer and sparser". |
| `trm_hier6` (six parallel low-level latents) and `trm_singlez` (one latent) | Requested by the author after Phase 1 review | Ask whether the result depends on TRM's particular latent structure. |
| Augmentation off (C2: fixed d = 1, no queue) | Phase 0's augmentation fix | Tests whether the queue helps more, or matters differently by trunk, on a harder task. |
| V100 GPUs | Wall-clock on the bigger environment | Made the 54-run matrix feasible; cost was again logged per run. |
| Post-hoc re-analysis using logged telemetry | Phase 1's v2/v3 logging, and the floor-effect literature (Matheron et al., 2019) | Showed which environments had actually been learned before drawing conclusions from them. |
| Comparing critic loss, not just return | The value function is what a delayed agent must fit; augmentation theory says a statistic exists, not that it is easy to fit | Produced the only significant positive detection in Phases 1–2: TRM's critic fits worse (7/7 cells early, median 2.72×, p = 0.008). |
| Reading the deficit against parameter count | The 797k Transformer and 167k LSTM as natural capacity references | The Transformer shared the value-fit deficit and the LSTM did not, which pointed to capacity. Phase 3 later overturned that reading. |
| Preconditions P1–P3 | In-domain analyses of TRM and HRM (ARC Prize, 2025; Roye-Azar et al., 2026) | Named what TRM's success relies on and the port lacks: refinement continuity (P1), a structured correctable answer (P2), per-instance capacity (P3). |
| Integrating the literature review into the report | Author review, which rejected the first draft | Brought in HRM-Agent (Dang & Rawlinson, 2025), which retracted the novelty claim and identified the latent reset as the weaker of its two variants. |

## 4. How it contributed to the research question

| Before Phase 2 | After Phase 2 |
|---|---|
| H1 threshold falsified on MountainCar | MountainCar and BipedalWalker were floors (median −8.9 vs a solved +90; best −81.1 vs +300). H1 is **untested**, not disconfirmed. |
| Recursive structure untested | Base, hier6 and singlez reversed order from seed to seed. No structure signal (though on a floor). |
| "Why not" unknown | TRM fits the critic worse at unchanged learning speed (E1: an optimisation or conditioning deficit). |
| Four loose explanations | Three named confounds: **C-1** capacity (788k vs 135k), **C-2** latent reset, **C-3** nothing to infer under fixed delay + queue. |

For the "does it benefit" half, Phase 2 added no evidence, because its rung was never learned. For the "why not" half it was decisive: it located the deficit in value fitting and turned three vague alternatives into three specific confounds, each removable by design.

## 5. How it shaped the experiments that followed

| Phase 2 output | Where it went |
|---|---|
| C-1 capacity | Phase 3 Stage 1: a 2×2 of {MLP, TRM} × {135k, 788k} (gate G1) |
| C-2 latent reset | Phase 3 Stage 2: Carry-Z versus Reset-Z, with stored-state replay (gate G3) |
| C-3 nothing to infer | Phase 3: stochastic delays and a hidden queue (C3, C4), then a latent per-episode delay (G2s) |
| Value fit worse (E1) | Tracked as critic-loss ratios in Phases 3, 4 and 5 |
| P1, refinement continuity | Phase 3 (persistent latent) and Phase 4 Stage A (full gradient, per-cycle supervision) |
| Floor effects | Phase 4 required a budget pilot with a fixed "learned" bar before any complexity matrix |
| Recommended experiments E1–E5 | Became Phase 3 Stages 1–2 and Phase 4 Stage A |

## 6. In hindsight

- **The design could not have reached significance.** At n = 3 per comparison the best possible p is 0.125, and the report said so. The ablations were therefore descriptive from the start.
- **The re-analysis was the most valuable work in the phase**, and it cost no compute. The habit it set, checking that the baseline learned before reading any contrast, ran through every later phase.
- **The capacity reading was wrong**, but it was the right hypothesis to test next, and Phase 3's design existed to test it.

## Sources

- Matheron et al. (2019), DDPG exploration failure on sparse-reward tasks.
- Dang & Rawlinson (2025), HRM-Agent.
- ARC Prize (2025), analysis of HRM; Roye-Azar et al. (2026), analysis of TRM.
- Project files:
  - `phase2/` reports
  - `experiments/analysis_phase2_final.json`
  - `../full_report.md` §§6–7
