# Phase 2 Summary: Dimensionality, Coupling and the Re-analysis

*trm-delay-robust-scaling · phase summary · 2026-09-09/10 · 54 runs*
*Labels are defined in `../glossary.md`.*

## Question

Does a different kind of difficulty, high-dimensional coupled dynamics, reward recursion? And is any null about recursion in general or about this particular recursive structure?

## Design

- **Environment:** BipedalWalker-v3 (24-D observation, 4-D action, coupled multi-joint dynamics).
- **Trunks (6):** MLP, Transformer, LSTM, TRM, plus two TRM variants requested after Phase 1 review:
  - `trm_hier6`, with six parallel low-level latents;
  - `trm_singlez`, with a single latent.
- **Conditions (3):**
  - no delay;
  - fixed d = 1 + queue;
  - fixed d = 1 without the queue (the augmentation ablation, C2).
- **Seeds:** 3, so 54 runs on V100 GPUs. A filename collision silently overwrote three MLP cells; it was caught before analysis and six jobs were re-run.

## Results

- **No comparison reached significance in any direction:** TRM − MLP 2/3, TRM − Transformer 0/3, TRM − LSTM 1/3.
- **The recursive structure made no detectable difference.** Base, hier6 and singlez reversed order from seed to seed.
- **Augmentation had no clear effect.**
- **Cost:** TRM 5.2×, hier6 6.9×, singlez 8.0× the MLP's wall-clock.

## The post-hoc re-analysis (all 102 Phase 1–2 runs, no new compute)

It changed what both phases meant.

1. **Two of three environments were floor effects.** MountainCar was exploration-limited (median −8.9 against a solved +90, the documented DDPG + OU-noise failure). No BipedalWalker run learned to walk (best −81.1 against +300). **The complexity question had never been put at risk: untested, not disconfirmed.**
2. **TRM fit the value function worse** (critic loss higher in 7/7 cells early, median 2.72×, p = 0.008) at unchanged learning speed (117 vs 115 episodes to 90% of span). This was explanation E1, an optimisation/conditioning deficit.
3. **The deficit looked like capacity.** The 797k Transformer showed it too and the 167k LSTM did not.
4. **Three confounds**, any one of which could explain the Phase 1 negative:
   - **C-1, capacity:** 788k vs 135k parameters.
   - **C-2, latent reset:** the port re-initialised its latent every call, HRM-Agent's weaker variant.
   - **C-3, nothing to infer:** under fixed delay + queue the observation is already sufficient.
5. **Preconditions P1–P3** for TRM's in-domain success, which the port lacks: refinement continuity, a correctable structured answer, per-instance capacity.

The report was rewritten after author review rejected the first draft, then the literature review was integrated (26 sources). That retracted the novelty claim, because HRM-Agent precedes this work.

## What it established

The honest position after Phase 2 was not "recursion does not work". It was "the experiment that would tell us has not been run yet". Removing C-1, C-2 and C-3 became Phase 3's job. The recommended experiments E1–E5 became Phase 3 Stages 1–2 and Phase 4 Stage A.
