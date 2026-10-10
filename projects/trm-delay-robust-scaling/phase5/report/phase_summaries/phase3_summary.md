# Phase 3 Summary: Removing the Confounds

*trm-delay-robust-scaling · phase summary · 2026-09-14 to 09-23 · 231 jobs*
*Labels are defined in `../glossary.md`. Detailed write-ups: `../../../phase3/report/`.*

## Question and hypothesis

> **H3, residual-uncertainty hypothesis.** Weight-tied recursion earns its cost only where (i) the agent's input leaves uncertainty that iteration could resolve, **and** (ii) the trunk carries information across decision steps.

Its load-bearing prediction (H3.2) was an *interaction*: a carried-latent TRM loses to the MLP when there is nothing to infer and wins when there is. No capacity story can flip that sign using identical networks. H3 was tested against H_cap (it's just parameter count) and H_null (recursion doesn't transfer).

## Design

- Six gated stages, each with a decision rule fixed before its data, run from frozen code snapshots on Katana GPUs.
- Six amendments recorded before any data. The key one, A-1: with a delay redrawn every step, the augmented observation is already Markov, so a condition that *hides* the queue (C4) was added.

| Stage | What it asked |
|---|---|
| S0.1 | Is there headroom for memory to recover? (gate G0) |
| S0.2 | What do the 102 banked checkpoints say as controllers? Latency. (gate G4) |
| Stage 1 | Capacity or architecture? {MLP, TRM} × {135k, 788k} + depth H = 1, 6 (gate G1) |
| Stage 2 | Uncertainty × memory: {MLP, Reset-Z, Carry-Z, carried LSTM} × {C1, C3, C4} (gates G2, G3) |
| Supplementary | Latent per-episode delay C6 (gate G2s), rule fixed before data |
| Stages 3–4 | TD3, unstable plant, training discipline: **not run** (gated on G2/G3) |

## Results

| Gate | Outcome |
|---|---|
| **G0** | Pass, memory headroom only: hiding the queue costs the MLP 70.8 (6/6). Revealing the delay *hurt* it (0/6), as A-1 predicted. |
| **G1** | **Architecture, not capacity.** TRM 136k − MLP 135k −50.2 (0/12); TRM 788k − MLP 788k −27.1 (2/12); MLP 788k − 135k −3.8 (5/12). |
| **G2** | Fail. The interaction has the wrong sign at C4 (−40.0, 1/6). **H3 rejected.** |
| **G3** | Fail. Carry-Z never beats Reset-Z in ≥ 5/6 seeds (+4.8, −22.0, +16.7). |
| **G2s** | Fail. The latent delay gives the same answer (Carry-Z − Reset-Z 2/6). |
| **G4** | Weak pass on the literal rule. |

**Other findings:**
- **The Phase 1 headline reproduced** at higher power: TRM 788k − MLP 135k −34.4 (0/12, p = 0.0005).
- **Depth is inert:** H = 1 and H = 6 vs H = 3 give −3.3 (5/12) and −10.9 (4/12). Only 7 applications are ever trained.
- **TRM's critic fits worse at matched capacity** (1.45× and 1.89×, 12/12 each), while the bigger MLP fits *better*.
- **The gap is mostly in training, not the final policy:** deterministic battery −3.6 (2/12, p = 0.039).
- **Control trade-off:**
  - TRM's greedy torque is much smoother (0.4 vs 9 reversals/s, 11/12, p = 0.006).
  - But it is less robust to sensor noise (−75.8 more, 2/12), a heavier plant and actuator dropout.
  - It costs 65× the latency per decision (4.13 vs 0.064 ms p99; 242 Hz ceiling) and 11–13× the training cost.

## What it established

A confound-controlled negative. With capacity matched, the latent carried (with verified stored-state replay), and conditions built to leave something to infer, recursion does not earn its cost. The deficit is architectural and shows up in value fit more than in the final policy. **Use the MLP.**

**What was left open:** TRM's training discipline (full gradient, per-cycle supervision), a second algorithm, an unstable plant, a tuning check, and the complexity question. Phase 4 took all five on.
