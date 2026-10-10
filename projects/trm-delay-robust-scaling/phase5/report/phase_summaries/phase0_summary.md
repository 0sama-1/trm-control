# Phase 0 Summary: Groundwork

*trm-delay-robust-scaling · phase summary · up to 2026-09-08*
*Labels are defined in `../glossary.md`.*

## Question

Before comparing trunks, could the comparison be trusted? The delay had to be real, the agent had to be told about it the standard way, and the recursive trunk had to be a faithful port of TRM.

## Why this phase existed

Tiny Recursive Models reach strong puzzle results (45% on ARC-AGI-1 with about 7M parameters) by applying one small weight-tied cell many times instead of adding parameters (Jolicoeur-Martineau, 2025). If iteration can replace parameters, the place it matters most is embedded control, where parameters, memory and latency are scarce. TRM is also made of parts control already uses: a small feed-forward cell applied repeatedly, like a recurrent network. So it can be swapped in as an actor-critic trunk with nothing else changed.

Action delay was chosen as the test bed for a principled reason. Under a d-step delay the observation stops being a sufficient statistic. Appending the queue of pending actions restores sufficiency in theory (Katsikopoulos & Engelbrecht, 2003), but says nothing about whether a small network can fit the resulting value function. Iterative latent refinement, something like a learned state estimator, was a plausible inductive bias for exactly that.

## What was done

1. **The pilot.** On Pendulum-v1 under a one-step delay, the MLP converged while TRM and Transformer trunks did not.
2. **The bug it exposed.** The delay wrapper applied actions late but never told the agent which action was about to land, so every stored transition paired a reward with an action that had not caused it.
3. **The augmentation fix.** Appending the pending-action queue to the observation. It was verified exact for fixed delays (5 delay values × 10 steps, zero mismatches). An audit also fixed a silent bug in the stochastic delay.
4. **"Comparable reward."** With augmentation, all three trunks reached comparable reward, with TRM about 18× slower to train (55 vs 3 minutes per 200 episodes).
5. **The TRM port.** It keeps the shared SwiGLU/RMSNorm cell, the two latents (z_H, z_L), 3 outer × 6 inner cycles, and the one-step gradient truncation, under which **only 7 of 21 cell applications get gradient**. It drops token attention, per-puzzle embeddings and ACT halting, which have no analogue for a flat observation, and it re-initialises the latent every call.
6. **Scoping as a gated research epic.**
   - A literature review positioned the project against the model-based TRM-for-control work (Jain & Linares, 2025; Wijayatunga & Armellin, 2026).
   - Three rounds of hypothesis review produced **H1** (TRM's advantage grows with task complexity), with significance capped at 3/5.
   - Experiment planning replaced a regression design (best possible p = 0.125) with a pooled sign test.

## What it established

- A delay test bed that is verifiably correct.
- A TRM port with two known simplifications. Both were later tested as confounds.
- An anecdote to test: "comparable reward at 18× the cost".
- H1, with a pre-registered statistic.

## Carried forward

The latent reset and the gradient truncation, both properties of the port, became Phase 3's Carry-Z test and Phase 4's training-discipline test.
