# Stage D: Embedded Export Support

*trm-delay-robust-scaling · Phase 4 stage note · 2026-10-08*
*Plan §9. Engineering only, no gate. Outputs in `../results/export_c/`; script `../../experiments/phase4/export_c.py`.*

## 1. What was asked

A script to export the actor weights of three Pendulum actors to float32 C headers for an on-device test on the author's STM32: MLP 135k, TRM 136k (H = 3) and TRM H = 1. For each, write 20 reference input/output vector pairs from PyTorch, and report the parameter count and the FP32 and INT8 weight sizes against a 512 KiB flash budget. The STM32 side is the author's.

## 2. What was done

`export_c.py` rebuilds each actor from its results file (Phase 3 Stage 1, fixed d = 1 + queue, seed 1) and writes one header per actor containing:

- every parameter tensor as a `static const float` array in PyTorch's row-major layout;
- the trunk hyperparameters as `#define`s, and for TRM the fixed recursion buffers (the initial latents `H_init`, `L_init`), which the forward pass needs but which are not trainable parameters;
- 20 reference pairs: inputs are observations visited by the deterministic policy (4-dimensional: three plant observations and the one-slot action queue), outputs are the PyTorch float32 actions.

INT8 size assumes per-tensor symmetric quantisation: one byte per weight plus one float32 scale per tensor, with buffers kept in float32. No quantised model was built.

| Actor (seed 1) | Actor parameters | Buffer floats | FP32 size | INT8 size | Fits 512 KiB |
|---|---:|---:|---:|---:|---|
| MLP 135k (`dnn`) | 67,329 | 0 | 263.0 KiB (269,316 B) | 65.8 KiB (67,353 B) | FP32 and INT8 |
| TRM 136k (`trm_small`, H = 3) | 67,805 | 88 | 265.2 KiB (271,572 B) | 66.6 KiB (68,185 B) | FP32 and INT8 |
| TRM H = 1 (`trm_h1`, 788k config) | 393,857 | 256 | 1539.5 KiB (1,576,452 B) | 385.7 KiB (394,909 B) | INT8 only |

*Source: `phase4/results/export_c/export_summary.json`. Parameter counts here are the actor alone; the "135k"/"788k" labels are actor + critic totals under DDPG. TRM 788k (H = 3) has the same parameters as TRM H = 1 and would have the same size; it was not exported.*

## 3. Reading the table

- **At matched size, the two trunks are the same flash cost.** MLP 135k and TRM 136k actors differ by 476 parameters and 88 buffer floats; both fit the budget in FP32 with room to spare.
- **The 788k-configuration TRM fits only after quantisation.** In FP32 it needs about three times the budget; INT8 brings it to 75% of it.
- **Flash is not where TRM's embedded cost lies.** The TRM 136k actor still performs 21 cell applications per decision. Phase 3 measured it at 2.69 ms per decision (p99, one thread of a Katana Xeon) against the MLP's 0.064 ms, and TRM H = 1 at 1.42 ms. On a microcontroller both would be far slower; the ratio, not the size, is the deployment question.

## 4. Not done

- **The on-device equivalence test** (the author's STM32 side) has not been run, so it is not known whether the headers reproduce PyTorch's outputs on the target to within float32 tolerance.
- **Quantised accuracy.** The INT8 figures are sizes only. No INT8 actor was evaluated, so the effect of quantisation on return is unknown.
- **On-device latency** was not measured.
- Only seed 1 of each actor was exported.
