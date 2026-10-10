# Phase 5 Plan: Does the Actor Learning Rate Explain TRM's TD3 Failure?

*trm-delay-robust-scaling · pre-registration · frozen 2026-10-10, before any Phase 5 data*
*Rules are machine-readable in `../../experiments/phase5/gates.json`. Same procedure as Phase 4: frozen code snapshot, ledger, gate evaluated from the rule as written, stop and report at the gate.*

## 1. Why

Phase 4's Stage T reran Phase 3's capacity contrast under TD3 with `conf/arch/td3.yaml` unchanged. GT passed (TRM 788k − MLP 788k negative in 12/12 pairs), but by an optimisation failure:
- TRM 788k stayed at random-policy level in 10/12 runs, and its critic loss rose through training;
- both MLPs learned in every run.

The suspected but untested cause is the actor learning rate. `td3.yaml` uses 1e-3, 10× the DDPG actor rate (1e-4) under which TRM trained normally in Phases 1–4, and Phase 4's Stage C showed that raising TRM's actor rate under DDPG hurts it. Stage T2 tests that explanation directly.

## 2. Stage T2

**Setting.**
- **Environment and algorithm:** Pendulum-v1, TD3, `td3.yaml`. Only `arch.actor_lr` is overridden; critic LR stays 1e-3, and batch 256, warmup 10,000 and policy delay 2 are unchanged.
- **Budget:** 200 episodes.
- **Conditions:** C0 (no delay) and C1 (fixed d = 1 + queue, no previous action), the same two as Stage T.

**Arms** (seeds 1–6):

| Arm | Actor LR | Runs |
|---|---|---:|
| TRM 788k (`trm`) | 1e-4 | 12 |
| TRM 788k | 3e-4 | 12 |
| TRM 788k | 1e-3 (= Stage T's setting) | 12 |
| MLP 788k (`dnn_large`), control | 1e-4 | 12 |

**"Learned"** means final avg-100 > −1000. Stage T used this as a descriptive cut; it is fixed here before any data.

> **Gate GT2.** For each actor LR, report TRM 788k runs learned out of 12.
> - **"LR explains the TD3 failure"** if TRM 788k learns in **≥ 10/12** runs at some actor LR **< 1e-3**.
> - If it does, also report TRM 788k − MLP 788k at that LR: n = 12 (seed, condition) pairs, sign test, two-sided p.
> - If both 1e-4 and 3e-4 pass, both contrasts are reported. The 1e-4 one is LR-matched and primary. The 3e-4 one compares unequal actor LRs and is labelled so.
>
> *Secondary:* late critic-loss medians (last quarter of episodes), eval last-5, and the control MLP's learned count against Stage T's MLP 788k at actor LR 1e-3.

## 3. Code and verification

- No code changes. `src/` and `conf/` are byte-identical to Phase 4's frozen Stage T snapshot (`p4code/t`), checked with an rsync dry run on 2026-10-10.
- Job manager `experiments/phase5/p5.py` is a copy of `p4.py` with Phase 5 paths (`p5code/`, `phase5_logs/`, `~/jobs/phase5/`).

## 4. Threats recorded now

1. **The threshold** (−1000) is coarse. It separates "learned" from "stayed near random" (uniform-random ≈ −1164), not good from mediocre runs.
2. **A pass would show the TD3 failure is LR-sensitive**, not that 1e-4 is TRM's best TD3 setting. The grid has three points.
3. **MLP control at one LR.** The MLP control runs at actor LR 1e-4 only, so a contrast at 3e-4 mixes architecture with LR.
4. **Cross-GPU reproducibility.** Runs are not bit-reproducible across GPU models (Phase 4 B0). Any reuse of Stage T runs pairs runs that may have run on different hardware.

## 5. Amendments

- **P5-1 (2026-10-10, before data).** The TRM 788k arm at actor LR 1e-3 reuses Phase 4 Stage T's 12 runs (same configuration; code byte-identical to `p4code/t`), with the author's approval. Stage T2 submits 36 jobs.
