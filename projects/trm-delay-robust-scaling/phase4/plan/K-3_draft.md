> **Superseded 2026-10-08.** The author chose not to run B1. See amendment K-3 in `gates.json` and plan §10.

# Amendment K-3 (DRAFT, awaiting author approval; not in effect)

*Drafted 2026-10-07 after the IDP diagnostic (K-2). No B1 data exists.*

## Trigger

Per K-2, B1 is confirmatory at 1M only if the 1M diagnostic passes GB0's rule (≥ 2/3 MLP seeds with final avg-100 ≥ 8190). **It does not:** 0/3 (4147, 2590, 1733). So B1 runs **exploratory**.

## B1 (exploratory)

- **Environment.** InvertedDoublePendulum-v5, TD3 (`td3.yaml` unchanged), C1 (fixed d = 1 + queue, no previous-action input).
- **Trunks.** `dnn` (MLP 135k), `dnn_large` (MLP 788k), `trm` (TRM 788k, H = 3), `trm_h1` (TRM H = 1). Seeds 1–6, so **24 jobs**.
- **Budget.** **300k** env steps (author to confirm). Reasons:
  - 300k gives the highest median MLP final return of the four budgets tried (300k: 3861; 500k: 3781; 150k: 2291; 1M: 2590).
  - 300k and 500k are the only budgets where every seed clears the floor below.
  - 300k costs ~40% less TRM compute than 500k.
  - The 500k alternative has the higher worst seed (3592 vs 3078).
- **"Clear of floor" replaces "learned".** A run clears the floor if its final avg-100 is ≥ max(25% × 9100, 10 × 49.9) = **2275**. B1's H1′ test is computed only if **all 6 B1 MLP 135k seeds** clear the floor. Otherwise B1 is reported descriptively, with no H1′ statement.
- **H1′, labelled exploratory.** The statistic is as pre-registered in §7: δ_E(s) = [final_MLP135k(s) − final_TRM788k(s)] / span_E, "supported" if δ_IDP(s) < δ_Pendulum(s) in ≥ 5/6 seed pairs. Changes:
  - **Pendulum reference.** Stage T's TD3 C1 runs (`dnn` and `trm`, seeds 1–6) are the **primary** reference, so both rungs use the same algorithm and TD3 settings. Phase 3 Stage 1's DDPG runs are reported as secondary. This removes threat 1 of §7. If Stage T fails to produce all 12 runs, the DDPG reference is used and labelled as such.
  - **Spans.** span_E = best_E − untrained_E, with untrained_E frozen in R-2. best_E is the highest final avg-100 in each rung's comparison set (Stage T C1 runs on Pendulum; all B1 runs on IDP).

## Walltime (from B0 timing on an L40S: TRM 35 ms/step, TRM H = 1 20 ms/step, MLP 4.2 ms/step; V100 ≈ 1.8×)

| Arm | 300k on L40S | 300k on V100 | Requested walltime | Queue tier |
|---|---:|---:|---:|---|
| MLP 135k / 788k | ~25 min | ~45 min | 01:59 | mechgpu2 (all GPU nodes) |
| TRM H = 1 | ~1.7 h | ~3.0 h | 03:59 | mechgpu12 (group nodes) |
| TRM 788k | ~2.9 h | ~5.2 h | 05:59 | mechgpu12 (group nodes) |

- **GPU time:** about 40–55 GPU-h in total.
- **Queue risk:** walltimes over 2 h route to group-owned nodes only. In Phase 3 that tier ran 2 jobs at a time, so the 12 TRM jobs could take 1–2 days of queueing. The alternative is shortening the TRM arms, which this draft does not propose.
- **At 500k:** TRM 788k needs ~4.8 h on an L40S and ~8.7 h on a V100, which is over the 6 h tier on a V100.

## Threats (to be stated in the B1 note)

1. **Exploratory, post-hoc criterion.** The floor and the budget were chosen after seeing B0 and diagnostic data. H1′ is hypothesis-generating, not confirmatory.
2. **The baseline is unreliable on this rung.** The MLP's C1 runs are bimodal (20–48% of episodes fully solved) and oscillate (collapse/recovery cycles within single seeds), so a final avg-100 snapshot carries large seed variance. With n = 6 the normalised-gap test has low power.
3. **Delay, not just complexity, makes IDP hard.** Without delay, the same MLP reaches avg-100 ≥ 8190 in all 3 seeds at some point (by 106k–249k steps). With d = 1 + queue it never does, even at 1M. A TRM − MLP gap on IDP-C1 therefore mixes task complexity with sensitivity to delay on an unstable plant.
4. **Cross-environment seed pairing is arbitrary** (as in §7).
5. **The parameter basis differs under TD3.** "788k" names the DDPG-basis config; TD3 totals include twin critics (~1.18M for TRM 788k).
6. **The bar is far from reached.** No C1 configuration reached the original 8190 bar, so "clear of floor" means "learning visibly", not "solved".

## Not changed

GB0, KB0 and their outcomes (both fail) stand. H1′'s confirmatory version is recorded as not run.
