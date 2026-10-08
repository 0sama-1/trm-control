# Glossary: Every Label, Code and Abbreviation in the Project

*trm-delay-robust-scaling · study reference · 2026-10-08*
*Covers Phases 0–4. Each entry gives the label, what it means, and where it is used. Gate outcomes are the final recorded ones.*

<!-- toc -->

## 0. Read this first: labels that look alike

Several short codes are reused with different meanings in different phases. These are the ones most likely to confuse:

| You see | It can mean | How to tell |
|---|---|---|
| **C1** vs **C-1** | C1 = a delay *condition* (fixed 1-step delay + queue). C-1 = the *capacity confound* from the Phase 1–2 analysis. | The hyphen. Conditions have none; confounds do. |
| **A-1** | Phase 3 amendment A-1 (theory check on the headroom probe) **or** Phase 4 amendment A-1 (Stage A clip norms). | Check the phase. |
| **A1–A4** | Phase 3 Stage 0.1 headroom *arms* (no hyphen). Also codes A1–A9 in the Phase 3 plan's exploration inventory. | Arms appear in Stage 0 tables; inventory codes only in the plan's §6. |
| **E1–E5** | Phase 1–2 report: E1–E4 = mechanistic *explanations* (§7), E1–E5 = recommended *experiments* (§10). Phase 3 plan: E-1–E-9 = *engineering items*. Phase 4: **Stage E** = the Lipschitz analysis. | Context: explanation, experiment, engineering task or stage. |
| **G1** | Phase 3 *gate* G1 (capacity vs architecture) **or** inventory item G1 in the Phase 3 plan (out-of-scope: embedded deployment). | Gates are in tables of outcomes; inventory items only in plan §6. |
| **"788k"** | The parameter count of TRM / MLP *under DDPG* (actor + one critic). Under TD3 (two critics) the same configs total ~1.18M. | The label names the configuration, not the TD3 total. |
| **H1** vs **H1′** | H1 = Phase 1's complexity hypothesis. H1′ = Phase 4's measurable version of it. | The prime mark. |
| **H** (alone) | `H_cycles`, TRM's number of outer cycles (H = 1, 3, 6). | Appears as "H = 3", "TRM H = 1". |
| **T** | **Stage T** in Phase 4 (G1 under TD3) **or** TRC's number of control steps (T = 50–100). | Stage T is ours; T as a horizon only appears in the TRC comparison. |

---

## 1. The project in one line

| Term | Meaning |
|---|---|
| **Research question** | Does weight-tied recursion (TRM), used as the actor-critic trunk of a model-free RL agent, benefit continuous control under action delay — and if not, why not? |
| **trm-delay-robust-scaling** | The project's name, the epic under `epics/`, and the W&B project. |
| **Trunk** | The shared body of a network that turns an observation into features. The actor and the critic each have one; a "head" (one linear layer) turns features into an action or a Q-value. The project swaps only the trunk. |
| **TRM vs MLP** | The central comparison: a weight-tied recursive trunk against a plain feed-forward one. |

## 2. Hypotheses and preconditions

| Label | Statement | Introduced | Outcome |
|---|---|---|---|
| **H1** | TRM's advantage over the MLP (and Transformer) is non-decreasing in task complexity; measurable once the horizon is ~5× Pendulum's with sparse reward. | Phase 0 ideation, tested Phase 1 | Threshold prediction falsified on MountainCar; complexity scaling as a whole **untested** (no harder task ever learned by the baseline). |
| **H1′** | Phase 4's operational H1: TRM's deficit, normalised by the random-to-best span, is smaller on a harder task than on Pendulum (≥ 5/6 seed pairs). | Phase 4 plan §7 | **Not run** (K-3). |
| **H_precondition** / **H_transfer** | Phase 1–2 report's two readings of the negative: TRM needs preconditions (P1–P3) that the port lacks / recursion transfers and the negative reflects something else. | Phase 1–2 report | Superseded by H3, H_cap and H_null in Phase 3. |
| **P1** | Precondition: *refinement continuity*. Per-step supervision plus a persistent state to refine. The port breaks both halves. | Phase 1–2 report | The best-supported account; Phase 3 G3 tested the "persistent state" half (failed), Phase 4 GA tested the "supervision" half (failed). |
| **P2** | Precondition: a structured, compositionally correctable answer (a grid can be partly right; a 1-D torque cannot). | Phase 1–2 report | Not manipulable here. |
| **P3** | Precondition: per-instance adaptive capacity (TRM's per-puzzle embedding). No analogue in replay-based RL. | Phase 1–2 report | Not testable here. |
| **H3** | Residual-uncertainty hypothesis: recursion earns its cost only where (i) the input leaves uncertainty iteration could resolve *and* (ii) the trunk carries information across steps. | Phase 3 plan §2 | **Rejected** (G2, G2s fail; G3 fails). |
| **H3.1** | The deficit is a capacity artefact. | Phase 3 | Rejected (G1: architectural). |
| **H3.2** | The load-bearing interaction: Carry-Z − MLP ≤ 0 with nothing to infer, > 0 where there is. | Phase 3 | Rejected (G2). |
| **H3.3** | Depth pays only when there is something to infer. | Phase 3 | Not supported: depth inert everywhere. |
| **H3.4** | Recursion trades return for command regularity (smoother at matched return). | Phase 3 | Supported at n = 12 (0.4 vs 9 reversals/s). |
| **H3.5** | Recursion degrades more gracefully off-nominal. | Phase 3 | Contradicted (TRM less robust). |
| **H3.6** | Recursion is real-time feasible at robot rates. | Phase 3 | Quantified: 4.1 ms/decision, 100 Hz yes, 500 Hz no. |
| **H_cap** | The deficits track parameter count; recursion is irrelevant. | Phase 3 | Rejected (G1). |
| **H_null** | Recursive refinement does not transfer to model-free continuous control under any conditioning. | Phase 3 | What remains, restricted to the setting tested. |
| *(Phase 4, as hypotheses)* | TRM was just untuned (Stage C); restoring the training discipline recovers a benefit (Stage A); the G1 attribution holds under TD3 (Stage T). | Phase 4 | Untuned: no (GC insensitive). Discipline: no (GA fail). TD3: direction holds, but via an optimisation failure (GT pass with caveat). |

## 3. Phases and stages

### Phases

| Phase | Dates | What it did |
|---|---|---|
| **Phase 0** | to 2026-09-08 | Groundwork: pilot, delay wrapper, augmentation fix, TRM port, scoping, literature review, hypothesis ideation. |
| **Phase 1** | 2026-09-08/09 | Horizon and sparsity: Pendulum-v1 vs MountainCarContinuous-v0, 4 trunks, 48 runs (submitted 3 times: v1, v2, v3; v3 canonical). |
| **Phase 2** | 2026-09-09/10 | Dimensionality and coupling: BipedalWalker-v3, 6 trunks, 3 conditions, 54 runs; then the post-hoc re-analysis. |
| **Phase 3** | 2026-09-14 to 09-23 | Confound removal: capacity, carried latent, inferable delay; control-quality battery; 231 jobs. |
| **Phase 4** | 2026-10-06 to 10-08 | Complexity rung, training discipline, LR sweep, G1 under TD3, Lipschitz analysis, C export. |

### Phase 3 stages

| Stage | Question |
|---|---|
| **S0.1** (Stage 0.1) | Headroom probe: is there anything for inference to recover? Arms A1–A4. |
| **S0.2** (Stage 0.2) | Checkpoint battery over the 102 banked Phase 1–2 checkpoints, plus the latency benchmark. |
| **Stage 1** | Capacity × depth: {MLP, TRM} × {135k, 788k}, plus H = 1 and H = 6. |
| **Stage 2** | Uncertainty × memory: {MLP, Reset-Z, Carry-Z, carried LSTM} × {C1, C3, C4}. |
| **Supplementary (s2x)** | Latent per-episode delay C6 (the G2s test). |
| **Stage 3** | TD3 cross-check and an unstable plant. **Not run** (gated on G2). |
| **Stage 4** | Restore TRM's training discipline. **Not run** in Phase 3 (gated on G3); became Phase 4 Stage A. |
| **Stage 5** | Synthesis. |

### Phase 4 stages

| Stage | Question | Runs |
|---|---|---|
| **B0** | Complexity pilot: does the MLP learn InvertedDoublePendulum-v5 at 150k / 300k / 500k steps? | 9 + 2 timing |
| **B0h** | Kill-switch fallback: the same pilot on Hopper-v5. | 9 |
| **B0d** | IDP diagnostic: C1 at 1M steps; no delay at 500k. | 6 |
| **B1** | Complexity rung main matrix (4 trunks × 6 seeds on the harder task). | **Not run** (K-3) |
| **C** | LR sweep: TRM 788k, actor × critic learning-rate multipliers {0.3, 1, 3}. | 27 |
| **A** (+ **a0** pilot) | Training discipline: truncated vs full-gradient vs full-gradient + deep supervision, at C1 and C4 with previous action. | 4 + 36 |
| **T** | Phase 3 Stage 1 contrast under TD3. | 48 |
| **E** (run as **e2**) | Exploratory: actor Lipschitz constant over the 72 Phase 3 Stage 1 checkpoints. | 6 shards |
| **D** | Export actors to float32 C headers for an STM32. | local |
| **smoke** | One GPU job exercising every new code path before any data. | 1 |

## 4. Gates (pre-registered decision rules)

| Gate | Phase | Rule (short) | Outcome |
|---|---|---|---|
| **G0** | 3 | Headroom: oracle (A3 − A2) or memory (A2 − A4) headroom > pooled SD in ≥ 5/6 seeds. | Pass (memory headroom only) |
| **G1** | 3 | Attribution: 2×2 capacity × architecture, "different" = ≥ 10/12 pairs share a sign. | Architecture, not capacity |
| **G2** | 3 | Interaction: [Carry-Z − MLP] improves from C1 to C3 or C4 in ≥ 5/6 seeds. | Fail (H3 rejected) |
| **G2s** | 3 | Latent-delay version of G2 (C6), supplementary. | Fail (rejection stands) |
| **G3** | 3 | Continuity: Carry-Z beats Reset-Z in ≥ 5/6 seeds somewhere. | Fail |
| **G4** | 3 | Practicality: a non-MLP trunk non-dominated on (return, command rate) and admitting 100 Hz. | Weak pass (literal rule) |
| **GB0** | 4 | MLP avg-100 ≥ 8190 on IDP in ≥ 2/3 seeds at some budget. | Fail |
| **KB0** | 4 | Kill switch: same on Hopper (≥ 3420). | Fail (Hopper dropped) |
| **GC** | 4 | Any non-default LR cell beats ×1/×1 by > pooled SD in 3/3 seeds. | Insensitive (no cell flagged) |
| **GA** | 4 | A discipline arm beats truncated TRM in ≥ 5/6 seeds at C1 or C4. | Fail |
| **GT** | 4 | TRM 788k − MLP 788k negative in ≥ 10/12 pairs under TD3. | Pass, with caveat: TRM 788k does not train under `td3.yaml` |
| **H1p** (H1′) | 4 | Normalised deficit smaller on the new rung in ≥ 5/6 seed pairs. | Not run |
| **E** | 4 | — (exploratory, not a gate) | Lipschitz does not explain TRM's noise sensitivity |

## 5. Delay conditions and observation design

| Label | Meaning |
|---|---|
| **d** | Action delay in steps: the action chosen now reaches the plant d steps later. |
| **Queue / augmentation** (`augment_obs`) | Appending the pending (committed, not-yet-applied) actions to the observation. Restores a sufficient statistic for fixed delays (Katsikopoulos & Engelbrecht, 2003). The Phase 0 fix. |
| **Previous action / "pa"** (`prev_action_obs`) | Also appending the last executed action. Phase 3 Stage 2 (amendment A-6) and Phase 4 Stage A only. |
| **Oracle** (`expose_realised`) | Revealing the realised delay to the agent (an information upper bound). |
| **U{0..4}** | Delay drawn uniformly from {0, 1, 2, 3, 4} (exact discrete uniform, `uniform_int`, amendment A-4 of Phase 3). |
| **C0** | No delay. |
| **C1** | Fixed d = 1 + queue. The standard condition. (Phase 3 Stage 1 called it `fixed1-aug`; Phase 4 B/C/T use it without previous action.) |
| **C2** | Fixed d = 1, no queue (Phase 2's augmentation ablation). |
| **C3** | d ~ U{0..4} redrawn every step, queue shown. Turned out to leave nothing to infer (A-1). |
| **C4** | d ~ U{0..4} redrawn every step, queue hidden. Memory has something to recover (G0 memory headroom). |
| **C5** | Time-varying (non-stationary) delay. Optional in the Phase 3 plan; not run. |
| **C6** | d ~ U{0..4} drawn once per episode and held, queue shown (the G2s condition). **C6o**: the same with the oracle. |
| **C1pa / C4pa** | Phase 4 Stage A labels for C1 / C4 with the previous action. |
| **Headroom arms A1–A4** (Phase 3 S0.1) | A1 = fixed d = 1 + queue; A2 = stochastic + queue; A3 = A2 + oracle; A4 = stochastic, no queue. |
| **Sufficient statistic / Markov** | An observation that contains everything needed to predict the future. Delay breaks it; the queue restores it. |

## 6. Confounds

| Label | Meaning | Removed by |
|---|---|---|
| **C-1, capacity** | Every recursive trunk had ~788k parameters vs the MLP's 135k. | Phase 3 Stage 1 (G1) |
| **C-2, latent reset** | The port re-initialised its latent every call (HRM-Agent's weaker "Reset-Z" variant). | Phase 3 Stage 2 Carry-Z (G3) |
| **C-3, nothing to infer** | Under fixed delay + queue the observation is already sufficient. | Phase 3 G0 headroom, Stage 2 C4, C6 (G2, G2s) |

## 7. Trunks (arms)

Parameter counts are actor + critic under DDPG on Pendulum with the queue, unless stated.

| Config | What it is | Params |
|---|---|---:|
| **`dnn`** (MLP 135k) | MLP, hidden [256, 256], ReLU | 134,914 |
| **`dnn_large`** (MLP 788k) | MLP, hidden [624, 624] | 788,114 |
| **`trm`** (TRM 788k) | TRM, hidden 128, H = 3, L = 6, 2-layer SwiGLU cell | 787,842 |
| **`trm_small`** (TRM 136k) | TRM, hidden 44 | 135,654 |
| **`trm_h1`** | `trm` with H = 1 (7 cell applications instead of 21) | 787,842 |
| **`trm_h6`** | `trm` with H = 6 (42 applications) | 787,842 |
| **`trm_hier6`** | Phase 2 variant: six parallel low-level latents | ~794k |
| **`trm_singlez`** | Phase 2 variant: a single latent, no z_H / z_L split | — |
| **`trm_carryz`** (Carry-Z) | `trm` that carries its latent across steps (stored-state replay) | = `trm` |
| **Reset-Z** | The ordinary `trm`: latent re-initialised every call | = `trm` |
| **`trm_fullgrad`** | `trm` + `full_grad`: gradient through all 21 applications | = `trm` |
| **`trm_fullgrad_ds`** | `trm_fullgrad` + `deep_sup`: critic supervised on every outer cycle | = `trm` |
| **`transformer`** | Self-attention trunk (Phases 1–2) | ~797k |
| **`lstm`** | LSTM, stateless across calls (Phases 1–2) | ~167k |
| **`lstm_carry`** | LSTM with its hidden state carried (Phase 3, untuned) | — |
| **TRM-trunc** | Phase 4 name for truncated `trm` with clipping (Stage A baseline) | = `trm` |

Under TD3 (two critics) the "788k" configs total about 1.18M and the "135k" configs about 202k.

## 8. TRM internals

| Term | Meaning |
|---|---|
| **TRM** | Tiny Recursive Model (Jolicoeur-Martineau, 2025): a small shared cell applied many times instead of adding parameters. |
| **Cell / reasoning module** | The shared network applied every step: 2 SwiGLU blocks with RMSNorm residuals. Attention was dropped in the port (a flat observation is a length-1 sequence). |
| **z_H, z_L** | The high-level ("answer") and low-level ("reasoning") latents. |
| **H_cycles (H)** | Outer cycles. Each updates z_H once. |
| **L_cycles (L)** | Inner cycles per outer cycle. Each updates z_L. |
| **21 applications** | H × (L + 1) = 3 × 7 cell applications per forward pass at H = 3. |
| **One-step gradient truncation** | All outer cycles but the last run under `no_grad`; **only 7 of 21 applications receive gradient**. |
| **`full_grad`** | Phase 4 flag removing the truncation (all 21 receive gradient). |
| **`deep_sup`** / deep supervision | Phase 4 flag: the critic's TD loss is applied to every outer cycle's readout (Lee et al., 2015). TRM's original deep supervision re-feeds the carry over many steps against a fixed answer. |
| **ACT / halting** | Adaptive computation time: TRM learns when to stop iterating. Dropped in the port (no analogue in off-policy RL). |
| **H_init, L_init** | The fixed initial latents. |
| **Carry** | The latent passed between decision steps in Carry-Z. |
| **SwiGLU / RMSNorm** | The gated MLP block and the normalisation TRM uses. |
| **embed_scale** | The input projection is multiplied by √hidden_size (needed for the C export). |

## 9. Reinforcement-learning terms

| Term | Meaning |
|---|---|
| **Model-free RL** | Learning a policy and value function from sampled transitions, never using the plant's equations. |
| **Actor / critic** | The policy network (observation → action) and the value network (observation, action → Q). |
| **DDPG** | Deep Deterministic Policy Gradient (Lillicrap et al., 2016). Phases 1–3 and Phase 4 Stages A and C. Actor LR 1e-4, critic LR 1e-3, batch 128, warmup 1000. |
| **TD3** | Twin Delayed DDPG (Fujimoto et al., 2018): twin critics, delayed actor updates, target smoothing. Phase 4 Stages B and T. `td3.yaml`: LR 1e-3 / 1e-3, batch 256, warmup 10,000. |
| **SAC, PPO** | Other algorithms in the codebase; never used for results. |
| **OU noise** | Ornstein–Uhlenbeck exploration noise used by DDPG. |
| **Replay buffer** | Stored transitions sampled for off-policy updates. |
| **Warmup** | Initial steps with random actions. |
| **TD target, critic loss** | The bootstrapped value target and the critic's squared error against it. |
| **Value fit** | How well the critic fits; measured as late critic loss (median over the last quarter of episodes). |
| **Stored-state replay (R2D2-style)** | Saving the carried latent with each transition so updates start from the state the policy acted from (Kapturowski et al., 2019). Verified exact for Carry-Z. |
| **Burn-in** | Alternative to stored state (recompute the carry over a prefix). Not used. |
| **Gradient clipping** | Capping the gradient norm (Phase 4 Stage A: actor 15, critic 2800, from the a0 pilot's p99). |
| **Critic divergence** | Critic loss rising instead of falling over training (TRM 788k under TD3 in Stage T). |

## 10. Environments

| Environment | Obs / act dims | "Solved" reference | Status in this project |
|---|---|---|---|
| **Pendulum-v1** | 3 / 1 | ≈ −150 | The only task every trunk learned. All decisive tests ran here. |
| **MountainCarContinuous-v0** | 2 / 1 | +90 | Floor effect (Phase 1): exploration-limited under DDPG. |
| **BipedalWalker-v3** | 24 / 4 | +300 | Floor effect (Phase 2): no run learned to walk. |
| **InvertedDoublePendulum-v5 (IDP)** | 9 / 1 | 9100 (bar 8190) | Phase 4: MLP not reliable with delay (GB0 fail); learns without delay but doesn't hold it. |
| **Hopper-v5** | 11 / 3 | 3800 (bar 3420) | Phase 4 fallback: fail; dropped. |
| **Floor effect** | — | — | Nothing learns, so every trunk scores the same and comparisons mean nothing. |
| **Complexity rung** | — | — | A harder task, used to test H1. |

## 11. Metrics and statistics

| Term | Meaning |
|---|---|
| **avg-100 / final return** | Mean training return over the last 100 episodes; the primary metric throughout. Includes exploration noise. |
| **eval_return / eval last-5** | A noise-free evaluation episode every 10 training episodes; the mean of the last 5. Secondary. |
| **Battery** | Phase 3's evaluation harness (`w0_eval.py`): 20 deterministic episodes on unseen seeds plus grids for delay mismatch, sensor noise, plant mass/length, actuator dropout; control metrics. |
| **IQM** | Interquartile mean (Agarwal et al., 2021). Used for battery returns. |
| **Stratified bootstrap CI** | Confidence interval resampling runs then episodes. |
| **Sign test** | Counts how many paired differences are positive; no distributional assumption. |
| **k/n** | k of n paired differences positive (e.g. 0/12). |
| **One-sided / two-sided p** | 5/6 → one-sided 0.109; 6/6 → 0.016; 12/12 → two-sided 0.0005; 10/12 → two-sided 0.039. |
| **Pair** | Two runs matched by seed (and condition, when pooled to n = 12). |
| **Pooled SD** | √((var_a + var_b)/2) across seeds; the magnitude threshold in G0, GC, G2s. |
| **Span, untrained_E, best_E** | best − untrained return on an environment; normalises deficits for H1′. Untrained = uniform-random policy (R-2: Pendulum −1164, IDP 49.9, Hopper 16.3). Phases 1–3 quoted ≈ −1460 for an untrained Pendulum *actor*. |
| **δ_E(s)** | Normalised deficit: (MLP − TRM) / span on environment E for seed s. |
| **"Learned"** | B0's bar: avg-100 ≥ 90% of the reference return. |
| **"Clear of floor"** | Draft K-3's exploratory bar (≥ 2275 on IDP). Superseded. |
| **Reversals/s** | Torque direction changes per second in deterministic rollouts; smoothness. |
| **abs(Δu), effort, saturation, hf_frac, work** | Command rate, control energy, fraction at the limit, high-frequency power share, mechanical work. |
| **Noise drop** | Nominal IQM − IQM under sensor noise σ = 0.05 of the observation half-range. |
| **Lipschitz constant** | Spectral norm of ∂a/∂s, how sharply the actor reacts to its input (Stage E). |
| **Latency p50 / p99** | Median / 99th-percentile time per decision (batch 1, 1 CPU thread). |
| **Matched episodes / wall-clock / latency** | The three ways Phase 3 reported cost. |
| **Clip fraction** | Share of updates where gradient clipping fired. |

## 12. Amendments

### Phase 3 (recorded 2026-09-21, before data)

| Label | Change |
|---|---|
| **A-1** | Theory check: with per-step redrawn delay and the queue, the observation is already Markov, so the oracle arm must be null. |
| **A-2** | Adds arm A4 and lets G0 pass on memory headroom (A2 − A4). |
| **A-3** | C4 becomes co-primary with C3 for G2. |
| **A-4** | Exact discrete-uniform delay sampling. |
| **A-5** | Every job on a GPU node; battery as 12 shard jobs. |
| **A-6** | Every Stage 2 trunk also sees its previous action. |

### Phase 4 (each dated, before the data it affects)

| Label | Date | Change |
|---|---|---|
| **R-1** | 10-06 | Stage A reuses Phase 3's MLP runs (bit-identity check passed). |
| **R-2** | 10-06 | Random-policy baselines frozen for the H1′ span. |
| **V-1** | 10-07 | Code verification passed (30/30 bit-identical; gradient-hook test). |
| **A-1** | 10-07 | Stage A clip norms: actor 15, critic 2800. |
| **K-1** | 10-07 | Hopper kill switch fired. |
| **K-2** | 10-07 | Hopper dropped; IDP diagnostic added. |
| **T-1** | 10-07 | Stage T added (both conditions, `td3.yaml` as-is). |
| **K-3** | 10-08 | B1 not run; H1′ not run; exploratory draft superseded. |

## 13. Other code lists you may meet

| Codes | Where | Meaning |
|---|---|---|
| **E1–E4** | Phase 1–2 report §7 | Mechanistic explanations: E1 optimisation/conditioning deficit (supported), E2 slower learning, E3 lower asymptote, E4 instability. |
| **E1–E5** | Phase 1–2 report §10 | Recommended experiments: E1 capacity-matched MLP, E2 Carry-Z, E3 depth sweep, E4 longer budgets, E5 restore per-cycle supervision. They became Phase 3 Stage 1/2 and Phase 4 Stage A. |
| **E-1–E-9** | Phase 3 plan §7 | Engineering items (evaluation harness, Carry-Z, perturbation wrappers, oracle arm, configs, latency benchmark, MuJoCo, deep-supervision variants, observation-delay wrappers). |
| **A1–A9, B1–B10, C1–C8, D1–D5, E1–E4, F1–F5, G1–G4** | Phase 3 plan §6 | The exploration inventory (mechanism, conditioning, control practicality, tasks, learner, methodology, out of scope). Not used elsewhere. |

## 14. Infrastructure

| Term | Meaning |
|---|---|
| **Katana** | UNSW's HPC cluster (PBS Pro scheduler), where every Phase 1–4 job ran. |
| **PBS, qsub, qstat, qdel** | The scheduler and its submit / status / delete commands. |
| **mechgpu2 / mechgpu12** | GPU queues; ≤ 2 h jobs reach every GPU node, longer ones only group-owned nodes. |
| **Walltime** | A job's time limit (1:59 for most Phase 3–4 jobs). |
| **Frozen snapshot** (`p3code/<stage>`, `p4code/<stage>`) | A copy of the code per stage; `sync.sh` refuses to overwrite one jobs were submitted from. |
| **Ledger** | `~/jobs/phase{3,4}/ledger.jsonl`: every submitted job. |
| **`p3.py` / `p4.py`** | The job managers (manifest, submit, status, resubmit). |
| **`gates.json` / `gates_eval.py`** | The frozen rules and the scripts that evaluate them. |
| **Run board** | The live status artifact (Phase 3 and Phase 4 each have one). |
| **model_id** | A unique ID per run; the checkpoint is `models/<model_id>.pt`. |
| **Exit −19** | A PBS code reported after two Phase 4 jobs had finished cleanly; counted done. |
| **k207 / RTX PRO 6000** | A Blackwell GPU node with no kernels in the venv's PyTorch build; excluded with `gpu_rp6000=False`. |
| **Bit-identical** | Same weights and histories to the last bit; the standard for "unchanged code path". |

## 15. Literature you will see cited

| Short name | Work | Why it matters here |
|---|---|---|
| **TRM** | Jolicoeur-Martineau (2025), *Less is More: Recursive Reasoning with Tiny Networks* | The architecture under test. |
| **HRM** | Hierarchical Reasoning Model, TRM's predecessor | TRM simplifies its two networks into one cell. |
| **HRM-Agent** | Dang & Rawlinson (2025) | Closest precedent: RL-trained recursive model on discrete mazes; source of Carry-Z vs Reset-Z. |
| **TRC** | Jain & Linares (2025), *Tiny Recursive Control* | Model-based TRM for control; each pass gets dynamics-error feedback. The contrast that explains why the port's depth goes unused. |
| **TRM-PL** | Wijayatunga & Armellin (2026) | TRM for the J2-perturbed Lambert problem; also model-based. |
| **ARC Prize (2025)** | *The Hidden Drivers of HRM's Performance* | Training-time refinement, not inference depth, drives HRM. |
| **Roye-Azar et al. (2026)** | TRM on ARC-AGI-1 | Most accuracy arrives at the first recursion step; removing the puzzle ID collapses accuracy. |
| **Katsikopoulos & Engelbrecht (2003)** | Delayed MDPs | The queue-augmentation theorem. |
| **Derman et al. (2021)** | Non-stationary delays | Why time-varying delay (C5) is the hardest case. |
| **Ni et al. (2022)** | Recurrent model-free RL for POMDPs | Previous-action input (A-6); tuned recurrent baselines. |
| **R2D2** | Kapturowski et al. (2019) | Stored-state replay for carried latents. |
| **DSN** | Lee et al. (2015), Deeply-Supervised Nets | The deep-supervision surrogate in Stage A. |
| **DEQ** | Bai et al. (2019), Deep Equilibrium Models | Alternative to truncation; not run. |
| **Agarwal et al. (2021)** | Statistical precipice | IQM and stratified bootstrap. |
| **Matheron et al. (2019)** | The problem with DDPG | Why MountainCar was a floor. |
| **Fujimoto et al. (2018)** | TD3 | The second algorithm; IDP reference curves. |
| **Lillicrap et al. (2016)** | DDPG | The main algorithm. |
