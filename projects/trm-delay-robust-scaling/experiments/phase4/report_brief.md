# Brief: write the Phase 4 report, the whole-project report, and the summary

You are writing the final documents for a finished research project in the repository at
`/home/osama/trm-control`. Project folder: `projects/trm-delay-robust-scaling/` (below: PROJ).
Start by reading `PROJ/README.md`, which maps every folder.

**The project.** Does weight-tied recursion (Tiny Recursive Models, TRM) earn its cost as the
actor/critic trunk of a model-free RL agent controlling a plant through delayed actions?
Phases 0–3 are written up already. Phase 4 has just finished on the UNSW Katana cluster.

## Read first (in this order)

1. `PROJ/phase3/report/3_final_report.md`: the detailed report of Phases 0–3. **This is the style reference for everything you write**: Markdown source, numbered sections, tables carrying the numbers, "k/n" sign counts with two-sided p alongside, plain direct sentences, no thesis prose, no hype.
2. `PROJ/phase3/report/1_summary_phases_0-3.md`: the style reference for the summary.
3. `PROJ/phase4/plan/plan.md`: Phase 4's pre-registration. Each stage's question, arms and decision rule are fixed here. §10 lists amendments.
4. `PROJ/experiments/phase4/gates.json`: the machine-readable rules, frozen values, every gate's final status and detail, and all dated amendments. **Gate outcomes come from here. Do not re-judge them.**
5. The analysis outputs `PROJ/experiments/analysis_p4_*.json` (written by `PROJ/experiments/phase4/gates_eval.py`). Raw per-run results are in `PROJ/phase4/results/` (one JSON per run: `final_avg100_return`, `episode_history`, `wall_clock_seconds`, `device_name`, the full `config`).
6. Any Stage E (Lipschitz) or Stage D (C-header export) outputs under `PROJ/phase4/results/`. If either is absent, say in the report that it was not run. Do not run it.

If `gates_eval.py` has an evaluator for a stage whose `analysis_p4_*.json` is missing or stale, you may run it (`python3 gates_eval.py <cmd>` from `PROJ/experiments/phase4`; it only reads local files). Never edit `gates.json` rules, frozen values or statuses. Never submit or modify Katana jobs.

## Deliverables

All Markdown is the source of truth. For each document, also build tex + pdf with the same
pipeline as Phase 3: copy `PROJ/phase3/report/build.py` into the new folder and run it there
(it calls `~/.local/bin/tectonic`). If the build fails after a fair attempt, keep the Markdown and
report the failure. Make the figures as PNG + PDF with matplotlib in a new script
`PROJ/experiments/phase4/make_figures_phase4.py` that reads only the analysis JSONs and the
`PROJ/phase4/results/` files, writing to the folder of the document that uses them (`figures/`).

1. **`PROJ/phase4/report/`**
   - `stage_<X>.md` for each stage that ran: B0 (with its Hopper fallback B0h and the IDP diagnostic B0d in the same note), C, A (with its a0 pilot), T, E and D. B1 did not run, so it gets no note; it is covered under deviations. This is what the plan's deliverables section asks for: a short results note with tables, figures, the gate outcome and threats to validity.
   - `phase4_report.md`: the Phase 4 report proper. Cover why Phase 4, what was pre-registered, the engineering and verification (amendment V-1: bit-identity, gradient-hook test), each stage's results against its rule, deviations from the plan (every amendment, and every stage that did not run and why), threats to validity, and what Phase 4 changes about the Phase 0–3 conclusions.
   - Known facts to get right:
     - GB0 failed on InvertedDoublePendulum-v5. The MLP plateaued at avg-100 ≈ 3500–4300 from ~150k steps on: episodes are bimodal, 32–45% full 1000-step solves and the rest early falls, so the threshold was missed through unreliability, not budget.
     - Under TD3 the "788k" TRM trunk totals ~1.18M parameters (twin critics). Name the counting basis wherever parameter counts appear.
     - Runs are not bit-reproducible across GPU models (the nested-budget check in `analysis_p4_b0.json`).
     - Stage A's MLP arm reuses Phase 3 Stage 2 runs, justified by amendment R-1.
     - The H1′ span uses a uniform-random untrained baseline (amendment R-2), which differs from the "≈ −1460" figure Phases 1–3 quoted.
     - **Stage T** (amendment T-1; plan §9b; `analysis_p4_t.json`) is the Phase 3 Stage 1 contrast under TD3. GT is recorded as **pass** (TRM 788k − MLP 788k negative in 12/12), but `gates.json` records a caveat: TRM 788k does not train under `td3.yaml` (2/12 seeds learn; critic loss rises through training, i.e. diverges), so the pass is an optimisation failure, not a Phase-3-sized deficit. Report both the pass and the caveat; do not state that Phase 3's magnitude replicated. The likely contributor (td3.yaml's actor lr 1e-3, 10× DDPG's; Stage C showed higher actor lr hurts TRM) is untested.
     - **IDP diagnostic** (amendment K-2; `analysis_p4_b0d.json`): C1 at 1M steps 0/3 learned; without delay all 3 seeds reach 8190 at some point but do not hold it. One-step delay on an unstable plant is the main obstacle. The Hopper rung was dropped.
     - **B1 and H1′ were not run** (amendment K-3, author decision 2026-10-08). The exploratory draft `PROJ/phase4/plan/K-3_draft.md` is superseded. H1, the complexity question, remains untested across the whole project (MountainCar, BipedalWalker, InvertedDoublePendulum and Hopper were all unlearnable by the baseline at budget). State this as a limitation, not as evidence against TRM.
     - **GA secondaries** (deltas vs MLP, critic-loss ratios, eval, training cost per GPU model, latency) are in `analysis_p4_a.json`. Latency was measured on a local workstation (about 2× slower than the Phase 3 Katana Xeon): compare ratios only.
     - **Stage E** (`analysis_p4_e.json`, exploratory): TRM actors have *lower* local Lipschitz constants than MLPs yet larger sensor-noise drops; the Lipschitz constant does not explain TRM's noise sensitivity.
     - **Stage D**: the export summary and headers are in `PROJ/phase4/results/export_c/`.
     - Stage A had two jobs with PBS exit −19 after clean completion (counted done); Stage E's first submission failed on a snapshot path bug and was rerun as `e2`. These are execution notes for the deviations section.
   - The author's thesis framing (keep it): the goal is to assess whether TRM's architecture works as a model-free RL trunk, not to redesign it. The Tiny Recursive Control contrast (Jain & Linares 2025: recursion there gets per-pass error feedback from known dynamics and solver labels) belongs in the discussion as context for *why* the port's extra depth goes unused. Do not present it as a recommended pivot.
2. **`PROJ/overall/final_report.md`**: the whole project, Phases 0–4, as one detailed report. Build on `3_final_report.md`: keep its structure and numbers for Phases 0–3 (copy and adapt; do not re-derive them), add Phase 4, and rewrite the executive summary, synthesis table, threats and recommendations so they reflect all five phases. Copy any Phase 0–3 figures it uses into `PROJ/overall/figures/`.
3. **`PROJ/overall/summary.md`**: a short summary of the whole project (2–3 pages) for a supervisor: the question, what each phase did and found in one short paragraph each, the final answer, the practical recommendation, the open questions. Same register as `1_summary_phases_0-3.md`.

## Rules

- Every number comes from a file. Cite which analysis JSON a table comes from in a short note under it. Never estimate, round loosely or invent.
- Report gate outcomes exactly as `gates.json` records them. Where the data suggest something beyond a gate, label it exploratory.
- A negative result is a valid outcome. Do not frame any result as more positive or more negative than its numbers.
- Do not edit anything outside `PROJ/phase4/report/`, `PROJ/overall/` and the new figure script, and do not commit to git.
- When done, reply with: the files written, each gate's outcome in one line, any build failures, and anything you could not verify.
