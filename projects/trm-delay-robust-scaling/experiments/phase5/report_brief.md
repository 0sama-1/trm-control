# Brief: write the Phase 5 report and update the whole-project report and summary

Repository `/home/osama/trm-control`; project folder `projects/trm-delay-robust-scaling/` (PROJ). Read `PROJ/README.md` first.

**Context.** Phases 0–4 are fully written up in `PROJ/phase4/report/`. That folder holds:
- `full_report.md`: all phases, with research question, hypotheses, per-phase methodology and discussion;
- `phase4_report.md`;
- `stage_*.md`;
- `project_summary.md`;
- `build.py`, `figures/`, `derived_numbers.json`.

Phase 5 has just finished. It is one small pre-registered stage.

## Phase 5 in brief (verify every number against the files)

- **Pre-registration and rules:**
  - plan: `PROJ/phase5/plan/plan.md`;
  - rules: `PROJ/experiments/phase5/gates.json` (gate GT2, amendment P5-1);
  - analysis: `PROJ/experiments/analysis_p5_t2.json`;
  - raw runs: `PROJ/phase5/results/results_p5t2/`;
  - the reused 1e-3 arm: `PROJ/phase4/results/results_p4t/`.
- **Question:** does TD3's 10× higher actor learning rate explain why TRM 788k failed to train in Phase 4's Stage T?
- **Design:** Pendulum-v1, TD3 (`conf/arch/td3.yaml`, only `arch.actor_lr` overridden), conditions C0 (no delay) and C1 (fixed d = 1 + queue, no previous action), seeds 1–6, 200 episodes.
  - TRM 788k at actor LR 1e-4 and 3e-4 (12 runs each).
  - The 1e-3 arm reuses Stage T's 12 runs (P5-1).
  - Control: MLP 788k (`dnn_large`) at actor LR 1e-4.
  - 36 new Katana GPU jobs, frozen snapshot `p5code/t2`, code byte-identical to Phase 4's `p4code/t`.
- **Gate GT2 outcome (from gates.json): "LR explains"**:
  - TRM 788k learned (final avg-100 > −1000) in 12/12 runs at 1e-4, 11/12 at 3e-4, and 2/12 at 1e-3;
  - MLP 788k learned 12/12 at both 1e-4 and 1e-3.
  - Primary, LR-matched: TRM 788k − MLP 788k at 1e-4 is median −13.0, 2/12 pairs positive, two-sided p = 0.039.
  - Secondary, unequal LRs: TRM at 3e-4 − MLP at 1e-4 is −53.6, 0/12.
  - Late critic loss: TRM 36.0 at 1e-4, 47.4 at 3e-4 and 561 at 1e-3, against MLP 25.4.
- **What it means:**
  - Stage T's failure was an actor-LR effect.
  - At a DDPG-like actor LR, TRM trains under TD3 and shows the same direction and similar size of deficit as Phase 3's G1 under DDPG (TRM 788k − MLP 788k −27.1, 2/12): a modest, consistent deficit with a worse value fit.
  - GT's caveat in `PROJ/experiments/phase4/gates.json` (notes) is thereby resolved. Phase 3's architectural attribution now holds under two algorithms at matched, sensible LRs.
  - Also note: TRM is markedly more LR-sensitive than the MLP. The MLP is unaffected by a 10× actor-LR change; TRM goes from 12/12 to 2/12.

## Deliverables (all in `PROJ/phase5/report/`, each .md + .tex + .pdf)

Build each with a copy of `PROJ/phase4/report/build.py` in that folder (it calls `~/.local/bin/tectonic`).

1. **`stage_T2.md`:** a short note in the style of `PROJ/phase4/report/stage_T.md`, with tables, a figure, the gate outcome and threats.
2. **`phase5_report.md`:** the Phase 5 report in the style of `phase4_report.md`: why, pre-registration, method, results against the rule, discussion, threats, and what it changes.
3. **`full_report.md`:** the whole project, Phases 0–5. Start from `PROJ/phase4/report/full_report.md` (copy it) and update it:
   - add Phase 5 (methodology, results, discussion);
   - update the executive summary, the research question and hypotheses section (GT2 as a tested hypothesis, and GT's resolution), the synthesis table, the cross-phase mechanism discussion, threats, recommendations and appendices.

   Keep every Phase 0–4 number as is.
4. **`project_summary.md`:** start from `PROJ/phase4/report/project_summary.md` and add Phase 5. Keep it to about 3 pages.

**Figures:** add a script `PROJ/experiments/phase5/make_figures_phase5.py` (reading only the analysis JSON and the result files) that writes to `PROJ/phase5/report/figures/`, for example learned-run counts and final returns by actor LR, and learning curves by LR. Copy any Phase 0–4 figures the documents use from `PROJ/phase4/report/figures/`.

## Rules

- Every number comes from a file; cite the source JSON under each table.
- Report gate outcomes exactly as `gates.json` records them.
- Label anything beyond the gate as exploratory.
- **Keep the author's framing.** The thesis assesses TRM's architecture as-is; Tiny Recursive Control is context, not a recommended pivot. A negative or modest result is a valid outcome. Plain sentences, no hype.
- Do not edit anything outside `PROJ/phase5/report/` and the new figure script. Do not touch `gates.json` files. No git commits.
- **Do not write** per-phase summaries, a Q&A or a glossary; those are written separately into `PROJ/phase5/report/phase_summaries/`, `qa_all_phases.md` and `glossary.md`.
- When done, reply with: the files written, the gate outcome in one line, any build failures, and anything you could not verify.
