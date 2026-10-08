# trm-delay-robust-scaling

Does weight-tied recursion (Tiny Recursive Models, TRM) earn its cost as the actor/critic trunk of a model-free RL agent controlling a plant through delayed actions? Run on UNSW Katana, tracked as the epic `epics/trm-delay-robust-scaling/`.

Each phase folder holds that phase's plan, results and report. Code lives in `experiments/` (all phases) and the repo's `src/` and `conf/`.

| Folder | Contents |
|---|---|
| `phase0-literature/` | `literature-review/review.md`; hypothesis ideation records in `ideation/` (concept index, round-1 and final proposals) |
| `phase1/` | Horizon and sparsity: Pendulum-v1 and MountainCarContinuous-v0, 48 runs. `report/` (Phase 1 report + figures), `results/` (canonical v3 runs), `results-archive/v1_partial_40/` (superseded first submission) |
| `phase2/` | Dimensionality and coupling: BipedalWalker-v3, 54 runs. `report/` (Phase 2 report + figures), `results/`, `results-archive/partial/`. `report-phases1-2/` holds the combined Phase 1–2 report (`report.md`), the post-hoc analysis, the 2026-09-17 plan review (`summary.md`), their figures and build scripts, and an earlier draft (`drafts/medium_sonnet/`) |
| `phase3/` | Confound removal: capacity, carried latent, inferable delay; 231 jobs. `plan/` (plan.md with its §10 amendments, the original proposal), `results/`, `report/` (summary of Phases 0–3, Phase 3 paper, final report; `build.py` makes tex/pdf from md), `qa/` (story and supervisor Q&A), `dashboard/` (Phase 3 run board) |
| `phase4/` | Complexity rung, training discipline, LR sweep, G1 under TD3, Lipschitz analysis, C export (2026-10-06 to 10-08). `plan/plan.md` (pre-registration and amendments; rules in `experiments/phase4/gates.json`), `results/` (incl. `export_c/` C headers), `report/`: the final document set — `full_report` (all phases), `phase4_report` + `stage_*` notes, `project_summary`, `phase_summaries/phase0–4_summary`, `qa_all_phases`, `glossary` (each .md + .tex + .pdf) |
| `overall/` | Unused: the whole-project report and summary live in `phase4/report/` |
| `experiments/` | Submission and analysis code for every phase, the `analysis_*.json` outputs, `phase3/` and `phase4/` job tooling (`p3.py`/`p4.py`, `sync.sh`, `poll.sh`, `gates_eval.py`) |

Live Phase 4 run board: https://claude.ai/artifact/AiHU7T9dgTWNWzXnaP8Us9

## Old paths

The folders were reorganised by phase on 2026-10-07. Older documents (finished reports, epic comments, `epics/.../project.json`) cite the old paths as they were when written:

| Old path | New path |
|---|---|
| `docs/literature-review/` | `phase0-literature/literature-review/` |
| `ideas/{concept-index,proposal-final,round1-proposal}.json` | `phase0-literature/ideation/` |
| `ideas/phase3-proposal.md` | `phase3/plan/phase3-proposal.md` |
| `results_v3/` | `phase1/results/` |
| `old-summarised-results/results_v1_partial_40/` | `phase1/results-archive/v1_partial_40/` |
| `papers/phase1/` | `phase1/report/` |
| `results_phase2/`, `results_phase2_partial/` | `phase2/results/`, `phase2/results-archive/partial/` |
| `papers/phase2/` | `phase2/report/` |
| `papers/{report,summary}.*`, `papers/figures/`, `papers/build*`, `papers/medium_sonnet/` | `phase2/report-phases1-2/` |
| `papers/phase3/` | `phase3/plan/` |
| `results_phase3/` | `phase3/results/` |
| `Phase 3/` | `phase3/report/` |
| `Q&A/` | `phase3/qa/` |
| `dashboard/` | `phase3/dashboard/` |
| `papers/phase4/` | `phase4/plan/` |
| `results_phase4/` | `phase4/results/` |

Paths on Katana (`/srv/scratch/z5308482/trm-control/...`) did not change.
