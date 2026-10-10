# Phase 5 poll (run from /home/osama/trm-control/projects/trm-delay-robust-scaling/experiments/phase5)

1. `bash poll.sh`, then update the shared run board: `python3 ../phase4/board_doc.py` (it merges Phase 5's status, gates and `dashboard_state/phase5_summary.json`), and push `../phase4/dashboard_state/board_doc.json` with ArtifactData `set` (url https://claude.ai/artifact/AiHU7T9dgTWNWzXnaP8Us9, collection `p4`, doc_id `board`, `file_path` that JSON, `if_version` = the number in `../phase4/dashboard_state/db_version.txt`). Write the returned version back to that file. Do this every poll so the author can follow progress.
   Resubmit any job that failed for infrastructure reasons (node/GPU failure; a job whose log ends "exit 0" with a complete result counts as done) with `p5.py resubmit` via the venv python on Katana. Report code failures to the author.
2. Stay quiet if nothing changed (still push the board).
3. When all 36 Stage T2 jobs are done:
   - run `python3 gates_eval.py fetch` and `python3 gates_eval.py gt2`;
   - report GT2's outcome to the author: learned counts per actor LR, the TRM − MLP contrast if the gate passes, and the secondaries;
   - update `dashboard_state/phase5_summary.json` (status "Phase 5 · gate evaluated", next steps), rebuild the board with `board_doc.py --event "<GT2 outcome>"`, and push;
   - cancel this poll's cron job.
4. Then, as the author asked, write the Phase 5 reports the way Phase 4's were done, in `projects/trm-delay-robust-scaling/phase5/report/` (md + tex + pdf, built with a copy of `phase4/report/build.py`):
   - a stage note and a Phase 5 report;
   - updates of the whole-project documents to cover Phases 0–5: full report, project summary, per-phase summaries (add phase5), Q&A and glossary.

   Use a background agent for the full report, Phase 5 report and stage note (same brief style as `experiments/phase4/report_brief.md`). Write the phase summary, Q&A and glossary updates directly. Report the files and any caveats when done.
