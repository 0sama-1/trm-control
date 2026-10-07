# Phase 4 poll (run from /home/osama/trm-control/projects/trm-delay-robust-scaling/experiments/phase4)

1. `bash poll.sh`, then `python3 board_doc.py`. Push `dashboard_state/board_doc.json` to the run board with ArtifactData `set` (url https://claude.ai/artifact/AiHU7T9dgTWNWzXnaP8Us9, collection `p4`, doc_id `board`, `file_path` that JSON, `if_version` = the number in `dashboard_state/db_version.txt`), then write the returned version back to `db_version.txt`.
2. Resubmit any job that failed for infrastructure reasons (node/GPU failure) with `p4.py resubmit` via the venv python on Katana. Report code failures to the author.
3. When a stage completes, run `python3 gates_eval.py fetch` and act per stage. Then re-run `board_doc.py --event "<outcome>"`, push again, and STOP and report to the author:
   - **gated stages:** run the evaluator (gb0 [--hopper], gc, a0, ga, gt, h1p) and report the gate outcome;
   - **b0d (IDP diagnostic, K-2):** report passes of the 8190 bar per seed, learning curves (avg-100 and % fully solved episodes per 100k steps), and whether no-delay learns. Draft amendment K-3 for the author's approval before any B1 data, per K-2: confirmatory B1 at 1M if GB0's rule passes at 1M; otherwise exploratory B1 at the best budget with a proposed "clear of floor" criterion and threats;
   - **e (Lipschitz):** analyse the per-checkpoint outputs against the Phase 3 battery (exploratory).

   Never choose B1's budget, set clip norms, or submit a main matrix (B1, T) without the author's approval. Show job counts and walltime estimates first.
4. Report trigger: run `python3 phase_complete.py`. If it prints READY and `dashboard_state/reports_triggered.txt` does not exist:
   - first run `python3 gates_eval.py fetch` so every result is local;
   - write the current timestamp to `dashboard_state/reports_triggered.txt`;
   - launch one background Agent (subagent_type general-purpose, description "Write Phase 4 + overall reports") whose prompt is the full text of `report_brief.md` in this folder;
   - add a board event "Phase 4 complete: report agent started", push, and tell the author.

   When the agent finishes, relay its file list, gate lines and any failures to the author.
5. Stay quiet if nothing changed.
