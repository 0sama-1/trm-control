"""Build the run-board document (db doc `p4/board`) from the latest poll and gates.json.

  python board_doc.py [--event "text"] -> dashboard_state/board_doc.json

Keeps a short event log across polls (dashboard_state/board_log.json): state
changes of jobs are logged automatically, plus any --event text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "dashboard_state"
STAGES = {"b0": "B0 · IDP complexity pilot", "b0h": "B0 · Hopper fallback (dropped)", "b0d": "B0 · IDP diagnostic (K-2)",
          "c": "C · LR sweep", "a0": "A pilot · grad norms", "a": "A · training discipline",
          "t": "T · G1 under TD3", "e": "E · Lipschitz, run 1 (path bug)", "e2": "E · Lipschitz (exploratory)", "b1": "B1 · complexity rung"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", action="append", default=[])
    a = ap.parse_args()
    status = json.loads((STATE / "status_latest.json").read_text())
    gates = json.loads((HERE / "gates.json").read_text())
    log_path, prev_path = STATE / "board_log.json", STATE / "board_prev_states.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    now = status["polled_at"]

    jobs, states = [], {}
    for j in status["jobs"]:
        if j["stage"] == "smoke":
            continue
        p, r = j.get("progress") or {}, j.get("result") or {}
        gpu = re.search(r"gpu (.+?) start", p.get("log_head", ""))
        jobs.append({
            "name": j["name"], "stage": j["stage"], "group": j["group"].rstrip("|"), "model": j["model"],
            "cond": j["cond"], "seed": j["seed"], "state": j["state"], "host": j["host"],
            "gpu": gpu.group(1) if gpu else None, "wall_used": j["walltime_used"], "wall_req": j["walltime_req"],
            "steps": j.get("steps"), "episode": p.get("episode"), "avg100": p.get("avg100"),
            "final": r.get("final_avg100_return"), "wall_s": r.get("wall_clock_seconds"),
            "error": (j.get("error") or "")[-400:] or None, "jobid": j["jobid"],
        })
        states[j["name"]] = j["state"]
        if prev.get(j["name"]) != j["state"] and j["state"] in ("running", "done", "failed"):
            if j["state"] != "running" or prev.get(j["name"]) in (None, "queued"):
                log.append({"t": now, "text": f"{j['name']} {j['state']}", "kind": j["state"]})
    for e in a.event:
        log.append({"t": now, "text": e, "kind": "note"})
    log = log[-60:]
    log_path.write_text(json.dumps(log))
    prev_path.write_text(json.dumps(states))

    doc = {
        "polled_at": now,
        "stages": [{"id": k, "label": v} for k, v in STAGES.items() if any(j["stage"] == k for j in jobs)],
        "jobs": jobs,
        "gates": [{"id": g["id"], "name": g["name"], "stage": g["stage"], "status": g["status"],
                   "detail": " ".join(t for t in [g.get("detail"), g.get("notes")] if t), "rule": g["rule"]} for g in gates["gates"]],
        "amendments": [{"id": x["id"], "date": x["date"], "text": x["text"]} for x in gates.get("amendments", [])],
        "log": log,
        # Curated result tables (dashboard_state/findings.json), updated by hand as stages report.
        "findings": json.loads((STATE / "findings.json").read_text()) if (STATE / "findings.json").exists() else [],
        # Curated phase summary (dashboard_state/summary.json), shown at the top of the board.
        "summary": json.loads((STATE / "summary.json").read_text()) if (STATE / "summary.json").exists() else None,
    }
    (STATE / "board_doc.json").write_text(json.dumps(doc))
    print(f"board doc: {len(jobs)} jobs, {len(log)} log entries, {len(json.dumps(doc)) // 1024} KB")


if __name__ == "__main__":
    main()
