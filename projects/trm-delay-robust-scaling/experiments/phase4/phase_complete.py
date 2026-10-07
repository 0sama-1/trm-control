"""Is Phase 4 finished? Used by the poll to decide when to start the report-writing agent.

  python phase_complete.py      prints READY or WAITING plus the reasons; exit 0 only when READY

READY means: the latest poll (dashboard_state/status_latest.json) shows no Phase 4 job
queued, held or running, and every gate in gates.json except the exploratory Stage E entry
has an outcome. A gate counts as resolved once it has any status other than pending/armed;
when the author decides a stage will not run, its gate is set to "not run" so the phase can close.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPEN_GATE = {"pending", "armed"}
OPEN_JOB = {"queued", "held", "running", "unknown"}


def main() -> None:
    status = json.loads((HERE / "dashboard_state" / "status_latest.json").read_text())
    gates = json.loads((HERE / "gates.json").read_text())["gates"]
    reasons = []
    busy = [j["name"] for j in status["jobs"] if j["stage"] != "smoke" and j["state"] in OPEN_JOB]
    if busy:
        reasons.append(f"{len(busy)} job(s) still queued/running")
    open_gates = [g["id"] for g in gates if g["id"] != "E" and g["status"] in OPEN_GATE]
    if open_gates:
        reasons.append("gates without an outcome: " + ", ".join(open_gates))
    if reasons:
        print("WAITING: " + "; ".join(reasons))
        sys.exit(1)
    print("READY")


if __name__ == "__main__":
    main()
