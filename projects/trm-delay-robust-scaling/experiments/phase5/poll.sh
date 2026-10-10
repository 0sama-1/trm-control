#!/bin/bash
# One status poll: `p5.py status` from Katana -> dashboard_state/status_latest.json, plus a one-line-per-group summary.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/dashboard_state"
ssh -o ConnectTimeout=30 katana 'cd /srv/scratch/z5308482/trm-control/phase5 && /srv/scratch/z5308482/venvs/trm-control/bin/python p5.py status' > "$HERE/dashboard_state/status_latest.json.tmp"
mv "$HERE/dashboard_state/status_latest.json.tmp" "$HERE/dashboard_state/status_latest.json"
python3 - "$HERE/dashboard_state/status_latest.json" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1]))
g = collections.defaultdict(collections.Counter)
for j in d["jobs"]:
    g[(j["stage"], j["group"])][j["state"]] += 1
print("polled", d["polled_at"])
for (st, grp), c in sorted(g.items()):
    print(f"  {st:6s} {grp:28s} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
for j in d["jobs"]:
    if j["state"] == "failed":
        print("FAILED", j["name"], (j.get("error") or "")[-300:])
PY
