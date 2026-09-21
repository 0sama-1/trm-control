#!/bin/bash
# One dashboard poll: pull `p3.py status` from Katana and rebuild the run-board docs.
#   bash poll.sh [--event "kind: text"] ...    -> dashboard_state/out/{state,jobs,log}.json
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/dashboard_state"
ssh -o ConnectTimeout=30 katana 'cd /srv/scratch/z5308482/trm-control/phase3 && /srv/scratch/z5308482/venvs/trm-control/bin/python p3.py status' > "$HERE/dashboard_state/status_latest.json.tmp"
mv "$HERE/dashboard_state/status_latest.json.tmp" "$HERE/dashboard_state/status_latest.json"
python3 "$HERE/dashboard_update.py" "$HERE/dashboard_state/status_latest.json" "$@"
