#!/bin/bash
# Push the current code to Katana as a frozen per-stage Phase 4 snapshot, and refresh
# the live copy of the Phase 4 tooling (p4.py status and gates.json are read from it).
#   bash sync.sh <stage>      e.g. smoke, b0, c, a0, a, b1
set -euo pipefail
STAGE=${1:?stage}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
P4=$REPO/projects/trm-delay-robust-scaling/experiments/phase4
P3=$REPO/projects/trm-delay-robust-scaling/experiments/phase3
DEST=/srv/scratch/z5308482/trm-control
ssh katana "mkdir -p $DEST/p4code/$STAGE/phase4 $DEST/p4code/$STAGE/phase3 $DEST/phase4 $DEST/phase4_logs"
if ssh katana "test -f $DEST/p4code/$STAGE/.frozen"; then
  echo "snapshot $STAGE is frozen (jobs were submitted from it); refusing to overwrite" >&2; exit 1
fi
rsync -rc --delete --exclude=__pycache__ "$REPO/src/" "katana:$DEST/p4code/$STAGE/src/"
rsync -rc --delete --exclude=__pycache__ "$REPO/conf/" "katana:$DEST/p4code/$STAGE/conf/"
rsync -c "$REPO/main.py" "katana:$DEST/p4code/$STAGE/main.py"
rsync -c "$P4"/*.py "$P4"/gates.json "katana:$DEST/p4code/$STAGE/phase4/"
rsync -c "$P3"/w0_eval.py "$P3"/bench_latency.py "katana:$DEST/p4code/$STAGE/phase3/"
rsync -c "$P4"/*.py "$P4"/*.sh "$P4"/gates.json "katana:$DEST/phase4/"
echo "synced snapshot $STAGE"
