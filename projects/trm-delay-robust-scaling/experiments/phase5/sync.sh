#!/bin/bash
# Push the current code to Katana as a frozen per-stage Phase 5 snapshot, and refresh
# the live copy of the Phase 4 tooling (p4.py status and gates.json are read from it).
#   bash sync.sh <stage>      e.g. t2
set -euo pipefail
STAGE=${1:?stage}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
P5=$REPO/projects/trm-delay-robust-scaling/experiments/phase5
P3=$REPO/projects/trm-delay-robust-scaling/experiments/phase3
DEST=/srv/scratch/z5308482/trm-control
ssh katana "mkdir -p $DEST/p5code/$STAGE/phase5 $DEST/p5code/$STAGE/phase3 $DEST/phase5 $DEST/phase5_logs"
if ssh katana "test -f $DEST/p5code/$STAGE/.frozen"; then
  echo "snapshot $STAGE is frozen (jobs were submitted from it); refusing to overwrite" >&2; exit 1
fi
rsync -rc --delete --exclude=__pycache__ "$REPO/src/" "katana:$DEST/p5code/$STAGE/src/"
rsync -rc --delete --exclude=__pycache__ "$REPO/conf/" "katana:$DEST/p5code/$STAGE/conf/"
rsync -c "$REPO/main.py" "katana:$DEST/p5code/$STAGE/main.py"
rsync -c "$P5"/*.py "$P5"/gates.json "katana:$DEST/p5code/$STAGE/phase5/"
rsync -c "$P3"/w0_eval.py "$P3"/bench_latency.py "katana:$DEST/p5code/$STAGE/phase3/"
rsync -c "$P5"/*.py "$P5"/*.sh "$P5"/gates.json "katana:$DEST/phase5/"
echo "synced snapshot $STAGE"
