#!/bin/bash
# Push the current code to Katana as a frozen per-stage snapshot, and refresh the
# live copy of the Phase 3 tooling (p3.py status is always run from the live copy).
#   bash sync.sh <stage>      e.g. smoke, s0h, s0b, s0l, s1, s2
set -euo pipefail
STAGE=${1:?stage}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
P3=$REPO/projects/trm-delay-robust-scaling/experiments/phase3
DEST=/srv/scratch/z5308482/trm-control
ssh katana "mkdir -p $DEST/p3code/$STAGE/phase3 $DEST/phase3 $DEST/phase3_logs"
if ssh katana "test -f $DEST/p3code/$STAGE/.frozen"; then
  echo "snapshot $STAGE is frozen (jobs were submitted from it); refusing to overwrite" >&2; exit 1
fi
rsync -rc --delete --exclude=__pycache__ "$REPO/src/" "katana:$DEST/p3code/$STAGE/src/"
rsync -rc --delete --exclude=__pycache__ "$REPO/conf/" "katana:$DEST/p3code/$STAGE/conf/"
rsync -c "$REPO/main.py" "katana:$DEST/p3code/$STAGE/main.py"
rsync -c "$P3"/*.py "katana:$DEST/p3code/$STAGE/phase3/"
rsync -c "$P3"/*.py "$P3"/*.txt "katana:$DEST/phase3/"
# keep the canonical tree in step with local
rsync -rc --exclude=__pycache__ "$REPO/src/" "katana:$DEST/src/"
rsync -rc --exclude=__pycache__ "$REPO/conf/" "katana:$DEST/conf/"
echo "synced snapshot $STAGE"
