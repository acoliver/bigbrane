#!/usr/bin/env bash
# Replay of the changed S2 through live_check, routing gateway, against copies of the outcome stores.
# Usage: run_replay.sh GROUP   (recheck | batch8 | probeA | probeB)
set -uo pipefail
WT=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
MAIN=/Volumes/XS1000/acoliver/projects/bigbrane
PY=$MAIN/.venv-hybrid/bin/python
OUT=$WT/results-wt/r2-bar
cd "$WT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WT"
REASON="r2-bar replay: changed S2 radicality and difference bars (decisions R2A, R2C, R2D), stop after S2"
batch() {  # name cards
  $PY -m hybrid.research.live_check --source-run $MAIN/runs/discovery-live-20261017 --cards "$2" \
    --run-dir $OUT/runs/$1 --store $MAIN/results/discovery-live-20261017/outcomes-before --store-copy $OUT/stores/$1 \
    --config $OUT/configs/recheck.toml --routing gateway --role explorer --max-wall-s 7200 --reason "$REASON" \
    > $OUT/logs/$1.log 2>&1
  echo "$1 exit $?" >> $OUT/logs/exit.txt
}
probe() {  # cards...
  for CARD in "$@"; do
    $PY -m hybrid.research.live_check --source-run $MAIN/results/novelty-probe-20261017/later/source --cards "$CARD" \
      --run-dir $OUT/runs/probe-$CARD --store $MAIN/results/novelty-probe-20261017/later/empty-store \
      --store-copy $OUT/stores/probe-$CARD --config $OUT/configs/probe.toml --routing gateway --role explorer \
      --max-wall-s 7200 --reason "$REASON" > $OUT/logs/probe-$CARD.log 2>&1
    echo "probe-$CARD exit $?" >> $OUT/logs/exit.txt
  done
}
case "$1" in
  recheck) batch recheck card-76ee768c,card-a4f684fb,card-440f0204,card-7e993aab,card-83bd879a,card-89248ee5,card-8ad6004f,card-d6dfeac2,card-62785c10,card-c5524960,card-97baae43 ;;
  batch8) batch batch8 card-0fe3943c,card-3485a13f,card-a775f78d,card-ebf0e99d,card-f5ee7edb,card-8154d3b4,card-084a2510,card-909ad773 ;;
  probeA) probe card-17100001 card-17100002 card-17100003 card-17100004 card-17100009 ;;
  probeB) probe card-17100005 card-17100006 card-17100007 card-17100008 card-1710000a ;;
esac
