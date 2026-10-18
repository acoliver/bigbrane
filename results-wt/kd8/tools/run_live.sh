#!/usr/bin/env bash
# PA2 live check through live_check (gateway routing), against copies of the outcome stores. Usage: run_live.sh GROUP
set -uo pipefail
WT=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
MAIN=/Volumes/XS1000/acoliver/projects/bigbrane
PY=$MAIN/.venv-hybrid/bin/python
OUT=$WT/results-wt/kd8
CFG=$WT/results-wt/r2-bar/configs
cd "$WT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WT"
REASON="kd8/pa2 live check: cross-field prior-art search and KD8 restatement rule, stop after S2"
flip() {  # name cards
  $PY -m hybrid.research.live_check --source-run $MAIN/runs/discovery-live-20261017 --cards "$2" \
    --run-dir $OUT/runs/$1 --store $MAIN/results/discovery-live-20261017/outcomes-before --store-copy $OUT/stores/$1 \
    --config $CFG/recheck.toml --routing gateway --role explorer --max-wall-s 5400 --reason "$REASON" > $OUT/logs/$1.log 2>&1
  echo "$1 exit $?" >> $OUT/logs/exit.txt
}
probe() {
  for CARD in "$@"; do
    $PY -m hybrid.research.live_check --source-run $MAIN/results/novelty-probe-20261017/later/source --cards "$CARD" \
      --run-dir $OUT/runs/probe-$CARD --store $MAIN/results/novelty-probe-20261017/later/empty-store \
      --store-copy $OUT/stores/probe-$CARD --config $CFG/probe.toml --routing gateway --role explorer \
      --max-wall-s 5400 --reason "$REASON" > $OUT/logs/probe-$CARD.log 2>&1
    echo "probe-$CARD exit $?" >> $OUT/logs/exit.txt
  done
}
case "$1" in
  flipA) flip flipA card-3485a13f,card-f5ee7edb,card-8154d3b4 ;;
  flipB) flip flipB card-a4f684fb,card-89248ee5 ;;
  probe) probe card-17100001 card-17100003 card-17100006 card-17100008 ;;
esac
