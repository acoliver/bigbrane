#!/usr/bin/env bash
# Waits for the 12 replay runs to end, scores them and commits the score. REPORT.md is written from SCORE.md by hand.
OUT=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/r2-bar
while [ "$(wc -l < $OUT/logs/exit.txt 2>/dev/null || echo 0)" -lt 12 ]; do sleep 60; done
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
python3 $OUT/tools/score.py > /dev/null
echo "all runs ended $(date -u +%FT%TZ)" > $OUT/logs/finished.txt
git add results-wt/r2-bar/SCORE.md && git commit -qm "r2-bar: replay score" && echo committed >> $OUT/logs/finished.txt
