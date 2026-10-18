# Not ready: two original Phase7 assertions still fail

B4 downstream wiring and executed KD8/KD4/FIX10/FIX12 controls are implemented. The exact original 26 failing nodes now yield 24 passed and two failed. The final affected-file run yields 420 passed, two failed and one skipped. The remaining original Phase7 aggregate assertions and their expected kill codes were not weakened. See finish-PHASE7-REMAINING.md and finish-phase7-rows.json for all 16 per-card named-gate misses.

Final combined research/literature/evidence-tool verification is running once via finish-combined.sh, with pid in finish-combined.pid, output in finish-combined.log and final exit status in finish-combined.exit. Managed monitor: shell_a3f744d002d6. No duplicate test workers. Earlier partial runs are explicitly marked superseded, pre-fix10 and pre-kd5-kd7. The final run has shell PID 90531 and pytest PID 90533.

Tree content SHA-256: 383c1061d18f6a8c8faa325f12fdf0c86aacdd992d251ebf96a934a9083f524f. See finish-tested-manifest.json, finish-tested.diff and finish-manifest-check.json. Tested files remain unchanged after final launch.

No live LLM/gateway calls, OCR, installations, protected-path edits, staging, commits, pushes or merges by this finish worker. Prior worker edits preserved. The saved PA2 retrieval result stays FAIL 2/4 against 3/4, and the original preregistered guards stay FAIL 9/10 and FAIL 3/6. No follow-up review while tests fail.

Delivery and exact recovery commands: finish-REPORT.md. Evidence capture is separate from the unchanged historical records; bounded samples are not a general equivalence proof.
