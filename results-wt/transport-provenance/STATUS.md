# Transport provenance implemented; retrieval result unchanged

Owned paths are hybrid/literature/{provenance.py,transport.py,client.py} and hybrid/tests/literature/test_transport_provenance.py. Call-scoped records reach the existing per-card sink and ledger, including interleaved calls. No research wiring change was needed.

Worker verification: 79 targeted passed; full literature suite 212 passed, one live-only skipped; differential against eeaba07 12/12 unchanged. Final combined/package counts are in ../evidence-recovery/NEXT-CHECKPOINT.md. The added test file contains 22 cases. Existing assertions were not edited.

Cache/cooldown/local operations record zero HTTP requests. Redirects distinguish logical attempts from dispatched hops. Timeouts do not fabricate responses. Retry-After date parsing is measured separately from the unchanged backoff policy. Requests, pacing, retry limits, timeout values, sleeps, cooldowns, cache bytes, ranking and acceptance remain unchanged in the measured differential cases.

Original PA2 retrieval remains FAIL 2/4 against 3/4; named guards remain 9/10 and 3/6. The cause of the initial historical 429 remains unknown. New telemetry cannot reconstruct unsaved historical attempts. REPORT.md contains measurement boundaries, examples and reproduction instructions. No live API/model calls, new review, integration or merge occurred. The latest work has not been independently reviewed.
