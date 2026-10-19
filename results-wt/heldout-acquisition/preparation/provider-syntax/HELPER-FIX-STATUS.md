# Both diagnosed helper contracts now pass offline

Implementation v2 rejects policies outside Q0/Q1 before source dispatch and rejects arXiv queries with no `[A-Za-z0-9]+` tokens for either policy. OpenAlex and S2 keep their existing raw-text behavior, including tokenless text. Valid source aliases and unknown-source rejection are unchanged.

All 244 scoped tests passed: 44 original held-out tests, 77 provider-preflight cases, and 123 new focused cases. Exactly nine diagnostic cases were promoted by deleting their two xfail decorators. Their assertions and all original experimental assertions remain unchanged. Before/after request, encoded HTTP, request-ID, and association-ID checks matched for 72 valid combinations through 144 mock dispatches. The existing audit guard blocks network sockets, DNS, and datagrams while permitting AF_UNIX socket pairs.

The known invalid-input helper contract is no longer PENDING. The corpus, blinded labels, broader exclusion inventory, study acquisition, contrasts, and policy decision remain PENDING or unexecuted. This correction establishes no new acceptance result. PA2 remains FAIL 2/4; the initial 429 cause remains unknown.

The root plan, policy, exclusions, preregistration pins, corpus, production sources, historical manifests, original test logs, documentation snapshot, live probe records, and inventory v1 pins remain unchanged. Only the authorized harness changed among the 20 previously pinned base files. New versioned manifests identify the corrected implementation rather than rewriting the original hashes. See `helper-fix-v2/IMPLEMENTATION-v2.json`, `helper-fix-v2/FROZEN-UNCHANGED-v2.json`, and `helper-fix-v2/TEST-SUMMARY-v2.json`.

No live API calls, documentation fetches, probe reruns, corpus acquisition, independent review, commits, pushes, merges, or integration were performed. The existing exact-once pair of neutral live queries remains the only provider probe evidence.
