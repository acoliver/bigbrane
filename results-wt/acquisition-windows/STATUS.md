# Acquisition-window investigation status

Offline extraction complete. No new network calls, model calls, acceptance rerun, production edits, commits or integration.

Measured: 72 saved queries / 288 source contexts; 170 answered remote raw contexts; 761 raw records parsed without drops; 37 explicit empty pools, 11 exhausted nonempty pools and 122 windows with unseen results. Historical arXiv has 44 cooldown skips and two exhausted operations for these four cards. All twelve fresh raw envelopes were read in main, not copied. Their first-query totals distinguish three empty arXiv pools, one exhausted two-record arXiv pool, and eight truncated OpenAlex/S2 windows.

Generic ordering: 845/848 initial candidates have only one supporting query; 18/20 seeds appear in first queries. Earliest recorded losses for all eleven registrations are projected separately. These metrics do not prove provider-internal eligibility or locate missing works beyond saved windows.

All 274 originals and supplemental input hashes match before/after. Eighteen new scoped offline tests and the unchanged seven prior transport tests pass. Repeated extraction is byte-identical across seven structured outputs. Final preservation checks confirm all 274 originals, 39 supplemental inputs and twelve main raw envelopes unchanged. Main tracked worktree remains clean; other-worker untracked research files were untouched. Original PA2 FAIL 2/4 versus 3/4, PM2 failure and initial historical 429 unknown remain unchanged.

Next action: preregister query/seed policy on an independent held-out mechanism corpus under equal acquisition budgets, excluding these registrations from policy selection. No target tuning or acceptance rerun.
