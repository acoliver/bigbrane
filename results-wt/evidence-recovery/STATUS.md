# Two exact-code gaps resolved; fourteen remain

The production reference contract now admits `parityunionfind582f8abf` and `finitePGLversionspaced86dca4b`. The actual unchanged Phase7 replay restores `reduces_to_known:other_known` for original cards `582f8abf` and `d86dca4b`: **2/16 gaps closed**, **16 to 14 misses**, **5/19 required codes matched**. All 19 other rows keep their previous codes, including both original S2 passes.

Final targeted run: **89 passed, two unchanged Phase7 aggregate failures; exit 1**. Schema synchronization, unsupported-name/field rejection, source reexecution, complete capture reproduction, output perturbations and retained negative/mutant tests pass. Acceptance predicates, family policy, tolerances, expected codes, preregistration and original sources are unchanged. Main remains `p4-diagnosis`.

The original full capture remains local and byte-identical. Compact positive declarations (291,268 bytes), negative controls (5,020 bytes), source manifest (7,892 bytes) and small original program fixtures retain checksums and source/ledger provenance. Full matrices and publication snapshots remain local without packaging them. No original was deleted.

SUGS remains refuted on 1,174/2,418 accepted allocations. Concept evidence retains 84 empty-support and two roundoff-domain unsupported queries. Neither receives acceptance support. Tagged-memory policy mismatch and the other fourteen misses remain visible in `BINDING.md` and `binding-phase7-rows.json`.

Tested-tree SHA256: `82278025dc8176e1ce4cb408b232122a95dffc10b3a42488fe74096fc9de50e3`. File inventory/hashes: `binding-inventory.json`, `binding-tested-tree.json`. Tests: `binding-delivery.log`, `binding-delivery-junit.xml`. Exact accounting: `BINDING.md`, `binding-summary.json`. Earlier `HISTORICAL.md` and `OPERATIONS.md` retain the prior schema-frozen task's history.

No live LLM, review, full-suite run, main-code edit, integration, commit, push or merge. Concurrent literature edits are outside this task's ownership.
