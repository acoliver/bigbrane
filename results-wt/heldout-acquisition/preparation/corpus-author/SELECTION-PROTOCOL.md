# Frozen source-selection protocol

Frozen before any new-source fetch. Purpose: construct up to 12 externally authored mechanism cards for the retrieval pilot, not select mechanisms based on experiment results. The inventory v2 registry is conditional: broader known-local inventory is pending. Candidate paper sources are construction inputs acquired after the study cutoff and will not be represented as pre-cutoff exclusions.

## Fixed design

Twelve ordered slots, h01-h12, four proposed families with exactly three slots each: computational geometry, distributed algorithms, topological data analysis, and numerical algorithms. Candidate order is fixed below. Inspect no more than 24 candidates. A candidate is eligible only when an authentic public paper body or publisher/author full text identifies title/authors and supplies the mechanism definition. For each accepted source, retain source URL, retrieval timestamp, body hash, short literal excerpt, section/equation locator, paper aliases, and rationale distinguishing mechanism from the known corpus.

## Candidate order and slot mapping

1. Delaunay triangulation / Bowyer-Watson incremental cavity update (computational geometry)
2. Quickhull / convex hull conflict graph (computational geometry)
3. Bentley-Ottmann plane sweep for segment intersections (computational geometry)
4. Chandy-Lamport distributed snapshot (distributed algorithms)
5. Ricart-Agrawala distributed mutual exclusion (distributed algorithms)
6. Lamport logical clocks / happened-before ordering (distributed algorithms)
7. Persistent homology / persistence pairing (topological data analysis)
8. Mapper algorithm (topological data analysis)
9. Vietoris-Rips filtration (topological data analysis)
10. Conjugate gradient / Krylov residual update (numerical algorithms)
11. GMRES / Arnoldi least-squares iteration (numerical algorithms)
12. Multigrid V-cycle residual restriction/prolongation (numerical algorithms)
13. Fortune sweep-line Voronoi construction (computational geometry)
14. Distributed termination detection (distributed algorithms)
15. Zigzag persistence (topological data analysis)
16. Lanczos tridiagonalization (numerical algorithms)
17. Randomized incremental Delaunay construction (computational geometry)
18. Bully leader election (distributed algorithms)
19. Alpha complex filtration (topological data analysis)
20. Biconjugate gradient stabilized (numerical algorithms)
21. Half-plane intersection (computational geometry)
22. Two-phase commit (distributed algorithms)
23. Morse complex reduction (topological data analysis)
24. Successive over-relaxation (numerical algorithms)

The first three candidates of each family map to that family's three slots. Later candidates substitute only for rejected candidates in the same family, in listed order, with no outcome-dependent choice. Reject on (a) unavailable authentic full source after bounded public lookup; (b) failure to verify title/authors and formal mechanism from the source; (c) known registry or descriptor-prose alias collision; (d) mechanism is merely a paraphrase or close instance of an existing known mechanism; (e) source is not an actual external publication. Record every inspected candidate and disposition. Stop at 24. If any family cannot fill three slots, report fewer units; do not invent sources or change family quotas.

## Bounded source procedure and roles

For each candidate, use at most two public lookup pages and inspect at most one source body, with a body limit of 1200 words. No credentials, paid APIs, LLM/gateway, retrieval experiment, Q0/Q1 calls, result inspection, or adaptive selection. Preserve the exact accessed body when practical; otherwise retain the exact excerpt and SHA-256 of retrieved source bytes. Selection checks known identifiers and normalized titles against v2 registry and the known prose alias file, while recording that unknown aliases and unresolved inventory descriptors remain. A paper's identity/source provenance is independent of a future semantic label.

The corpus author writes neutral mechanism descriptions and blinded metadata only. This role does not assign relevance, semantic correctness labels, reference-work judgments, or vault records. A separate label worker must independently assess candidate relevance and construct any reference vault before arm results become visible. No corpus author claim substitutes for that work.

## Freeze and hashes

This protocol is preregistered by its SHA-256 before source fetch. Candidate list, ordering, family quotas, ceilings, rejection rules, and substitution rules are fixed. Subsequent source receipts and corpus outputs are construction-stage artifacts; their hashes do not amend this protocol.
