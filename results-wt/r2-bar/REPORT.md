# r2-bar report

## Verdict

INTERIM. The code change is committed and unit-tested, and the replay is running. At the time of writing 4 of the 10
novelty-probe cards had ended and all 4 still died at S2 as `reduces_to_known`. The recheck group (11 cards) and the
batch group (8 cards) had not reached S2 yet, so rules 2 and 3 are not decided. results-wt/r2-bar/SCORE.md is written
by tools/finish.sh when all 12 runs end and holds the final table for rule 1 and rule 2 and the list of flips for rule
3. Read it before relying on this report; the flips section below is filled in from it.

## What changed

- R2A: the reduction critic judges the general learned mechanism with learned parts free; a resemblance that holds only
  with a learned part fixed is recorded in `fixed_learned_parts` (only when the card states a capability the frozen
  form lacks) and cannot carry a cited-source kill.
- R2C: `radical_enough` is replaced by `known_work` (gated), `known_work_sources` and `surprise` (recorded). The host
  acts on `known_work: true` only with a host-counted comparator or a cited source it accepts (paper id present in the
  search results, named textbook section or standard algorithm). `textbook_transplant: true` now gates. Exclusion stays
  with families.toml and the counted-comparator gate. Code: hybrid/research/stages/known_work.py, s2_prior_art.py,
  schemas/reduction_critic.schema.json, prompts/reduction_critic.md.
- R2D: the difference checker reports claims its excerpts cannot support as `excerpt_gaps` (scoped, not a kill); the
  two required closest_prior entries count only relevant sources and family entries fill the rest; a paper the critic
  cited must be a literature entry. Code: stages/difference.py, prompts/difference.md, difference_check.md.
- DECISIONS.md rows R2A, R2C, R2D.

## Checks

- hybrid/tests/research/test_known_work.py: 14 tests pass. Critic fakes in the existing research tests were moved to
  the new fields (critic_answers.py); the full research suite result is in logs/tests-2.log.
- Replay: live_check, routing gateway, copies of the outcome stores, recheck.toml and probe.toml unchanged. Runs and
  logs under results-wt/r2-bar/runs and logs.

## Acceptance rules (pre-registered in STATUS.md)

See SCORE.md. Rule 1 so far: card-17100001, 17100002, 17100005, 17100006 died `reduces_to_known`. Rules 2 and 3 pending.

## Limits

- The 8 batch cards of the replay are the six that froze a learned part plus card-084a2510 and card-909ad773; the other
  four R2 kills of the batch were not replayed. Rule 2 covers the six recheck cards whose earlier S2 had a counted
  reduction; the batch's other counted-reduction cards were not replayed (budget).
- One sample of model judgment per card.
