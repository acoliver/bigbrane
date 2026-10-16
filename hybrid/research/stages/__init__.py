"""Stage implementations for the research loop (RESEARCH-SYSTEM-SPEC.md section 3).

Each stage module exposes `async def run(ctx, card_id) -> next_stage | None`, except S0, whose
`explore(ctx, task)` returns the new card id or None. M and C (DISCOVERY-PIPELINE-PLAN.md sections 4 and 5) replace
S3 when the run config sets s3.mode = "math". R (section 7) runs between S1 and S2 when refine.enabled.
"""
from hybrid.research.stages import (c_challenge, m_math, r_refine, s1_wellformed, s2_prior_art, s3_feasibility,
                                    s4_construct, s5_evaluate, s6_verdict)

STAGE_RUNNERS = {
    "S1": s1_wellformed.run,
    # refine.enabled: the refinement dialogue between S1 and S2 (Phase 7, DISCOVERY-PIPELINE-PLAN.md section 7).
    "R": r_refine.run,
    "S2": s2_prior_art.run,
    "S3": s3_feasibility.run,
    # s3.mode = "math": the math stage and the challenge stage in place of S3 (common.MATH_ORDER).
    "M": m_math.run,
    "C": c_challenge.run,
    "S4": s4_construct.run,
    "S5": s5_evaluate.run,
    "S6": s6_verdict.run,
}
