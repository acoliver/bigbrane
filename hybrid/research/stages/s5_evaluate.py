"""S5 evaluate: the trusted bench runs the candidate and the baselines through the executor.

Before the bench runs, `falsify.prepare` has the falsifier_designer write card-specific falsifier tasks and
validates them in the executor (s5.falsifiers, default true). The validated tasks run in the bench next to
the standard tasks, for the candidate and every baseline; the evaluator job scores them and decides each
pre-registered rule, under `falsifiers` in the bench result.

The spec says S5 records results with no pass or fail. The store needs a verdict to advance a
card, so S5 records `pass` when the candidate job produced results and `fail` when the candidate
job itself failed (nothing for S6 to analyse). The raw bench output is stored verbatim as a
`bench_raw` artifact and a summary in the research `bench_result` schema as `bench_result`.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hybrid.research.schemas import validate
from hybrid.research.stages import falsify
from hybrid.research.stages.common import StageContext, StageError, claimed_exponent


class BenchEvaluator:
    """Calls hybrid.bench.evaluate.evaluate_candidate with a Podman Executor or a LocalExecutor."""

    def __init__(self, mode: str, bench_config: Path, run_id: str, run_dir: Path):
        if mode not in ("podman", "local"):
            raise ValueError(f"s5.mode must be podman or local, got {mode!r}")
        self.mode, self.bench_config, self.run_id, self.run_dir = mode, Path(bench_config), run_id, Path(run_dir)

    async def evaluate(self, card_id: str, candidate_dir: Path, exponent: float | None,
                       falsifiers: list[Path] | None = None) -> dict:
        from hybrid.bench.evaluate import evaluate_candidate
        from hybrid.bench.local import LocalExecutor
        from hybrid.executor.executor import Executor

        base = self.run_dir / "bench_runs" / card_id
        attempt = 0
        while (base / f"attempt-{attempt}").exists():
            attempt += 1
        out = base / f"attempt-{attempt}"
        out.mkdir(parents=True)
        receipts = out / "receipts"
        executor = (LocalExecutor(self.run_id, receipts) if self.mode == "local"
                    else Executor(self.run_id, receipts))
        log_path = out / "bench.log"

        def log(message: str) -> None:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")

        return await asyncio.to_thread(evaluate_candidate, str(candidate_dir), str(self.bench_config), str(out),
                                       executor, claimed_exponent=exponent, log=log, falsifiers=falsifiers)


def summarize(raw: dict, card_id: str) -> dict:
    """Map bigbrane.bench_result/1 onto the research bench_result schema used by the report."""
    def task_block(entrant: dict) -> dict:
        block = {}
        for task, data in entrant.get("tasks", {}).items():
            metrics = {}
            for size, m in data["sizes"].items():
                if m.get("valid"):
                    metrics[f"n{size}_kl"] = m.get("kl")
                    metrics[f"n{size}_acc"] = m.get("acc")
                else:
                    metrics[f"n{size}_invalid"] = "; ".join(m.get("problems", []))[:300]
            ttt = data.get("tokens_to_target")
            if ttt:
                metrics["tokens_to_target_reached"] = ttt.get("reached")
                metrics["tokens_to_target"] = ttt.get("interp_tokens")
            block[task] = metrics
        return block

    entrants = raw["entrants"]
    candidate = entrants.get("candidate", {})
    slopes = {}
    for name, entrant in entrants.items():
        fit = (entrant.get("scaling") or {}).get("time_fit_tail")
        if fit and isinstance(fit.get("slope"), (int, float)):
            slopes[name] = float(fit["slope"])
    summary = {
        "card_id": card_id,
        "per_task": task_block(candidate) if candidate.get("status") == "ok" else {},
        "baselines": {name: task_block(e) for name, e in entrants.items()
                      if e.get("role") == "baseline" and e.get("status") == "ok"},
        "scaling_slopes": slopes,
        "receipts": [{"entrant": name, "job_id": r.get("job_id"), "status": r.get("status")}
                     for name, r in raw.get("host", {}).get("receipts", {}).items()],
        "candidate_status": candidate.get("status"),
        "scaling_verdict": (candidate.get("scaling") or {}).get("verdict"),
        "trusted": raw.get("trusted"),
        "falsifiers": falsifier_summary(raw),
    }
    return validate("bench_result", summary)


def falsifier_summary(raw: dict) -> dict:
    """Per falsifier task: each entrant's overall acc and kl, and each rule's decision."""
    out = {}
    for task_id, block in (raw.get("falsifiers") or {}).items():
        out[task_id] = {
            "title": block["title"],
            "entrants": {name: ({"acc": e["metrics"]["acc"], "kl": e["metrics"]["kl"]} if e["status"] == "ok"
                                else {"status": e["status"]}) for name, e in block["entrants"].items()},
            "rules": [{"claims": d["claims"], "outcome": d["outcome"], "reason": d["reason"]}
                      for d in block["decisions"]],
        }
    return out


def candidate_dir(ctx: StageContext, card_id: str) -> Path:
    built = ctx.card_events(card_id, "candidate_built")
    if not built:
        raise StageError(f"{card_id} has no candidate_built event")
    return ctx.run_dir / built[-1]["candidate_dir"]


async def run(ctx: StageContext, card_id: str) -> str | None:
    card = ctx.card(card_id)
    if card["status"]["verdicts"].get("S4", {}).get("verdict") != "pass":
        raise AssertionError(f"{card_id} reached S5 without an S4 pass")
    exponent = claimed_exponent(card["claims"]["complexity"]["total_sequence"])
    falsifier_dirs, falsifier_ids = await falsify.prepare(ctx, card_id)
    try:
        raw = await ctx.bench.evaluate(card_id, candidate_dir(ctx, card_id), exponent, falsifiers=falsifier_dirs)
    except (RuntimeError, OSError) as exc:
        raise StageError(f"bench evaluation failed: {type(exc).__name__}: {exc}") from exc
    ctx.put_json_artifact("bench_raw", f"{card_id}-{ctx.store.latest_version(card_id)}.json", raw, card_id)
    summary = summarize(raw, card_id)
    ctx.put_json_artifact("bench_result", f"{card_id}-{ctx.store.latest_version(card_id)}.json", summary, card_id)
    receipts = raw.get("host", {}).get("receipts", {})
    ids = falsifier_ids + [r["job_id"] for r in receipts.values() if r.get("job_id")]
    ids += [r["job_id"] for per_task in raw.get("host", {}).get("falsifier_receipts", {}).values()
            for r in per_task.values() if r.get("job_id")]
    evaluator = raw.get("host", {}).get("evaluator_receipt", {}).get("job_id")
    if evaluator:
        ids.append(evaluator)
    status = summary["candidate_status"]
    if status != "ok":
        detail = raw["entrants"].get("candidate", {}).get("job_detail", "")
        ctx.verdict(card_id, "S5", "fail", [f"candidate job did not produce results: {status} {detail}"[:800]], ids)
        return None
    reasons = [f"bench results recorded (trusted={raw.get('trusted')}); scaling verdict "
               f"{summary['scaling_verdict']}, claimed exponent {exponent}, slopes "
               + json.dumps(summary["scaling_slopes"])]
    if ctx.cfg["s5"]["falsifiers"]:
        reasons.append(f"falsifier tasks run: {len(summary['falsifiers'])}")
        for task_id, block in summary["falsifiers"].items():
            for rule in block["rules"]:
                reasons.append(f"falsifier {task_id} rule for {', '.join(rule['claims'])}: {rule['outcome']} "
                               f"({rule['reason']})"[:400])
    ctx.verdict(card_id, "S5", "pass", reasons, ids)
    return "S6"
