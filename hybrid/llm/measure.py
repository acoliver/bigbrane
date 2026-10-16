"""Measure latency and input-token overhead of llxprt profiles with one tiny call each.

    .venv-hybrid/bin/python -m hybrid.llm.measure opus sonnet haiku [--web]

Each profile costs one model call. Output is one JSON object per line.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import BackendRequest, LlxprtBackend
from .config import load_config

ROOT = Path(__file__).resolve().parents[2]


async def measure(profile: str, backend: LlxprtBackend, scratch: Path, allow_web: bool) -> dict:
    request = BackendRequest(call_id=f"measure-{profile}", model=profile, system="Reply with one word.",
                             user="Say: pong", allow_web=allow_web, max_output_tokens=None,
                             timeout_s=backend.config.timeout_s, scratch_dir=scratch / profile)
    started = time.monotonic()
    reply = await backend.complete(request)
    return {"profile": profile, "resolved_model": reply.model, "allow_web": allow_web,
            "latency_s": round(time.monotonic() - started, 2), "usage": reply.usage,
            "tool_calls": reply.detail["tools"]["totalCalls"], "text": reply.text.strip()[:40]}


async def main_async(profiles: list[str], allow_web: bool) -> None:
    config = load_config(ROOT / "hybrid" / "config" / "models.toml")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = ROOT / "runs" / f"llm-measure-{stamp}"
    backend = LlxprtBackend(config.backends["llxprt"], work / "llxprt-home")
    for profile in profiles:
        print(json.dumps(await measure(profile, backend, work / "scratch", allow_web)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+")
    parser.add_argument("--web", action="store_true", help="offer the web tools (only matters for overhead)")
    args = parser.parse_args()
    asyncio.run(main_async(args.profiles, args.web))


if __name__ == "__main__":
    main()
