"""python -m hybrid.gateway {login,logout,status,serve}"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import secrets
import sys
import webbrowser
from pathlib import Path

from . import app as app_module
from .anthropic_callback import CALLBACK_PORT_RANGE
from .errors import BadRequest, GatewayError
from .login import chatgpt_durability_lines, login_anthropic, login_chatgpt, logout_chatgpt, status_lines
from .paths import (CHATGPT_CONCURRENCY_CHOICES, load_config, pidfile_path, state_dir, token_path,
                    write_private)
from .server import GatewayServer
from .singleton import PidLock
from .store import PROVIDERS
from .tty_prompt import read_hidden

DEFAULT_PORT = 8765


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hybrid.gateway", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("login", "logout"):
        p = sub.add_parser(name, help=f"{name} of one provider")
        p.add_argument("provider", choices=PROVIDERS)
        if name == "login":
            p.add_argument("--paste-only", action="store_true",
                           help="anthropic: skip the loopback callback and only take a pasted code")
            p.add_argument("--new-registration", action="store_true",
                           help="chatgpt: register a new client under the same host id. Use only after OpenAI "
                                "answered invalid_client for the saved client")
    sub.add_parser("status", help="show which logins exist and when they expire")
    serve = sub.add_parser("serve", help="run the OpenAI-compatible gateway")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--chatgpt-concurrency", type=int, choices=CHATGPT_CONCURRENCY_CHOICES, default=None,
                       help="simultaneous ChatGPT calls (default: config.toml [chatgpt] max_concurrency, else 1)")
    serve.add_argument("--meter", type=Path, default=None, help="usage JSONL path (default: <state dir>/usage.jsonl)")
    return parser


def _report_scheduler_exit(task: asyncio.Task) -> None:
    """The scheduler only ends on a bug; say so in the serve log instead of losing it. Requests still refresh on
    demand afterwards."""
    if not task.cancelled() and task.exception() is not None:
        logging.getLogger("bigbrane.gateway").error("proactive refresh stopped: %r; refreshes now happen on demand only",
                                                    task.exception())


def _durability_lines(built: app_module.App) -> list[str]:
    return chatgpt_durability_lines(built.store.load("chatgpt"), pacing=built.pacing, skew_s=built.refresher.skew_s,
                                    proactive_refresh=built.chatgpt_config.proactive_refresh,
                                    last_refresh=built.journal.last("chatgpt"), meter_file=built.meter.path)


async def _serve(root: Path, args: argparse.Namespace) -> None:
    lock = PidLock(pidfile_path(root))
    lock.acquire()                         # a second gateway fails here, before anything else starts
    try:
        built = app_module.build(root, meter_file=args.meter, chatgpt_concurrency=args.chatgpt_concurrency)
        token = os.environ.get("BIGBRANE_GATEWAY_TOKEN") or secrets.token_urlsafe(32)
        write_private(token_path(root), token + "\n")
        server = GatewayServer(built.gateway, token, built.redactor, args.host, args.port)
        port = await server.start()
        print(f"gateway listening on http://{args.host}:{port}/v1  (pid {os.getpid()})", flush=True)
        print(f"bearer token file: {token_path(root)}   usage log: {built.meter.path}", flush=True)
        for line in status_lines(built.store) + _durability_lines(built):
            print(line, flush=True)
        tasks = [asyncio.create_task(scheduler.run()) for scheduler in built.schedulers]
        for task in tasks:
            task.add_done_callback(_report_scheduler_exit)
        print(f"proactive chatgpt refresh: {'on' if tasks else 'off'}", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            for task in tasks:
                task.cancel()
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    root = state_dir()
    try:
        if args.command == "status":
            built = app_module.build(root)
            print("\n".join(status_lines(built.store) + _durability_lines(built)))
            holder = PidLock.holder(pidfile_path(root))
            print(f"gateway: {'running, pid ' + holder if holder else 'not running'}")
        elif args.command == "login":
            if args.new_registration and args.provider != "chatgpt":
                raise BadRequest("--new-registration applies to the chatgpt login only")
            built = app_module.build(root)
            if args.provider == "anthropic":
                ports = None if args.paste_only else range(CALLBACK_PORT_RANGE[0], CALLBACK_PORT_RANGE[1] + 1)
                asyncio.run(login_anthropic(built.store, ask=getpass.getpass, say=print, open_browser=webbrowser.open,
                                            callback_ports=ports, ask_until=read_hidden))
            else:
                asyncio.run(login_chatgpt(built.store, root, load_config(root).chatgpt, say=print,
                                          open_browser=webbrowser.open, new_registration=args.new_registration))
        elif args.command == "logout":
            store = app_module.build(root).store
            if args.provider == "chatgpt":
                asyncio.run(logout_chatgpt(store, say=print))
            else:
                store.delete(args.provider)
                print(f"{args.provider}: local login removed (tokens were not revoked remotely)")
        else:
            asyncio.run(_serve(root, args))
    except GatewayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
