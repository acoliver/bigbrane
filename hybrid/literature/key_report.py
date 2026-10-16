"""The `keys` command: which API keys are configured, from where, and optionally whether they work.

Only the last four characters of a key are ever printed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import httpx

from hybrid.literature import sources
from hybrid.literature.keys import KEY_SPECS, ApiKey, keys_dir


@dataclass(frozen=True)
class KeyLine:
    source: str
    text: str
    ok: bool  # False only when a live check ran and did not get HTTP 200


def report_keys(
    keys: Mapping[str, ApiKey],
    *,
    live: bool,
    http: httpx.Client | None = None,
    home: Path | None = None,
) -> list[KeyLine]:
    """One line per source. With `live`, each found key makes one search request and the HTTP status is reported."""
    lines: list[KeyLine] = []
    owned = http is None
    if live:
        http = http or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        for source, spec in KEY_SPECS.items():
            key = keys.get(source)
            if key is None:
                where = keys_dir(home) / spec.file_name
                lines.append(KeyLine(source, f"{source}: no key (looked at env {spec.env_var}, then {where})", True))
                continue
            text = f"{source}: key found from {key.origin}, ends in {key.last4}"
            ok = True
            if live:
                assert http is not None
                status = _live_status(source, key, http)
                text += f"; live test: {status}"
                ok = status == "HTTP 200"
            lines.append(KeyLine(source, text, ok))
    finally:
        if owned and http is not None:
            http.close()
    return lines


def _live_status(source: str, key: ApiKey, http: httpx.Client) -> str:
    request = sources.MODULES[source].search_request("attention", 1)
    try:
        response = http.get(request.url, params=request.params, headers=key.headers)
    except httpx.TransportError as exc:
        return f"transport error: {key.redact(str(exc))}"
    return f"HTTP {response.status_code}"
