"""KD2: a materiality replay that stops on the card's own stated table size.

The host replays a conforming variant of a witness under the statement's frozen cost check, which sweeps n over
cost.DEFAULT_NS (64 to 4096). A card whose mechanism reads a finite learned table (k1: a position table P) states that
limit in its ``limits``: a table of N rows has no row N + 1. A variant that restores the strict lookup therefore
crashes with an out-of-bounds index as soon as the sweep passes the rows the card's own init built (k1: 256 rows,
replayed at n = 512). That crash is the stated limit, not a failed replay, so it must not leave the premise open
(KD2, decision KD2).

``table_overrun`` recognizes exactly that case from the check record and nothing else: an out-of-bounds index on
axis 0, raised on a line that reads ``params[KEY][row]`` for a KEY the card's limits name as a table, whose size is
fixed (not one of the statement's own size values) and below the largest replayed n. Any other crash returns None and
is settled as before.
"""
from __future__ import annotations

import re

# The evidence the cost job prints for a candidate crash (hybrid/math/jobmain.py): the numpy message, then the line of
# the submitted source that raised.
OUT_OF_BOUNDS = re.compile(
    r"IndexError: index (?P<index>\d+) is out of bounds for axis (?P<axis>\d+) with size (?P<rows>\d+); "
    r"raised at line \d+ in \w+\(\): (?P<line>.*)")
# A table read by row on that line (params['P'][t]), and no deeper: an index into one row is not a row of the table.
ROW_READ = re.compile(r"params\[(?P<quote>['\"])(?P<table>\w+)(?P=quote)\]\[[^\[\]]*\](?!\s*\[)")


def declared_tables(card_limits: list[dict]) -> set[str]:
    """The parameter names the card's limits call a table ("a learned position table P with N rows")."""
    return {name for limit in card_limits for name in re.findall(r"\btable (\w+)\b", limit["resources"])}


def table_overrun(check: dict, card_limits: list[dict], replay_ns: tuple[int, ...],
                  size_values: set[int]) -> dict | None:
    """The table limit a crashed replay hit, or None when the crash is anything else.

    ``check`` is the undecided check record of the replay, ``card_limits`` the limits frozen with the card's
    statements, ``replay_ns`` the sequence lengths the replay swept and ``size_values`` the values of the statement's
    own size parameters. A failing array with one of those sizes (E has V rows, O has d) grows with the sizes the
    statement sweeps, so it is not a table of fixed size. The result names the table, its rows, the index that was
    read, the lengths inside the table (``inside_ns``) and the ones beyond it."""
    if check["decided"] or check["status"] != "unknown":
        return None
    crash = OUT_OF_BOUNDS.search(check["evidence"] or "")
    if crash is None or crash["axis"] != "0":
        return None
    rows, index = int(crash["rows"]), int(crash["index"])
    tables = declared_tables(card_limits) & {read["table"] for read in ROW_READ.finditer(crash["line"])}
    if len(tables) != 1 or index < rows or rows in size_values or max(replay_ns) <= rows:
        return None
    return {"table": tables.pop(), "rows": rows, "index": index,
            "inside_ns": [n for n in replay_ns if n <= rows], "beyond_ns": [n for n in replay_ns if n > rows]}
