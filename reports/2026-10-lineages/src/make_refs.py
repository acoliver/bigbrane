#!/usr/bin/env python3
"""Build src/references.tex from the numbered reference lists of the two research notes.

Inputs (paths relative to this script):
  ../research-pre2017.md   numbered list under "## Numbered references"  -> keys pre<N>
  ../research-post2017.md  numbered list under "## 14. References"       -> keys post<N>
  lineages-brief.tex       the brief whose \\cite keys decide what is emitted

Keys use each file's own numbering, so \\cite{pre37} is item 37 of the pre-2017 list.

Deduplication
  Every URL in an entry is normalized: scheme dropped (http and https are equal), host
  lowercased with "www." removed, trailing slash removed, query and fragment kept only
  when the path is otherwise empty, and arXiv variants folded together (abs, pdf, html,
  ar5iv, version suffixes such as v2) into "arxiv:<id>". Two entries that share any
  normalized URL describe the same work. As a second key, titles are compared after
  lowercasing, dropping parentheticals and non-alphanumerics; this catches the one work
  both notes cite through different hosts (Chinchilla: arXiv in one, NeurIPS in the other). The group's primary key is its first member in
  reading order (all pre keys come before all post keys, then by number); the other
  keys are aliases.

Alias handling (chosen approach: rewrite)
  The script rewrites alias citations in lineages-brief.tex to the primary key, in place,
  before collecting citations. It emits no \\newcommand mappings. After a run the .tex only
  contains primary keys, so the key a reader sees in the source is the key that has a
  \\bibitem. The aliases are listed in a comment at the top of references.tex.

What is emitted
  Only cited primary entries, in order of first citation in the .tex, so LaTeX numbers the
  list in reading order. Each \\bibitem carries authors, title, year, venue and one \\url
  (the first URL the research note gives for the entry). Text is LaTeX-escaped; every
  non-ASCII character is converted to a LaTeX accent command or symbol, because tectonic
  runs XeTeX with T1 Latin Modern, which lacks glyphs such as U+0107.

Failure policy
  Fail fast: a cited key missing from both lists, an entry that cannot be parsed, or a
  non-ASCII character without a mapping stops the script with a non-zero exit.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRE_MD = HERE.parent / "research-pre2017.md"
POST_MD = HERE.parent / "research-post2017.md"
TEX = HERE / "lineages-brief.tex"
OUT = HERE / "references.tex"

URL_RE = re.compile(r"https?://[^\s<>]+")
YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
CITE_RE = re.compile(r"\\cite\{([^}]*)\}")

# Post-2017 entries whose free-form layout defeats the "Authors. Title. Venue." split.
# Values are (authors, title, venue); the text is copied from the research note.
POST_OVERRIDES: dict[int, tuple[str, str, str]] = {
    22: ("Wang, Ping et al. (NVIDIA)",
         "Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study",
         "EMNLP 2023"),
    39: ("LeCun", "Interview at AIM 2026, transcript (secondary)", "October 2026"),
    73: ("Symbolica AI",
         '"Today we set a new ARC-AGI-2 public eval SOTA: 85.28%."',
         "LinkedIn, 12 February 2026 (vendor claim)"),
    80: ("Authors not recorded in the retrieved copy",
         "Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures",
         "OpenReview submission, 2025"),
}

# Non-ASCII characters that occur in the reference lists, mapped to LaTeX.
CHAR_MAP: dict[str, str] = {
    "á": r"\'{a}", "à": r"\`{a}", "ä": r"\"{a}", "â": r"\^{a}", "ã": r"\~{a}", "å": r"\r{a}",
    "é": r"\'{e}", "è": r"\`{e}", "ë": r"\"{e}", "ê": r"\^{e}", "ę": r"\k{e}", "ě": r"\v{e}",
    "í": r"\'{\i}", "ï": r"\"{\i}", "î": r"\^{\i}",
    "ó": r"\'{o}", "ö": r"\"{o}", "ô": r"\^{o}", "ø": r"{\o}",
    "ú": r"\'{u}", "ü": r"\"{u}", "ů": r"\r{u}",
    "ç": r"\c{c}", "ć": r"\'{c}", "č": r"\v{c}", "Č": r"\v{C}", "ñ": r"\~{n}", "ń": r"\'{n}",
    "š": r"\v{s}", "Š": r"\v{S}", "ž": r"\v{z}", "ř": r"\v{r}", "ý": r"\'{y}",
    "ł": r"{\l}", "Ł": r"{\L}", "ß": r"{\ss}", "É": r"\'{E}", "Ö": r"\"{O}", "Ü": r"\"{U}",
    "–": "-", "—": ", ", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}",
    "×": r"$\times$", "\u00a0": "~",
}

ASCII_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


class RefError(Exception):
    pass


@dataclass
class Entry:
    key: str
    authors: str
    title: str
    year: str
    venue: str
    urls: list[str]
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- escaping

def escape_plain(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in (OPEN_Q, CLOSE_Q, OPEN_SQ):
            out.append(ch)
        elif ch in ASCII_ESCAPES:
            out.append(ASCII_ESCAPES[ch])
        elif ord(ch) < 128:
            out.append(ch)
        elif ch in CHAR_MAP:
            out.append(CHAR_MAP[ch])
        else:
            raise RefError(f"no LaTeX mapping for character {ch!r} (U+{ord(ch):04X}) in: {text}")
    return "".join(out)


OPEN_Q, CLOSE_Q, OPEN_SQ = "\x01", "\x02", "\x03"


def to_latex(text: str) -> str:
    """Escape text and convert markdown *italic* and quotes to LaTeX.

    Quotes are replaced by sentinels before escaping, so the accent command
    for a character such as u-umlaut is never mistaken for a quotation mark.
    """
    count = 0

    def quote(_: re.Match) -> str:
        nonlocal count
        count += 1
        return OPEN_Q if count % 2 else CLOSE_Q

    text = re.sub(r'"', quote, text.replace("**", ""))
    text = re.sub(r"(?:(?<=\s)|^)'", OPEN_SQ, text)
    parts = re.split(r"(\*[^*]+\*)", text)
    pieces = []
    for part in parts:
        if len(part) > 2 and part.startswith("*") and part.endswith("*"):
            pieces.append(r"\emph{" + escape_plain(part[1:-1]) + "}")
        else:
            pieces.append(escape_plain(part))
    return "".join(pieces).replace(OPEN_Q, "``").replace(CLOSE_Q, "''").replace(OPEN_SQ, "`")


def sentence(text: str) -> str:
    """Terminate a bibliography field with a period unless it already ends a sentence."""
    text = text.rstrip()
    return text if text.endswith((".", "?", "!", ".''", "?''", ".}", "?}", ".''}", "?''}")) else text + "."


def escape_url(url: str) -> str:
    # \url handles most characters verbatim; % and # are escaped so the argument
    # survives being read inside the bibliography.
    return url.replace("%", r"\%").replace("#", r"\#")


# ---------------------------------------------------------------- URL normalization

ARXIV_RE = re.compile(
    r"^(?:ar5iv\.labs\.arxiv\.org|ar5iv\.org|arxiv\.org)/(?:abs|pdf|html)/"
    r"([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?$"
)


def normalize_url(url: str) -> str:
    u = re.sub(r"^https?://", "", url.strip())
    host, _, rest = u.partition("/")
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    path = rest.split("#", 1)[0]
    norm = f"{host}/{path}".rstrip("/")
    m = ARXIV_RE.match(norm)
    if m:
        return f"arxiv:{m.group(1)}"
    return norm


def clean_url(raw: str) -> str:
    return raw.rstrip(").,;")


# ---------------------------------------------------------------- parsing

def reference_lines(md: Path, heading_pattern: str) -> dict[int, str]:
    text = md.read_text(encoding="utf-8")
    m = re.search(heading_pattern, text, flags=re.M)
    if not m:
        raise RefError(f"reference heading not found in {md.name}")
    section = text[m.end():]
    entries: dict[int, str] = {}
    for line in section.splitlines():
        mm = re.match(r"^(\d+)\.\s+(.*)$", line)
        if mm:
            entries[int(mm.group(1))] = mm.group(2).strip()
        elif line.startswith("#"):
            break
    numbers = sorted(entries)
    if numbers != list(range(1, len(numbers) + 1)):
        raise RefError(f"{md.name}: reference numbering is not contiguous: {numbers}")
    return entries


def initials(name: str) -> str:
    """'Chomsky, Noam' -> 'Chomsky, N.'; 'Kneser, Reinhard' -> 'Kneser, R.'"""
    last, _, first = name.partition(",")
    if not first.strip():
        return name.strip()
    parts = re.split(r"([\s-]+)", first.strip())
    out = "".join(p if re.fullmatch(r"[\s-]+", p) else (p if p.endswith(".") and len(p) <= 3 else p[0] + ".")
                  for p in parts if p)
    return f"{last.strip()}, {out}"


def short_authors(names: list[str]) -> str:
    """Pre-2017 author lists: initials for given names, first author plus et al. beyond three."""
    names = [initials(n) for n in names if n.strip()]
    if len(names) > 3:
        return names[0] + " et al."
    return "; ".join(names)


def strip_labels(meta: str) -> str:
    """Drop link labels such as 'Primary copy:' from the text before the first URL."""
    segments = re.split(r"(?<=\.)\s+", meta.strip())
    keep = [s for s in segments if s and not s.rstrip().endswith(":")
            and not re.match(r"^(Verified|Primary|Publisher|DOI|Author|Archived)\b", s)]
    return " ".join(keep).strip().rstrip(".").strip()


def parse_pre(num: int, line: str) -> Entry:
    m = re.match(r"^\*\*(.+?) \(([^)]*)\)\.\*\*\s+\*(.+?)\*\.?\s*(.*)$", line)
    if not m:
        raise RefError(f"pre{num}: cannot parse: {line}")
    authors_raw, year_text, title, rest = m.groups()
    urls = [clean_url(u) for u in URL_RE.findall(rest)]
    if not urls:
        raise RefError(f"pre{num}: no URL")
    first = URL_RE.search(rest)
    venue = strip_labels(rest[: first.start()])
    if authors_raw.endswith(", editors"):
        authors = short_authors(authors_raw[: -len(", editors")].split(";")) + " (eds.)"
    else:
        authors = short_authors(authors_raw.split(";"))
    return Entry(f"pre{num}", authors, title, year_text, venue, urls)


def parse_post(num: int, line: str) -> Entry:
    urls = [clean_url(u) for u in URL_RE.findall(line)]
    if not urls:
        raise RefError(f"post{num}: no URL")
    meta = line[: URL_RE.search(line).start()]
    if num in POST_OVERRIDES:
        authors, title, venue = POST_OVERRIDES[num]
    else:
        protected = re.sub(r"et al\.(?=\s*\()", "et al\u00a7", meta)
        segments = [s.replace("\u00a7", ".") for s in re.split(r"(?<=[.?])\s+", protected.strip()) if s]
        if len(segments) < 2:
            raise RefError(f"post{num}: cannot split authors and title: {line}")
        authors_seg, title_seg, *venue_segs = segments
        authors = authors_seg.rstrip(".")
        names = [n for n in authors.split(", ")]
        if len(names) > 3 and "(" not in authors:
            authors = names[0] + " et al."
        title = title_seg.rstrip(".") if not title_seg.endswith("?") else title_seg
        venue = strip_labels(" ".join(venue_segs))
    year_m = YEAR_RE.search(venue) or YEAR_RE.search(title) or YEAR_RE.search(meta)
    # The research note gives no date for a few web pages; those print as "n.d.".
    year = year_m.group(1) if year_m else "n.d."
    return Entry(f"post{num}", authors, title, year, venue, urls)


def load_entries() -> list[Entry]:
    pre = reference_lines(PRE_MD, r"^## Numbered references\s*$")
    post = reference_lines(POST_MD, r"^## 14\. References\s*$")
    entries = [parse_pre(n, s) for n, s in sorted(pre.items())]
    entries += [parse_post(n, s) for n, s in sorted(post.items())]
    print(f"parsed {len(pre)} pre-2017 and {len(post)} post-2017 references")
    return entries


# ---------------------------------------------------------------- dedup and output

def normalize_title(title: str) -> str:
    t = re.sub(r"\([^)]*\)", "", title.lower())
    return "title:" + re.sub(r"[^a-z0-9]+", "", t)


def deduplicate(entries: list[Entry]) -> tuple[dict[str, Entry], dict[str, str]]:
    primary_by_url: dict[str, Entry] = {}
    primaries: dict[str, Entry] = {}
    alias_to_primary: dict[str, str] = {}
    for e in entries:
        norms = [normalize_url(u) for u in e.urls] + [normalize_title(e.title)]
        owner = next((primary_by_url[n] for n in norms if n in primary_by_url), None)
        if owner is None:
            primaries[e.key] = e
            for n in norms:
                primary_by_url[n] = e
        else:
            owner.aliases.append(e.key)
            alias_to_primary[e.key] = owner.key
            for n in norms:
                primary_by_url.setdefault(n, owner)
    return primaries, alias_to_primary


def rewrite_aliases(tex: str, alias_to_primary: dict[str, str]) -> tuple[str, int]:
    changes = 0

    def fix(m: re.Match) -> str:
        nonlocal changes
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        new = []
        for k in keys:
            target = alias_to_primary.get(k, k)
            if target != k:
                changes += 1
            if target not in new:
                new.append(target)
        return r"\cite{" + ",".join(new) + "}"

    return CITE_RE.sub(fix, tex), changes


def cited_keys(tex: str) -> list[str]:
    # Strip TeX comments: an unescaped % to end of line. "\%" is a literal percent sign.
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())
    order: list[str] = []
    for m in CITE_RE.finditer(body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in order:
                order.append(k)
    return order


def bibitem(e: Entry) -> str:
    venue = e.venue
    year = e.year
    tail_parts = []
    if venue:
        tail_parts.append(to_latex(venue))
    if year and year not in venue:
        tail_parts.append(to_latex(year))
    tail = ", ".join(tail_parts)
    text = f"{sentence(to_latex(e.authors))} {sentence(chr(92) + 'emph{' + to_latex(e.title) + '}')}"
    if tail:
        text += f" {sentence(tail)}"
    text += f" \\url{{{escape_url(e.urls[0])}}}"
    return f"\\bibitem{{{e.key}}} {text}"


def main() -> int:
    entries = load_entries()
    primaries, alias_to_primary = deduplicate(entries)

    tex = TEX.read_text(encoding="utf-8")
    new_tex, changes = rewrite_aliases(tex, alias_to_primary)
    if changes:
        TEX.write_text(new_tex, encoding="utf-8")
        print(f"rewrote {changes} alias citation(s) in {TEX.name}")
    keys = cited_keys(new_tex)

    unknown = [k for k in keys if k not in primaries]
    if unknown:
        raise RefError(f"cited keys with no reference entry: {unknown}")

    lines = [
        "% Generated by make_refs.py. Do not edit by hand.",
        f"% Sources: {PRE_MD.name} ({sum(1 for e in entries if e.key.startswith('pre'))} entries), "
        f"{POST_MD.name} ({sum(1 for e in entries if e.key.startswith('post'))} entries).",
        "% Aliases (same normalized URL or title; citations rewritten to the primary key):",
    ]
    for alias, primary in sorted(alias_to_primary.items()):
        lines.append(f"%   {alias} -> {primary}")
    lines.append(r"\begin{thebibliography}{999}")
    for k in keys:
        lines.append(bibitem(primaries[k]))
    lines.append(r"\end{thebibliography}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    uncited = sorted(set(primaries) - set(keys), key=lambda k: (k[:3] != "pre", int(re.sub(r"\D", "", k))))
    print(f"unique works: {len(primaries)}; aliases: {len(alias_to_primary)} "
          f"({', '.join(f'{a}->{p}' for a, p in sorted(alias_to_primary.items()))})")
    print(f"cited and emitted: {len(keys)}; not cited (omitted): {len(uncited)}")
    return 0


if __name__ == "__main__":
    if "--dump" in sys.argv:
        ents = load_entries()
        prim, al = deduplicate(ents)
        for e in ents:
            print(bibitem(e))
        print("ALIASES", al)
        sys.exit(0)
    try:
        sys.exit(main())
    except RefError as exc:
        print(f"make_refs.py: error: {exc}", file=sys.stderr)
        sys.exit(1)
