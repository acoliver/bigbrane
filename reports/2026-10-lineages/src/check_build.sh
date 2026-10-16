#!/bin/bash
# Post-build checks for lineages-brief: citations, boxes, glyphs, annotations, pages, style greps.
cd "$(dirname "$0")" || exit 1
TEX=lineages-brief.tex
echo "== TeX passes:"; grep -cE "Running TeX|Rerunning TeX" build.log
echo "== Undefined citations/references in final log:"; grep -cE "LaTeX Warning: (Citation|Reference) .* undefined" lineages-brief.log
grep -E "There were undefined" lineages-brief.log
echo "== Overfull boxes (pt):"; grep -oE "Overfull \\\\[hv]box \(([0-9.]+)pt" build.log | grep -oE "[0-9.]+" | sort -n | tail -5 | tr '\n' ' '; echo
echo "== Missing glyphs:"; grep -c "Missing character" build.log
echo "== Annotation outside page:"; grep -c "Annotation out of page" build.log
echo "== Literal tabs in tex:"; grep -c "$(printf '\t')" "$TEX"
echo "== Non-ASCII bytes in tex/references:"; LC_ALL=C grep -c '[^ -~]' "$TEX" references.tex
python3 - <<'EOF'
import re, zlib
tex = open("lineages-brief.tex").read()
body = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in tex.splitlines())
cites = {k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", body) for k in m.split(",")}
items = set(re.findall(r"\\bibitem\{([^}]*)\}", open("references.tex").read()))
print("== cite keys:", len(cites), " bibitems:", len(items),
      " cited-not-in-bib:", sorted(cites - items), " bib-not-cited:", sorted(items - cites))
labels = set(re.findall(r"\\label\{((?:tab|fig):[^}]*)\}", body))
refs = set(re.findall(r"\\ref\{((?:tab|fig):[^}]*)\}", body))
print("== float labels:", sorted(labels), " unreferenced:", sorted(labels - refs))
print("== captions:", body.count("\\caption{"), " floats:", body.count("\\begin{table}") + body.count("\\begin{figure}"))
d = open("lineages-brief.pdf", "rb").read(); best = 0
for m in re.finditer(rb"stream\r?\n", d):
    s = m.end(); e = d.find(b"endstream", s)
    try: raw = zlib.decompress(d[s:e])
    except Exception: continue
    for c in re.findall(rb"/Count\s+(\d+)", raw): best = max(best, int(c))
print(f"== PDF pages: {best}  bytes: {len(d)}")
EOF
