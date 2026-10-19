"""Preserve independent publication bodies locally; tests use metadata only."""
from datetime import datetime, timezone
import hashlib, json, subprocess, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'results-wt/evidence-recovery'
DEST=REPORT/'nonlinear-sources'
DEST.mkdir(exist_ok=True)
sources=[('dpmeans','https://arxiv.org/pdf/1111.0352'),('art-survey','https://arxiv.org/pdf/1905.11437'),('fuzzyart','https://open.bu.edu/bitstream/handle/2144/2069/91.011.pdf?sequence=1')]
records=json.loads((REPORT/'nonlinear-publications-local.json').read_text())
for name,url in sources:
    try:
        with urllib.request.urlopen(url,timeout=60) as response:
            raw=response.read();final=response.url
        assert raw.startswith(b'%PDF'),name
        file=DEST/(name+'.pdf');file.write_bytes(raw)
        text=file.with_suffix('.txt')
        subprocess.run(['pdftotext','-layout',str(file),str(text)],check=True)
        records.append(dict(name=name,url=url,final_url=final,retrieved_utc=datetime.now(timezone.utc).isoformat(),file=str(file.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),text_file=str(text.relative_to(ROOT)),text_sha256=hashlib.sha256(text.read_bytes()).hexdigest()))
        print(name,len(raw),flush=True)
    except Exception as e:
        print(name,type(e).__name__,str(e),flush=True)
(REPORT/'nonlinear-publications-local.json').write_text(json.dumps(records,indent=2)+'\n')
