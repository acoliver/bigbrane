"""Bounded public source construction. No experiment/provider API calls."""
import datetime, hashlib, json, pathlib, re, subprocess, urllib.request, urllib.error
ROOT=pathlib.Path('/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/heldout-acquisition/preparation/corpus-author/v2')
PRIVATE=ROOT/'private'; PRIVATE.mkdir(exist_ok=True)
def sha(b): return hashlib.sha256(b).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def get(index,kind,url):
 d=PRIVATE/f'candidate-{index:02}'; d.mkdir(exist_ok=True)
 log=d/'access.json'; rows=json.loads(log.read_text()) if log.exists() else []
 assert kind in ('lookup','body')
 assert sum(r['kind']==kind for r in rows)<(2 if kind=='lookup' else 1),'budget exhausted'
 assert url.startswith('https://') and not any(s in url for s in ['api.openalex','api.semanticscholar','export.arxiv','/api/'])
 rec={'kind':kind,'url':url,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'post-cutoff corpus construction, not query arm'}
 rows.append(rec); save(log,rows)
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public research source verification)'})
 try:
  with urllib.request.urlopen(req,timeout=45) as res:
   body=res.read(); rec.update(final_url=res.url,status=res.status,content_type=res.headers.get('Content-Type'),headers=dict(res.headers))
 except urllib.error.HTTPError as e:
  body=e.read(); rec.update(status=e.code,error=str(e),headers=dict(e.headers))
 except Exception as e:
  rec['error']=str(e); save(log,rows); print(index,kind,rec['error']); return
 path=d/f'{kind}-{sum(r["kind"]==kind for r in rows)}.bin'; path.write_bytes(body)
 rec.update(raw_path=str(path.relative_to(ROOT)),raw_sha256=sha(body),bytes=len(body)); save(log,rows)
 if kind=='body' and body.startswith(b'%PDF'):
  out=d/'extracted.txt'; subprocess.run(['pdftotext','-layout',str(path),str(out)],check=True)
  rec.update(text_path=str(out.relative_to(ROOT)),text_sha256=sha(out.read_bytes()),extraction='pdftotext -layout; text is not PDF bytes'); save(log,rows)
 print(index,kind,rec.get('status'),len(body),rec.get('final_url'))
 return body
if __name__=='__main__':
 import sys
 get(int(sys.argv[1]),sys.argv[2],sys.argv[3])
