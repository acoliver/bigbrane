"""Re-authenticate the durable projection using retained original local receipts."""
import hashlib
import json
from pathlib import Path
from hybrid.literature.seed import BUILD_LOG_PATH, SEED_PROVENANCE_PATH, compile_seed_provenance, load_seed

results = Path(__file__).resolve().parent
paths = {(r['lens'], r['query'], r['source']): Path(r['path'])
         for r in json.loads((results / 'receipt-paths.json').read_text())}
document = compile_seed_provenance(load_seed(), BUILD_LOG_PATH, BUILD_LOG_PATH.parent, receipt_paths=paths)
raw = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode() + b'\n'
assert raw == SEED_PROVENANCE_PATH.read_bytes()
print('Recompiled original receipts exactly:', len(raw), hashlib.sha256(raw).hexdigest())
