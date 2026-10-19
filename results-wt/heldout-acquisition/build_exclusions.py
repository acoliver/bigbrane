"""Copy only exclusion identities after checking the policy freeze. No policy design from labels."""
import json
from pathlib import Path
from harness import ROOT, POLICY, verify_freeze, file_sha, identities

verify_freeze()
origin = ROOT.parent / 'acquisition-windows' / 'DELIVERY-RECEIPTS.json'
entries, missing = [], []
for row in json.loads(origin.read_text())['registered_trace']:
    identity = {}
    if row.get('registered_title'):
        identity['title'] = row['registered_title']
    if row.get('registered_arxiv_id'):
        identity['arxiv_id'] = row['registered_arxiv_id']
    if identity:
        identities(identity)
        entries.append({'scope': 'original_pa2_11', 'identity': identity, 'origin_sha256': file_sha(origin)})
    else:
        missing.append(row['label'])
registry = {'scopes': {scope: {'status': 'pending', 'inventory_sha256': None,
    'reason': 'Full prior-study inventory and alias audit not assembled in this run.'}
    for scope in POLICY['required_exclusion_scopes']}, 'entries': entries,
    'unknown_aliases': ['Unresolved cross-provider aliases remain.',
                        'Original registrations without usable metadata: ' + ', '.join(missing)]}
(ROOT / 'exclusions.json').write_text(json.dumps(registry, indent=2, sort_keys=True) + '\n')
(ROOT / 'exclusions.sha256').write_text(file_sha(ROOT / 'exclusions.json') + '  exclusions.json\n')
(ROOT / 'corpus.json').write_text(json.dumps({'status': 'PENDING', 'reason':
    'Independent corpus, blinded label vault and full exclusion inventories have not been assembled.',
    'cards': [], 'vault_sha256': None}, indent=2) + '\n')
print(json.dumps({'copied_registration_identities': len(entries), 'unresolved_registrations': missing,
                  'all_inventories': 'PENDING', 'exclusions_sha256': file_sha(ROOT / 'exclusions.json')}, indent=2))
