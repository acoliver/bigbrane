"""Offline projection of used-study identities, with separate logical and paper counts."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import tomllib

HERE = Path(__file__).resolve().parent
HELD = HERE.parents[1]
SIBLING = HELD.parents[1]
MAIN = SIBLING.with_name('bigbrane')
sys.path.insert(0, str(HELD))
from harness import identities, norm_title, digest, Pool, corpus_gate, verify_freeze

SCOPES = ('original_pa2_11', 'historical_recovery_19', 'seen_seed_109', 'other_used_study_data')
PASSES = {'card-46dbb349', 'card-c9615069'}
HISTORICAL = set('1184f7a7 14429dbc 2c5dd003 36725a7c 537e161c 582f8abf 68612a2a 6de11b01 6fc7efec 7ec5e5c2 8dc4db49 a36f1a3a bc3935ac d1d383d5 d86dca4b dc69816b ddb071bd ff8f801a ffdbe331'.split())
HISTORICAL = {'card-' + x for x in HISTORICAL}
FROZEN = ('PLAN.md', 'policy.json', 'PREREGISTRATION.sha256', 'exclusions.json', 'exclusions.sha256', 'corpus.json', 'harness.py', 'offline.py', 'SOURCE-HASHES.json')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def bare_identity(value):
    """Only established public identifier grammars; arbitrary source IDs stay unresolved."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if re.fullmatch(r'(?:https://openalex.org/)?W\d+', v, re.I):
        return {'source': 'openalex', 'source_id': re.sub(r'^https://openalex.org/', '', v, flags=re.I)}
    # An unqualified hex hash has no trustworthy provider attribution.
    if re.match(r'^(?:https?://(?:www\.)?arxiv.org/(?:abs|pdf)/|arxiv:|\d{4}\.\d{4,5}(?:v\d+)?$|[a-z-]+(?:\.[a-z]{2})?/\d{7})', v, re.I):
        try:
            identities({'arxiv_id': v})
            return {'arxiv_id': v}
        except ValueError:
            return None
    if re.match(r'^(?:https?://(?:dx\.)?doi.org/|doi:|10\.\d{4,9}/)', v, re.I):
        try:
            identities({'doi': v})
            return {'doi': v}
        except ValueError:
            return None
    if ':' in v:
        src, sid = v.split(':', 1)
        if src in {'openalex', 's2', 'semantic_scholar', 'arxiv'}:
            try:
                item = {'source': src, 'source_id': sid}
                identities(item)
                return item
            except ValueError:
                return None
    return None


def paper_identity(item, allow_title=True):
    if not isinstance(item, dict):
        return None
    result = {}
    if allow_title and isinstance(item.get('title'), str) and item['title'].strip():
        result['title'] = item['title']
    aliases = []
    for key in ('arxiv_id', 'doi'):
        if item.get(key):
            alias = {key: item[key]}
            try:
                identities(alias)
                aliases.append(alias)
            except (ValueError, TypeError):
                pass
    if item.get('source') and item.get('source_id'):
        try:
            alias = {'source': item['source'], 'source_id': item['source_id']}
            identities(alias)
            aliases.append(alias)
        except (ValueError, KeyError):
            pass
    else:
        for key in ('source_id', 'key', 'url', 'id'):
            if alias := bare_identity(item.get(key)):
                aliases.append(alias)
    # URL aliases are recorded only when the URL itself identifies a paper.
    if alias := bare_identity(item.get('url')):
        aliases.append(alias)
    if aliases:
        result['aliases'] = list({digest(a): a for a in aliases}.values())
    if result:
        identities(result)
        return result
    return None


class Sources:
    def __init__(self):
        self.records = {}

    def remember(self, path, role, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.records:
            self.records[key] = {'sha256': file_sha(path), 'bytes': path.stat().st_size, 'roles': []}
        r = self.records[key]
        if role not in r['roles']:
            r['roles'].append(role)
        if expected:
            require(r['sha256'] == expected, 'source SHA mismatch: ' + key)
        return r['sha256']

    def json(self, path, role):
        self.remember(path, role)
        return read_json(path)


def check_preserved(manifest, sources):
    for path, expected in manifest.items():
        expected = expected['sha256'] if isinstance(expected, dict) else expected
        sources.remember(path, 'preserved original input', expected)


def inspect_reference_fields(obj, add, unresolved, origin, cid=None):
    """Select bibliography fields only, never interpret authors/year as identifiers."""
    if isinstance(obj, list):
        for row in obj:
            inspect_reference_fields(row, add, unresolved, origin, cid)
        return
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        if key in {'references', 'evidence', 'candidates', 'citations', 'source_ids', 'papers', 'known_relatives', 'transplant_sources'}:
            for row in value if isinstance(value, list) else [value]:
                item = paper_identity(row) if isinstance(row, dict) else bare_identity(row)
                if item:
                    add(item, origin, cid, key)
                elif row:
                    descriptor = row.get('name') or row.get('source_id') if isinstance(row, dict) else str(row)
                    if descriptor:
                        unresolved.append({'source': origin, 'cid': cid, 'field': key,
                                           'descriptor': str(descriptor)[:500], 'reason': 'no unambiguous structured paper identity'})
                if isinstance(row, dict):
                    inspect_reference_fields(row, add, unresolved, origin, cid)
        elif isinstance(value, (dict, list)) and key not in {'status', 'episodes', 'reference_samples', 'files', 'states', 'inputs', 'card_outputs'}:
            inspect_reference_fields(value, add, unresolved, origin, cid)
        elif key in {'citation', 'publication'} and isinstance(value, dict):
            if item := paper_identity(value):
                add(item, origin, cid, key)
    # Explicit paper IDs in critic prose are allowed; quoted names are not inferred titles.
    for key in ('mapping', 'why', 'citation', 'cite', 'reference', 'textbook_why'):
        value = obj.get(key)
        if not isinstance(value, str):
            continue
        if key in {'citation','cite','reference'} and value.strip() and bare_identity(value) is None:
            unresolved.append({'source':origin,'cid':cid,'field':key,'descriptor':value[:500],
                               'reason':'citation prose requires source-qualified exact publication identity'})
        for token in re.findall(r'(?:https?://(?:arxiv\.org/(?:abs|pdf)/|(?:dx\.)?doi\.org/)\S+|arxiv:\s*\d{4}\.\d{4,5}(?:v\d+)?|10\.\d{4,9}/[^\s<>\"]+)', value, re.I):
            token = token.rstrip('.,;)')
            if item := bare_identity(token.replace('arxiv: ', 'arxiv:')):
                add(item, origin, cid, 'explicit critic identifier')


def discover():
    """Enumerate paths without opening raw model calls, response bodies or giant logs."""
    roots, selected, skipped = [], set(), []
    for repo in (MAIN, SIBLING):
        for base in ('runs', 'results', 'results-wt', 'hybrid/tests/research/fixtures', 'hybrid/research', 'hybrid/tests/research', 'tests', 'var', 'data'):
            parent = repo / base
            if not parent.exists():
                roots.append({'path': str(parent), 'status': 'absent', 'reason': 'not present in this checkout'})
                continue
            units = [parent] if base.startswith('hybrid/') or base == 'tests' else sorted(p for p in parent.iterdir() if p.is_dir())
            for unit in units:
                if unit == HELD or HELD in unit.parents:
                    roots.append({'path': str(unit), 'status': 'excluded', 'reason': 'current held-out preparation; no acquired corpus'})
                    continue
                reason = None
                # Test executions can manufacture cards; retained fixtures are scanned separately.
                if any(x in unit.name.lower() for x in ('pytest', 'bench-tests', 'verification', 'review-fixes', 'followup-', 'pilot-', 'confirmatory-')):
                    reason = 'generated verification/benchmark artifact root; not authenticated discovery evidence; coverage unresolved'
                if reason:
                    roots.append({'path': str(unit), 'status': 'excluded_pending', 'reason': reason})
                    continue
                counts = Counter()
                for folder, dirs, names in os.walk(unit):
                    dirs[:] = sorted(d for d in dirs if d not in {'fixtures' if base == 'hybrid/tests/research' else '__never__', 'llm', 'cache', 'raw-cache', 'diagnostic-cache', 'postfix-cache', '__pycache__', '.git', 'node_modules', 'frozen', 'jobs', 'stores', 'venv', '.venv'}
                                     and 'pytest' not in d and 'test-work' not in d and 'checkpoint' not in d and 'isolated' not in d)
                    f = Path(folder)
                    for name in sorted(names):
                        p = f / name
                        if name == 'ledger.jsonl':
                            selected.add(p); counts['ledgers'] += 1
                        elif name.endswith('.json') and ('cards' in p.parts or any(x in p.parts for x in ('prior_art', 'reduction_critic', 'source_reduction_critic')) or '/fixtures/' in str(p)):
                            if p.stat().st_size <= 2_000_000:
                                selected.add(p); counts['metadata'] += 1
                            else:
                                skipped.append({'path': str(p), 'reason': 'large numeric fixture; source/publication metadata read separately'})
                        elif name.endswith('.py') and (base == 'tests' or base.startswith('hybrid/')):
                            selected.add(p); counts['python_fixtures'] += 1
                roots.append({'path': str(unit), 'status': 'enumerated', 'selected_files': dict(counts),
                              'reason': 'path scan for cards, bibliography artifacts, ledgers and retained fixtures; raw model calls/jobs/caches excluded'})
    return roots, sorted(selected), skipped


def build():
    verify_freeze()
    sources = Sources()
    frozen_before = {str(HELD / n): sources.remember(HELD / n, 'frozen base') for n in FROZEN}
    replay_path = SIBLING / 'results-wt/retrieval-diagnosis/PRESERVATION.json'
    seed_path = SIBLING / 'results-wt/seed-provenance/raw-baseline.json'
    replay_baseline = sources.json(replay_path, '274-original baseline')['saved_inputs_sha256']
    seed_baseline = sources.json(seed_path, '75-seed baseline')
    require(len(replay_baseline) == 274 and len(seed_baseline) == 75, 'unexpected original baseline size')
    check_preserved(replay_baseline, sources)
    check_preserved(seed_baseline, sources)
    write('preservation-before.json', {'frozen': frozen_before, 'replay_inputs': 274, 'seed_inputs': 75,
                                      'verified_unique_inputs': len(set(replay_baseline) | set(seed_baseline)),
                                      'source_baselines': [str(replay_path), str(seed_path)]})
    print('Original baselines verified', flush=True)
    roots, paths, skipped = discover()
    print('Roots enumerated', len(roots), 'selected paths', len(paths), flush=True)
    boundaries=[]
    for root in (MAIN,SIBLING):
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if child.name in {'runs','results','results-wt','hybrid','tests','var','data'}:
                status='bounded_selection'; reason='Only metadata and ledger selection recorded in root manifest; not every file is read.'
            elif child.name.startswith('.') or child.name.endswith('.egg-info'):
                status='excluded'; reason='Repository/tool/dependency state; raw configuration not read.'
            else:
                status='excluded_pending'; reason='Outside metadata selection; possible prior-study coverage unresolved.'
            boundaries.append({'path':str(child),'status':status,'reason':reason})
    write('TOP-LEVEL-BOUNDARIES.v1.json',{'roots':boundaries,
        'external_roots':'Only two Git-registered checkouts are bounded here. Archives, deleted files and unregistered external study roots are unknown.'})
    write('root-manifest.v1.json', {'version': 1, 'roots': roots, 'selected_paths': [str(p) for p in paths],
          'excluded_large_numeric': skipped, 'unknown_roots': ['unlisted external studies/checkouts/archives',
          'excluded generated experiment roots not authenticated as discovery data'], 'status': 'PENDING broader scope'})
    inventories = {s: {'version': 1, 'scope': s, 'mechanisms': [], 'papers': [], 'unresolved': []} for s in SCOPES}
    used = inventories[SCOPES[3]]
    paper_index = {s:{} for s in SCOPES}

    def add_paper(scope, item, path, cid=None, relationship='known used paper'):
        ids = sorted(identities(item))
        key = digest(item)
        papers = inventories[scope]['papers']
        hit = paper_index[scope].get(key)
        evidence = {'path': str(path), 'sha256': sources.records[str(Path(path).resolve())]['sha256'],
                    'cid': cid, 'relationship': relationship}
        if hit is None:
            hit = {'record_id': key, 'identity': item, 'normalized_aliases': ids, 'evidence': []}
            papers.append(hit)
            paper_index[scope][key] = hit
        if evidence not in hit['evidence']:
            hit['evidence'].append(evidence)
        return key

    def add_used(item, origin, cid, relationship):
        return add_paper(SCOPES[3], item, origin, cid, relationship)

    # Keep all eleven registrations even if later alias unions connect their papers.
    original = sources.json(HELD / 'exclusions.json', 'frozen PA2 aliases')['entries']
    delivery = SIBLING / 'results-wt/acquisition-windows/DELIVERY-RECEIPTS.json'
    registrations = sources.json(delivery, 'original PA2 registration trace')['registered_trace']
    require(len(original) == len(registrations) == 11, 'need 11 logical PA2 registrations')
    for i, (entry, row) in enumerate(zip(original, registrations), 1):
        require(entry['identity'].get('title') == row['registered_title'], 'PA2 registration order/identity changed')
        ref = add_paper(SCOPES[0], entry['identity'], delivery, row['card_id'], 'original registration')
        inventories[SCOPES[0]]['mechanisms'].append({'registration_id': f'pa2-{i:02d}', 'label': row['label'],
                    'cid': row['card_id'], 'paper_record_id': ref})

    provenance_path = SIBLING / 'hybrid/literature/seed_provenance.json'
    seed_toml = SIBLING / 'hybrid/literature/seed_reading_list.toml'
    seed_evidence_path=SIBLING/'results-wt/seed-provenance/EVIDENCE.json'
    seed_evidence=sources.json(seed_evidence_path,'existing seed provenance inventory')
    sources.remember(provenance_path,'seed provenance pinned by prior evidence',
                     seed_evidence['files']['hybrid/literature/seed_provenance.json']['sha256'])
    projection = sources.json(provenance_path, 'authenticated seed projection')
    require(digest(projection['payload']) == projection['payload_sha256'], 'seed projection payload mismatch')
    sources.remember(seed_toml, 'original seed TOML')
    sources.remember(MAIN / 'hybrid/literature/seed_reading_list.toml', 'main original seed TOML', file_sha(seed_toml))
    seeds = tomllib.loads(seed_toml.read_text())['papers']
    require(len(seeds) == 109 and len({(p['source'], p['source_id']) for p in seeds}) == 109, 'need 109 unique seed entries')
    seen = {(p['source'], p['source_id']): (p, row) for row in projection['payload']['rows']
            for papers in row['per_source'].values() for p in papers}
    for seed in seeds:
        key = (seed['source'], seed['source_id'])
        require(key in seen, 'seed lacks provenance: ' + str(key))
        paper, row = seen[key]
        for k in ('title','authors','year','abstract','url','venue','citation_count','fetched_at','doi','arxiv_id','source','source_id'):
            require(seed.get(k) == paper.get(k), 'seed/projection disagreement: ' + k)
        item = paper_identity(paper)
        ref = add_paper(SCOPES[2], item, provenance_path, relationship='actual seed source paper')
        inventories[SCOPES[2]]['mechanisms'].append({'mechanism_id': seed['source'] + ':' + seed['source_id'],
            'paper_record_id': ref, 'identity': item, 'source_paper':{k:paper.get(k) for k in ('title','arxiv_id','doi','source','source_id','url')}, 'lens': seed['lens'], 'semantic_label': seed['relevance'],
            'normalized_semantic_label': norm_title(seed['relevance']), 'origin_line': row['origin']['line'],
            'raw_row_sha256': row['origin']['row_sha256'], 'raw_log_sha256': projection['payload']['log_sha256'],
            'receipts': {src: {k: v for k, v in receipt.items() if k.endswith('sha256')}
                         for src, receipt in row['origin']['receipts'].items()}})

    historic_path = SIBLING / 'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json'
    historical = sources.json(historic_path, '21 original historical card/critic rows')['cards']
    require({e['card_id'] for e in historical} == HISTORICAL | PASSES, 'historical CID set mismatch')
    historical_rows = {}
    for e in historical:
        cid = e['card_id']; scope = SCOPES[3] if cid in PASSES else SCOPES[1]
        card = e['card']; critic = e['critic']; label = critic.get('mapping') or card.get('core_idea') or card['title']
        mech = {'mechanism_id': cid, 'cid': cid, 'identity': {'title': card['title']},
                'bound_card_sha256': digest(card), 'bound_critic_sha256': digest(critic),
                'semantic_label': label[:1200], 'normalized_semantic_label': norm_title(label[:1200]),
                'source_path': str(historic_path), 'source_sha256': sources.records[str(historic_path)]['sha256'],
                'role': 'used historical pass' if cid in PASSES else 'historical recovery mechanism',
                'closest_known_paper_records': [], 'card_versions': []}
        inventories[scope]['mechanisms'].append(mech); historical_rows[cid] = mech
        for ev in e['prior_art'].get('evidence', []):
            if item := paper_identity(ev):
                ref = add_paper(scope, item, historic_path, cid, ev.get('relation', 'recorded prior art'))
                mech['closest_known_paper_records'].append(ref)
        inspect_reference_fields(card, lambda i,p,c,k: add_paper(scope,i,p,c,k), inventories[scope]['unresolved'], str(historic_path), cid)
        inspect_reference_fields(critic, lambda i,p,c,k: add_paper(scope,i,p,c,k), inventories[scope]['unresolved'], str(historic_path), cid)
        if not mech['closest_known_paper_records']:
            inventories[scope]['unresolved'].append({'cid': cid, 'reason': 'no structured closest-source paper metadata in original prior-art row'})

    print('Named scopes projected', flush=True)
    # Publication manifests carry public source URLs and hashes, never labels for future cards.
    title_path = HERE / 'source-title-bindings.v1.json'
    title_bindings = sources.json(title_path, 'source-qualified publication title spans')['bindings']
    for binding in title_bindings:
        sources.remember(binding['path'], 'original publication title source', binding['sha256'])
        lines = Path(binding['path']).read_text().splitlines()
        span = '\n'.join(lines[binding['line_start']-1:binding['line_end']])
        if Path(binding['path']).suffix == '.html':
            span = html.unescape(re.sub('<[^>]+>', '', span))
        require(norm_title(binding['title']) in norm_title(span), 'title is not in pinned source span')
    prior_manifests = []
    for path in [SIBLING / 'results-wt/retrieval-diagnosis/FILEMANIFEST.json',
                 SIBLING / 'results-wt/evidence-recovery/sources/provenance.json',
                 SIBLING / 'results-wt/evidence-recovery/MANIFEST.json',
                 SIBLING / 'results-wt/seed-provenance/EVIDENCE.json']:
        document = sources.json(path, 'existing source inventory')
        prior_manifests.append({'path':str(path), 'sha256':sources.records[str(path)]['sha256'],
                                'record_count':len(document.get('files', document)) if isinstance(document,dict) else len(document)})
    write('prior-source-inventories.v1.json', prior_manifests)
    fixture_root = SIBLING / 'hybrid/tests/research/fixtures/reference_replays'
    publications = []
    for path in sorted(fixture_root.glob('*publications.json')):
        rows = sources.json(path, 'source-qualified recovery bibliography')
        for row in rows:
            record = {k: row[k] for k in ('id', 'name', 'title', 'url', 'file', 'sha256', 'text_file', 'text_sha256', 'section') if k in row}
            publications.append(dict(record, manifest=str(path)))
            candidates = [SIBLING / row['file']]
            if '/' not in row['file']:
                candidates += [SIBLING / 'results-wt/evidence-recovery' / folder / row['file'] for folder in ('sources','next-sources','remaining-sources','composites-sources','nonlinear-sources')]
            local = next((p for p in candidates if p.exists()), None)
            if local:
                sources.remember(local, 'original recovery publication', row['sha256'])
                text_path = SIBLING / row['text_file'] if row.get('text_file') else local.with_suffix('.txt')
                binding = next((b for b in title_bindings if Path(b['path']) in {local, text_path}), None)
                item = paper_identity(row)
                if binding:
                    if item is None:
                        item = {'title':binding['title']}
                    else:
                        item.setdefault('aliases', []).append({'title':binding['title']})
                    if text_path.exists():
                        sources.remember(text_path,'recovery publication text',row.get('text_sha256'))
                        header = '\n'.join(text_path.read_text().splitlines()[:20])
                        for token in re.findall(r'10\.\d{4,9}/[A-Za-z0-9._/;()\-]+',header):
                            if alias := bare_identity(token.rstrip('.,;)')):
                                item.setdefault('aliases',[]).append(alias)
                if item:
                    add_used(item, str(path), None, 'source-qualified published recovery source')
                else:
                    used['unresolved'].append({'source':str(path),'descriptor':row['url'], 'reason':'publication has no supported paper ID or verified title span'})
            else:
                used['unresolved'].append({'source': str(path), 'descriptor': row['file'], 'reason': 'publication body not found locally'})
    write('publication-sources.v1.json', publications)
    for e in historical:
        cid=e['card_id']; scope=SCOPES[3] if cid in PASSES else SCOPES[1]
        mapping=norm_title(e['critic'].get('mapping') or '')
        for binding in title_bindings:
            title=norm_title(binding['title'])
            if len(title)>=20 and title in mapping:
                ref=add_paper(scope,{'title':binding['title']},binding['path'],cid,
                              'source-qualified title explicitly named in original reduction mapping; no equivalence claim')
                historical_rows[cid]['closest_known_paper_records'].append(ref)


    print('Publication sources verified', len(publications), flush=True)
    card_versions = defaultdict(list)
    ledger_summaries = []
    all_mechanisms = {}
    # Selected ledgers are streamed exactly once for attribution. No full ledger is shipped.
    for path in paths:
        if path.name != 'ledger.jsonl':
            continue
        h = hashlib.sha256(); n = 0; counts = Counter(); chosen = []
        with path.open('rb') as f:
            for n, raw in enumerate(f, 1):
                h.update(raw)
                try:
                    e = json.loads(raw)
                except (ValueError, UnicodeError):
                    used['unresolved'].append({'source': str(path), 'line': n, 'reason': 'malformed ledger row'})
                    continue
                typ = e.get('type'); counts[typ] += 1; data = e.get('data', {})
                cid = data.get('card_id')
                if typ in {'card_version', 'card_created', 'card_revised'} and cid:
                    chosen.append({'line': n, 'cid': cid, 'type': typ,
                                   **{k: data[k] for k in ('path', 'sha256', 'version') if k in data}})
                if typ == 'artifact' and data.get('kind') in {'prior_art', 'reduction_critic', 'source_reduction_critic'} and data.get('path'):
                    candidate = path.parent / data['path']
                    if candidate.exists():
                        sources.remember(candidate, 'ledger-bound bibliography/critic', data.get('sha256'))
                        paths.append(candidate) if candidate not in paths else None
        sources.records[str(path)] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size, 'roles': ['single-pass ledger attribution']}
        ledger_summaries.append({'path': str(path), 'sha256': h.hexdigest(), 'rows': n, 'events': dict(counts), 'card_events': chosen})
    print('Ledgers streamed', len(ledger_summaries), flush=True)
    # Freeze a set after ledger-discovered artifact paths have been added.
    for path in sorted(set(paths)):
        if path.name == 'ledger.jsonl':
            continue
        if path.suffix == '.py':
            sources.remember(path, 'research fixture/test source')
            text = path.read_text(errors='replace')
            # Only explicit identifiers, not invented bibliographic interpretation of test prose.
            for token in re.findall(r'https?://(?:arxiv\.org/(?:abs|pdf)/|(?:dx\.)?doi\.org/)[^\s\"\'<>]+', text):
                if item := bare_identity(token.rstrip(').,;')):
                    add_used(item, str(path), None, 'explicit fixture citation')
            continue
        sources.remember(path, 'used card/fixture/critic metadata')
        try:
            obj = read_json(path)
        except (ValueError, UnicodeError):
            used['unresolved'].append({'source': str(path), 'reason': 'metadata file is not valid JSON'})
            continue
        def visit(value):
            if isinstance(value, list):
                for v in value:
                    visit(v)
            elif isinstance(value, dict):
                cid = value.get('id') or value.get('card_id')
                if isinstance(cid, str) and re.fullmatch(r'card-[0-9a-f]{8}', cid) and 'title' in value and ('mechanism' in value or 'core_idea' in value):
                    binding = {'path': str(path), 'sha256': sources.records[str(path)]['sha256'],
                               'card_payload_sha256': digest({k:v for k,v in value.items() if k != 'status'}), 'version': value.get('version'), 'title':value['title']}
                    card_versions[cid].append(binding)
                    key = (cid, binding['card_payload_sha256'])
                    if key not in all_mechanisms:
                        all_mechanisms[key] = {'mechanism_id': cid, 'identity': {'title': value['title']},
                            'semantic_label': value.get('core_idea') or value.get('summary', ''),
                            'normalized_semantic_label': norm_title(value.get('core_idea') or value.get('summary', '')),
                            'bound_card_sha256': binding['card_payload_sha256'], 'source': binding}
                for k, v in value.items():
                    if isinstance(v, (dict,list)) and k not in {'episodes', 'reference_samples', 'files', 'states', 'inputs', 'card_outputs'}:
                        visit(v)
        visit(obj)
        inspect_reference_fields(obj, add_used, used['unresolved'], str(path), obj.get('id') if isinstance(obj,dict) else None)
    print('Card metadata projected', len(all_mechanisms), flush=True)
    for cid, mech in historical_rows.items():
        mech['card_versions'] = sorted(card_versions[cid], key=lambda v: (v['path'], v['sha256']))
        titles = sorted({v['title'] for v in mech['card_versions']})
        mech['identity']['aliases'] = [{'title':t} for t in titles if t != mech['identity']['title']]
        mech['exact_bound_version_matches'] = sum(v['card_payload_sha256']==mech['bound_card_sha256'] for v in mech['card_versions'])
        if not mech['card_versions']:
            inventories[SCOPES[1] if cid not in PASSES else SCOPES[3]]['unresolved'].append({'cid':cid,'reason':'original card file variants not found'})
    grouped = defaultdict(list)
    for (cid,_),record in sorted(all_mechanisms.items()):
        if cid not in HISTORICAL | PASSES:
            grouped[cid].append(record)
    for cid, versions in sorted(grouped.items()):
        first = versions[0]
        label = first['semantic_label'][:600]
        titles = sorted({r['identity']['title'] for r in versions})
        used['mechanisms'].append({'mechanism_id':cid,'identity':{'title':titles[0], 'aliases':[{'title':t} for t in titles[1:]]},
            'semantic_label':label,'normalized_semantic_label':norm_title(label),
            'bound_card_sha256':first['bound_card_sha256'],'source':first['source'],
            'card_versions':card_versions[cid]})
    write('ledger-discovery.v1.json', ledger_summaries)

    registry = {'entries': [], 'scopes': {}, 'unknown_aliases': []}
    for scope, inventory in inventories.items():
        inventory['papers'].sort(key=lambda p: p['record_id'])
        unknowns = {}
        for u in inventory['unresolved']:
            record = {k:v for k,v in u.items() if k not in {'source','line'}}
            key = digest(record)
            if key not in unknowns:
                unknowns[key] = dict(record, unresolved_id=key, sources=[], occurrence_count=0)
            unknowns[key]['occurrence_count'] += 1
            if u.get('source') and u['source'] not in unknowns[key]['sources']:
                unknowns[key]['sources'].append(u['source'])
        inventory['unresolved'] = sorted(unknowns.values(),key=lambda u:u['unresolved_id'])
        entries = []
        # Logical registrations already represented by their paper records; mechanisms get their own title identity.
        if scope != SCOPES[0]:
            for mech in inventory['mechanisms']:
                entries.append({'scope':scope,'identity':mech['identity'],
                                'origin_sha256':mech.get('source_sha256') or (mech.get('source') or {}).get('sha256') or sources.records[str(provenance_path)]['sha256']})
        for paper in inventory['papers']:
            entries.append({'scope':scope, 'identity':paper['identity'], 'origin_sha256':paper['evidence'][0]['sha256']})
        entries = list({digest(e):e for e in entries}.values())
        pool = Pool()
        for e in entries:
            pool.add(e['identity'])
        logical_ok = len(inventory['mechanisms']) == {'original_pa2_11':11,'historical_recovery_19':19,'seen_seed_109':109}.get(scope,len(inventory['mechanisms']))
        # Bibliographic uncertainty remains separate from enumeration completeness.
        complete = logical_ok and scope != SCOPES[3]
        if scope == SCOPES[1]:
            complete = complete and all(m['card_versions'] and m['closest_known_paper_records'] for m in inventory['mechanisms'])
        minimum = dict(zip(SCOPES,[11,19,109,1]))[scope]
        complete = complete and len(pool.items) >= minimum
        inventory['status'] = 'complete' if complete else 'pending'
        inventory['logical_count'] = len(inventory['mechanisms'])
        inventory['known_alias_count'] = len(set().union(*(identities(e['identity']) for e in entries))) if entries else 0
        inventory['identity_union_count'] = len(pool.items)
        inventory['unknown_alias_count'] = len(inventory['unresolved'])
        inventory['coverage_note'] = 'Complete named logical enumeration, conditional on recorded aliases; bibliography gaps retained.' if complete else 'PENDING: broader root coverage or named provenance gaps remain.'
        write(scope + '.v1.json', inventory)
        registry['entries'].extend(entries)
        registry['scopes'][scope] = {'status': inventory['status'], 'inventory_sha256':digest(entries), 'reason':inventory['coverage_note']}
        registry['unknown_aliases'].extend(scope + '.v1.json#unresolved/' + u['unresolved_id'] for u in inventory['unresolved'])
    registry['unknown_aliases'].append('Unknown external study roots and excluded generated experiment roots prevent full other-used scope closure. Title-only matching cannot establish semantic independence.')
    write('exclusions.v1.json', registry)
    (HERE/'exclusions.v1.sha256').write_text(file_sha(HERE/'exclusions.v1.json')+'  exclusions.v1.json\n')
    pool = Pool()
    for e in registry['entries']:
        pool.add(e['identity'])
    write('alias-unions.v1.json', {'method':'frozen harness Pool; shared exact normalized identifiers, including title equality, are identity assumptions',
        'identities': [{'union_id':k,'aliases':sorted(v['ids'])} for k,v in pool.items.items()],
        'scope_unions': {s:len({pool.owners[a] for e in registry['entries'] if e['scope']==s for a in identities(e['identity'])}) for s in SCOPES},
        'unique_alias_count':len(pool.owners), 'union_count':len(pool.items),
        'title_only_unions':sum(all(a.startswith('title:') for a in v['ids']) for v in pool.items.values())})
    paper_pool=Pool(); mechanism_pool=Pool()
    for inv in inventories.values():
        for paper in inv['papers']:
            paper_pool.add(paper['identity'])
        for mechanism in inv['mechanisms']:
            if mechanism.get('identity'):
                mechanism_pool.add(mechanism['identity'])
    coverage={'paper_identity_unions':len(paper_pool.items),'mechanism_identity_unions':len(mechanism_pool.items),
        'scope_alias_sum':sum(inv['known_alias_count'] for inv in inventories.values()),
        'cross_scope_alias_union':len(pool.owners),'title_only_global_unions':sum(all(a.startswith('title:') for a in v['ids']) for v in pool.items.values()),
        'root_status_counts':dict(Counter(r['status'] for r in roots)),
        'scope_paper_coverage':{scope:{'records':len(inv['papers']),
            'with_title':sum(any(a.startswith('title:') for a in p['normalized_aliases']) for p in inv['papers']),
            'with_arxiv':sum(any(a.startswith('arxiv:') for a in p['normalized_aliases']) for p in inv['papers']),
            'with_doi':sum(any(a.startswith('doi:') for a in p['normalized_aliases']) for p in inv['papers']),
            'with_provider_id':sum(any(a.startswith(('s2:','openalex:','arxiv:')) for a in p['normalized_aliases']) for p in inv['papers']),
            'unresolved_by_field':dict(Counter(u.get('field','provenance') for u in inv['unresolved']))} for scope,inv in inventories.items()},
        'limits':['No future corpus or labels exist. Normalized title equality is an identity assumption, not a semantic novelty judgment.',
                  'Enumeration covers named local roots only. Excluded roots and unknown external roots keep other-used scope PENDING.']}
    write('COVERAGE.v1.json',coverage)
    # Actual bytes are rechecked after all inventory extraction, including the 274 and 75 original inputs.
    unchanged = {p: file_sha(p) == expected for p,expected in frozen_before.items()}
    for baseline in (replay_baseline,seed_baseline):
        for p,expected in baseline.items():
            expected = expected['sha256'] if isinstance(expected,dict) else expected
            unchanged[p] = file_sha(p) == expected
    require(all(unchanged.values()), 'original evidence changed during build')
    write('preservation-after.json', {'all_unchanged':all(unchanged.values()),'checks':unchanged,'replay_inputs':274,'seed_inputs':75})
    write('source-hashes.v1.json', {'files': sources.records, 'duplicate_hashes': duplicate_hashes(sources.records),
          'note':'Hashes only for raw inputs/model calls listed by prior baseline. No raw bodies or secrets are copied.'})
    names = [s+'.v1.json' for s in SCOPES] + ['exclusions.v1.json','exclusions.v1.sha256','root-manifest.v1.json',
             'ledger-discovery.v1.json','source-hashes.v1.json','publication-sources.v1.json','alias-unions.v1.json',
             'preservation-before.json','preservation-after.json','build_inventory.py', 'source-title-bindings.v1.json',
             'prior-source-inventories.v1.json','TOP-LEVEL-BOUNDARIES.v1.json','COVERAGE.v1.json','test_inventory.py','run_tests.py','pytest.ini']
    stage = {'version':1,'stage':'known-used-data inventory preparation', 'policy_amendment':False,
             'frozen_base':frozen_before,'artifacts':{n:file_sha(HERE/n) for n in names},
             'source_baselines':{str(replay_path):file_sha(replay_path),str(seed_path):file_sha(seed_path)},
             'corpus_selected':False,'labels_selected':False,'source_reauthentication':'local original hashes verified; compact tests do not reconstruct response bodies'}
    write('STAGE.v1.json',stage)
    (HERE/'STAGE.v1.sha256').write_text(file_sha(HERE/'STAGE.v1.json')+'  STAGE.v1.json\n')
    result = {'counts':{s:{k:inventories[s][k] for k in ('status','logical_count','known_alias_count','identity_union_count','unknown_alias_count')} for s in SCOPES},
              'roots':len(roots),'selected_files':len(set(paths)),'ledgers_streamed':len(ledger_summaries),
              'source_files':len(sources.records),'unique_aliases':len(pool.owners),'union_count':len(pool.items),
              'corpus_gate':corpus_gate(read_json(HELD/'corpus.json'),registry),
              'stage_sha256':file_sha(HERE/'STAGE.v1.json'),'preservation':all(unchanged.values())}
    write('BUILD-RECEIPT.v1.json',result)
    print(json.dumps(result,indent=2))


def duplicate_hashes(records):
    groups = defaultdict(list)
    for p,row in records.items():
        groups[row['sha256']].append(p)
    return {h:paths for h,paths in sorted(groups.items()) if len(paths)>1}


def validate_inventory(inv, scope):
    require(inv['scope']==scope,'inventory scope mismatch')
    require(inv['status'] in {'complete','pending'},'inventory status invalid')
    mechanisms=inv['mechanisms']; papers=inv['papers']
    require(len({p['record_id'] for p in papers})==len(papers),'duplicate paper record')
    paper_ids={p['record_id'] for p in papers}
    for paper in papers:
        require(paper['record_id']==digest(paper['identity']),'paper record identity mismatch')
        require(set(paper['normalized_aliases'])==identities(paper['identity']),'paper aliases mismatch')
        require(bool(paper['evidence']),'paper missing provenance')
    for m in mechanisms:
        if 'paper_record_id' in m:
            require(m['paper_record_id'] in paper_ids,'mechanism missing source paper')
        if 'semantic_label' in m:
            require(m['normalized_semantic_label']==norm_title(m['semantic_label']),'semantic normalization mismatch')
    if inv['status']!='complete':
        return
    if scope==SCOPES[0]:
        require(len(mechanisms)==11 and len({m['registration_id'] for m in mechanisms})==11,'PA2 logical registration count')
    elif scope==SCOPES[1]:
        require(len(mechanisms)==19 and {m['cid'] for m in mechanisms}==HISTORICAL,'historical logical CID set')
        require(all(m['card_versions'] and m['exact_bound_version_matches']>0 and m['closest_known_paper_records'] for m in mechanisms),'historical provenance incomplete')
        require(all(set(m['closest_known_paper_records']) <= paper_ids for m in mechanisms),'historical closest source missing')
    elif scope==SCOPES[2]:
        require(len(mechanisms)==109 and len({m['mechanism_id'] for m in mechanisms})==109,'seed logical identity count')
        require(all(m['source_paper']['title'] and m['receipts'] and m['raw_row_sha256'] for m in mechanisms),'seed provenance incomplete')


def validate_stage(directory=HERE, sources=False):
    directory = Path(directory)
    stage = read_json(directory/'STAGE.v1.json')
    require(file_sha(directory/'STAGE.v1.json') == (directory/'STAGE.v1.sha256').read_text().split()[0], 'stage SHA mismatch')
    require(set(stage['artifacts']) >= {s+'.v1.json' for s in SCOPES}, 'missing required scope inventory')
    for n,h in stage['artifacts'].items():
        require(file_sha(directory/n)==h,'artifact SHA mismatch: '+n)
    for p,h in stage['frozen_base'].items():
        require(file_sha(p)==h,'frozen base SHA mismatch: '+p)
    registry = read_json(directory/'exclusions.v1.json')
    require(set(registry['scopes'])==set(SCOPES),'missing scope')
    require(file_sha(directory/'exclusions.v1.json')==(directory/'exclusions.v1.sha256').read_text().split()[0], 'registry SHA mismatch')
    for scope in SCOPES:
        inv = read_json(directory/(scope+'.v1.json'))
        validate_inventory(inv,scope)
        require(inv['status']==registry['scopes'][scope]['status'],'inventory status mismatch')
        entries = [e for e in registry['entries'] if e['scope']==scope]
        require(digest(entries)==registry['scopes'][scope]['inventory_sha256'],'inventory entries mismatch')
    roots=read_json(directory/'root-manifest.v1.json') if (directory/'root-manifest.v1.json').exists() else None
    if roots and roots['unknown_roots']:
        require(registry['scopes'][SCOPES[3]]['status']!='complete','unknown roots cannot claim complete scope')
    if sources:
        for p,row in read_json(directory/'source-hashes.v1.json')['files'].items():
            require(file_sha(p)==row['sha256'],'source SHA mismatch: '+p)
    return corpus_gate(read_json(HELD/'corpus.json'),registry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--verify-sources',action='store_true')
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_stage(sources=args.verify_sources),indent=2))
    else:
        build()
