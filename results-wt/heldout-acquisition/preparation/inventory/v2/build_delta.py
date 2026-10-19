"""Local source-backed inventory delta. No acquisition or changes to earlier stages."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
HELD = V1.parents[1]
SIBLING = HELD.parents[1]
MAIN = SIBLING.with_name('bigbrane')
sys.path.insert(0, str(HELD))
from harness import identities, Pool, norm_title


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':')) + '\n')


def verify_map(mapping):
    for path, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        require(sha(path) == expected, 'source SHA mismatch: ' + path)


def preserved():
    replay = SIBLING / 'results-wt/retrieval-diagnosis/PRESERVATION.json'
    seed = SIBLING / 'results-wt/seed-provenance/raw-baseline.json'
    a, b = load(replay)['saved_inputs_sha256'], load(seed)
    require((len(a), len(b)) == (274, 75), 'raw baseline counts')
    verify_map(a)
    verify_map(b)
    return {'replay_inputs': len(a), 'seed_inputs': len(b),
            'unique_paths': len(set(a) | set(b)), 'all_match': True,
            'baselines': {str(p): sha(p) for p in (replay, seed)}}


def public_identity(obj):
    if not isinstance(obj, dict):
        return None
    out = {}
    if isinstance(obj.get('title'), str) and obj['title'].strip():
        out['title'] = obj['title']
    aliases = []
    for field in ('arxiv_id', 'doi'):
        if obj.get(field):
            alias = {field: obj[field]}
            try:
                identities(alias)
                aliases.append(alias)
            except (ValueError, TypeError):
                pass
    if obj.get('source') and obj.get('source_id'):
        try:
            alias = {k: obj[k] for k in ('source', 'source_id')}
            identities(alias)
            aliases.append(alias)
        except (ValueError, TypeError):
            pass
    else:
        for field in ('source_id', 'id', 'url', 'key'):
            val = obj.get(field)
            if not isinstance(val, str):
                continue
            alias = identifier(val)
            if alias:
                aliases.append(alias)
    if aliases:
        out['aliases'] = list({digest(a): a for a in aliases}.values())
    return out or None


def identifier(value):
    value = value.strip()
    if re.fullmatch(r'(?:https://openalex.org/)?W\d+', value):
        return {'source': 'openalex', 'source_id': value}
    for field, pattern in [('arxiv_id', r'(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}|(?:arxiv:|https?://arxiv.org/(?:abs|pdf)/)\S+)'),
                           ('doi', r'(?:(?:https?://(?:dx\.)?doi.org/|doi:)?10\.\d{4,9}/\S+)')]:
        if re.fullmatch(pattern, value, flags=re.I):
            try:
                identities({field: value})
                return {field: value}
            except ValueError:
                return None
    return None


REFERENCE_FIELDS = {'references', 'papers', 'candidates', 'evidence', 'citations',
                    'source_ids', 'transplant_sources'}
SKIP_FIELDS = {'episodes', 'reference_samples', 'states', 'inputs', 'card_outputs',
               'abstract', 'response', 'request', 'messages', 'prompt', 'body', 'raw'}


def objects(value, pointer=''):
    if isinstance(value, dict):
        yield pointer, value
        for key, child in value.items():
            if key not in SKIP_FIELDS:
                yield from objects(child, pointer + '/' + str(key))
    elif isinstance(value, list):
        for j, child in enumerate(value):
            yield from objects(child, pointer + '/' + str(j))


def extract(value):
    papers, mechanisms = [], []
    for pointer, obj in objects(value):
        if isinstance(obj.get('id'), str) and obj['id'].startswith('card-') and isinstance(obj.get('title'), str) and 'core_idea' in obj and 'mechanism' in obj:
            mechanisms.append((obj.get('id'), {'title': obj['title']}, pointer,
                               digest(obj)))
        for field in sorted(REFERENCE_FIELDS):
            data = obj.get(field, [])
            for j, row in enumerate(data if isinstance(data, list) else [data]):
                item = public_identity(row) if isinstance(row, dict) else identifier(row) if isinstance(row, str) else None
                if item:
                    papers.append((item, pointer + '/' + field + '/' + str(j)))
    return papers, mechanisms


def pending_count(unknown, resolved):
    return len(set(unknown) - set(resolved))


class Reader:
    def __init__(self):
        self.records = {}
        self.documents = {}

    def bytes(self, path, role):
        path = str(Path(path).resolve())
        raw = Path(path).read_bytes()
        self.records[path] = {'sha256': hashlib.sha256(raw).hexdigest(),
                              'bytes': len(raw), 'role': role}
        return raw

    def json(self, path, role='source metadata'):
        path = str(Path(path).resolve())
        if path not in self.documents:
            self.documents[path] = json.loads(self.bytes(path, role))
        return self.documents[path]


PRUNE = {'.git', '.venv', '.venv-hybrid', 'venv', 'node_modules', '__pycache__',
         'llm', 'jobs', 'stores', 'cache', 'raw-cache', 'diagnostic-cache', 'postfix-cache',
         'deer-venv', 'rd-venv', 'deerflow', 'rdagent', 'site-packages', 'oauth', '.llxprt', 'profiles'}
ARCHIVES = ('.tar', '.tar.gz', '.tgz', '.zip', '.tar.xz', '.7z')


def selected(path):
    if path.name == 'ledger.jsonl':
        return 'ledger'
    if path.suffix in {'.json', '.toml'}:
        return 'metadata'
    if path.suffix in {'.md', '.py', '.tex', '.txt'}:
        return 'text/source'
    if path.suffix == '.log' and path.stat().st_size < 30000:
        return 'verification log'
    return None


def scan_root(root, reader, old_hashes, add, maxdepth=20, limit=200000):
    root = Path(root)
    receipt = {'path': str(root), 'files_enumerated': 0, 'selected_files': [],
               'pruned': [], 'unknown': [], 'suffixes': Counter(), 'copy_groups': {},
               'mechanism_ids': set(), 'paper_aliases': set(), 'ledgers': [],
               'immediate_entries': sorted(p.name for p in root.iterdir())}
    hashes = defaultdict(list)
    for folder, dirs, files in os.walk(root):
        if receipt['files_enumerated'] >= limit:
            receipt['unknown'].append({'path': str(root), 'reason': 'explicit file-count bound; later subtrees unexamined'})
            break
        folder = Path(folder)
        for name in sorted(dirs):
            if name in PRUNE or len(folder.relative_to(root).parts) >= maxdepth:
                receipt['pruned'].append({'path': str(folder / name),
                    'role': 'excluded runtime/raw bodies' if name in PRUNE else 'depth bound; unexamined',
                    'pending': name not in PRUNE or name in {'cache', 'raw-cache', 'diagnostic-cache', 'postfix-cache'}})
        dirs[:] = sorted(d for d in dirs if d not in PRUNE and len(folder.relative_to(root).parts) < maxdepth)
        for name in sorted(files):
            p = folder / name
            receipt['files_enumerated'] += 1
            receipt['suffixes'][p.suffix] += 1
            if receipt['files_enumerated'] > limit:
                receipt['unknown'].append({'path': str(folder), 'reason': 'explicit file-count bound'})
                dirs[:] = []
                break
            if name.endswith(ARCHIVES):
                receipt['unknown'].append({'path': str(p), 'reason': 'archive member study inputs not inspected'})
                continue
            if p.is_symlink() or not p.is_file():
                receipt.setdefault('special_files_excluded', []).append(str(p))
                continue
            role = selected(p)
            if not role:
                continue
            if p.stat().st_size > 2_000_000 and role != 'ledger':
                receipt['unknown'].append({'path': str(p), 'reason': 'over 2MB metadata/text bound'})
                continue
            if role == 'ledger':
                h, rows, bad, cids, paths = hashlib.sha256(), 0, 0, set(), set()
                with p.open('rb') as stream:
                    for line in stream:
                        h.update(line)
                        rows += 1
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeError):
                            bad += 1
                            continue
                        for _, obj in objects(row):
                            for key in ('card_id', 'cid'):
                                if isinstance(obj.get(key), str):
                                    cids.add(obj[key])
                            for key in ('artifact_path', 'path'):
                                if isinstance(obj.get(key), str) and obj[key].endswith('.json'):
                                    paths.add(obj[key])
                        # Identity-bearing structured event fields only, never event bodies.
                        ps, ms = extract(row)
                        for item, pointer in ps:
                            add(item, p, 'ledger:' + str(rows) + pointer, 'paper')
                        for cid, item, pointer, payload in ms:
                            add(item, p, 'ledger:' + str(rows) + pointer, 'mechanism', cid, payload)
                reader.records[str(p)] = {'sha256': h.hexdigest(), 'bytes': p.stat().st_size, 'role': role}
                receipt['ledgers'].append({'path': str(p), 'rows': rows, 'malformed': bad,
                    'card_ids': sorted(cids), 'artifact_paths': sorted(paths), 'passes': 1})
                if bad:
                    receipt['unknown'].append({'path': str(p), 'reason': 'malformed ledger rows'})
                raw = None
            else:
                raw = reader.bytes(p, role)
            sh = reader.records[str(p)]['sha256']
            hashes[sh].append(str(p))
            receipt['selected_files'].append(str(p))
            if role == 'metadata' and p.suffix == '.json':
                try:
                    data = json.loads(raw)
                except (ValueError, UnicodeError):
                    receipt['unknown'].append({'path': str(p), 'reason': 'invalid JSON'})
                    continue
                ps, ms = extract(data)
                for item, pointer in ps:
                    receipt['paper_aliases'].update(identities(item))
                    add(item, p, pointer, 'paper')
                for cid, item, pointer, payload in ms:
                    receipt['mechanism_ids'].add(cid or 'unknown')
                    add(item, p, pointer, 'mechanism', cid, payload)
                if p.name == 'manifest.json' and 'source_hashes' in data:
                    links = []
                    for rel, expected in data['source_hashes'].items():
                        target = p.parent / 'frozen' / rel
                        if target.is_file():
                            require(sha(target) == expected, 'pilot frozen source mismatch: ' + str(target))
                            links.append({'relative': rel, 'sha256': expected, 'verified_copy': str(target)})
                    receipt.setdefault('pilot_source_receipts', []).extend(links)
    receipt['copy_groups'] = {h: {'selected': ps, 'v1_sources': old_hashes[h]}
                              for h, ps in hashes.items() if h in old_hashes}
    receipt['selected_sha_groups'] = dict(hashes)
    receipt['mechanism_ids'] = sorted(receipt['mechanism_ids'])
    receipt['paper_aliases'] = sorted(receipt['paper_aliases'])
    receipt['status'] = 'bounded_scanned_pending' if receipt['unknown'] or any(r['pending'] for r in receipt['pruned']) else 'bounded_metadata_scanned'
    return receipt


def resolve_descriptors(scopes, reader, papers):
    by_alias, titles = defaultdict(list), {}
    for record in papers:
        for alias in record['normalized_aliases']:
            by_alias[alias].append(record)
        title = record['identity'].get('title')
        if title and len(norm_title(title)) >= 20:
            titles[norm_title(title)] = record
    resolutions, prose = [], []
    for scope, data in scopes.items():
        for u in data['unresolved']:
            if 'field' not in u:
                # These two named source gaps are handled from original card/PDF evidence below.
                continue
            candidates = []
            for path in u['sources']:
                p = Path(path)
                if p.suffix != '.json' or p.stat().st_size > 2_000_000:
                    continue
                document = reader.json(p, 'descriptor original context')
                for pointer, obj in objects(document):
                    for j, rel in enumerate(obj.get('known_relatives', []) if isinstance(obj.get('known_relatives'), list) else []):
                        if isinstance(rel, dict) and rel.get('name') == u['descriptor']:
                            candidates.append((path, pointer + '/known_relatives/' + str(j), rel))
                    if u['field'] != 'known_relatives':
                        val = obj.get(u['field'])
                        if isinstance(val, str) and val[:500] == u['descriptor']:
                            candidates.append((path, pointer + '/' + u['field'], val))
            found = {}
            for path, pointer, context in candidates:
                text = json.dumps(context, ensure_ascii=False)
                tokens = re.findall(r'(?<![\w.])(?:\d{4}\.\d{4,5}(?:v\d+)?|W\d{6,}|10\.\d{4,9}/[^\s"<>]+)', text)
                for token in tokens:
                    item = identifier(token.rstrip('.,;)'))
                    if item:
                        for alias in identities(item):
                            for record in by_alias.get(alias, []):
                                found[record['record_id']] = {'record_id': record['record_id'], 'identity': record['identity'],
                                    'match': alias, 'source': path, 'pointer': pointer,
                                    'context_sha256': digest(context), 'basis': 'explicit public identifier in original context'}
                normalized = norm_title(text)
                for title, record in titles.items():
                    if title in normalized:
                        found[record['record_id']] = {'record_id': record['record_id'], 'identity': record['identity'],
                            'match': 'title:' + title, 'source': path, 'pointer': pointer,
                            'context_sha256': digest(context), 'basis': 'exact normalized full recorded title in original context'}
            if found:
                resolutions.append({'scope': scope, 'unresolved_id': u['unresolved_id'],
                    'descriptor': u['descriptor'], 'links': list(found.values()),
                    'status': 'partial_explicit_links_ambiguity_retained',
                    'note': 'Named links resolved; family/composite prose is not asserted to name only these papers.'})
            if u['field'] == 'known_relatives':
                prose.append({'scope': scope, 'unresolved_id': u['unresolved_id'],
                    'normalized_descriptor': norm_title(u['descriptor']), 'classification': 'prose_alias_not_paper_title',
                    'source_refs': u['sources'], 'paper_merge': False})
    return resolutions, prose


def build():
    cutoff = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    reader = Reader()
    stage = load(V1 / 'STAGE.v1.json')
    v1_snapshot = {str(V1 / n): sha(V1 / n) for n in stage['artifacts']}
    verify_map({str(V1 / n): h for n, h in stage['artifacts'].items()})
    require(sha(V1 / 'STAGE.v1.json') == (V1 / 'STAGE.v1.sha256').read_text().split()[0], 'v1 stage pin')
    before = preserved()
    historical_harness = stage['frozen_base'][str(HELD / 'harness.py')]
    current_harness = sha(HELD / 'harness.py')
    stable = {p: h for p, h in stage['frozen_base'].items() if Path(p).name != 'harness.py'}
    verify_map(stable)
    write('preservation-before.json', {**before, 'v1_artifact_hashes': v1_snapshot,
        'stable_frozen_base': stable, 'historical_v1_harness_sha256': historical_harness,
        'implementation_revision_sha256': current_harness})
    print('Preserved 274 replay and 75 seed inputs', flush=True)
    sources = load(V1 / 'source-hashes.v1.json')['files']
    old_hashes = defaultdict(list)
    for path, record in sources.items():
        old_hashes[record['sha256']].append(path)
    scopes = {name: load(V1 / (name + '.v1.json')) for name in
              ['original_pa2_11', 'historical_recovery_19', 'seen_seed_109', 'other_used_study_data']}
    base = load(V1 / 'exclusions.v1.json')
    pool = Pool()
    for entry in base['entries']:
        pool.add(entry['identity'])
    base_aliases = set(pool.owners)
    base_union_count = len(pool.items)
    old_cids = {m.get('cid') or m.get('mechanism_id') for d in scopes.values() for m in d['mechanisms']}
    old_payloads = {v['card_payload_sha256'] for d in scopes.values() for m in d['mechanisms'] for v in m.get('card_versions', [])}
    additions, occurrences = {}, []
    def add(item, path, pointer, kind, cid=None, payload=None):
        aliases = identities(item)
        key = digest(item)
        occurrences.append({'identity_sha256': key, 'path': str(path), 'pointer': pointer,
                            'kind': kind, 'cid': cid, 'card_payload_sha256': payload,
                            'known_cid': cid in old_cids if cid else None,
                            'exact_known_card_payload': payload in old_payloads if payload else None,
                            'aliases_already_in_v1': aliases <= base_aliases})
        if not aliases <= base_aliases:
            additions.setdefault(key, {'scope': 'other_used_study_data', 'identity': item,
                                       'origin_path': str(path), 'kind': kind})
    roots = [r for r in load(V1 / 'root-manifest.v1.json')['roots'] if r['status'] == 'excluded_pending']
    receipts = []
    for index, root in enumerate(roots):
        print('Root', index + 1, root['path'], flush=True)
        receipts.append(scan_root(root['path'], reader, old_hashes, add))
        write('root-progress.json', {'cutoff': cutoff, 'finished': len(receipts), 'total': len(roots), 'roots': receipts})
    boundaries = []
    for record in load(V1 / 'TOP-LEVEL-BOUNDARIES.v1.json')['roots']:
        if record['status'] != 'excluded_pending':
            continue
        root = Path(record['path'])
        if root.name == '__pycache__':
            boundaries.append({'path': str(root), 'role': 'compiled Python cache', 'status': 'excluded_tool_output',
                               'entries': sorted(p.name for p in root.iterdir())})
            continue
        print('Top boundary', root, flush=True)
        # Temporary trees get a stated depth cutoff; other top-level directories are walked fully.
        receipt = scan_root(root, reader, old_hashes, add, maxdepth=5 if root.name == 'tmp' else 20,
                            limit=30000 if root.name == 'tmp' else 200000)
        receipt['role'] = 'mixed temporary study and test inputs' if root.name == 'tmp' else 'source/report/platform metadata'
        boundaries.append(receipt)
    write('metadata-scan-cache.json', {'roots': receipts, 'boundaries': boundaries,
        'sources': reader.records, 'additions': additions, 'occurrences': occurrences,
        'scanner_revision_sha256': sha(__file__), 'cutoff': cutoff})
    papers = [p for d in scopes.values() for p in d['papers']]
    resolutions, prose = resolve_descriptors({k: scopes[k] for k in ['historical_recovery_19', 'other_used_study_data']}, reader, papers)
    print('Descriptor links', len(resolutions), flush=True)
    probe_path = HELD / 'preparation/provider-syntax/PROBE-EXCLUSIONS.json'
    probe = reader.json(probe_path, 'returned provider probe exclusions')
    require(len(probe['papers']) == len(probe['registry_entries']) == 12, 'probe identity count')
    probe_bodies = {h for p in probe['papers'] for h in p['body_sha256']}
    body_receipts = []
    for p in sorted(probe_path.parent.rglob('*')):
        if p.is_file() and not p.is_symlink() and p.stat().st_size <= 2_000_000:
            h = sha(p)
            if h in probe_bodies:
                reader.records[str(p)] = {'sha256': h, 'bytes': p.stat().st_size, 'role': 'returned probe body hash only'}
                body_receipts.append({'path': str(p), 'sha256': h})
    require(probe_bodies <= {r['sha256'] for r in body_receipts}, 'probe response body SHA not found')
    probe_entries = probe['registry_entries']
    for entry in probe_entries:
        identities(entry['identity'])
        pool.add(entry['identity'])
    # Exact human-authored publication footer, not a title inferred from the URL.
    pdf = SIBLING / 'results-wt/evidence-recovery/remaining-sources/mitchell-ch2.pdf'
    extraction = HERE / 'mitchell-ch2.header.txt'
    text = extraction.read_text()
    require('lecture slides for textbook   Machine Learning, T. Mitchell, McGraw Hill, 1997' in text, 'Mitchell source footer missing')
    reader.bytes(pdf, 'retained original PDF, hash only')
    reader.bytes(extraction, 'local pdftotext first-page header/footer span')
    mitchell = {'scope': 'other_used_study_data', 'identity': {'title': 'Machine Learning'},
                'origin_path': str(pdf), 'origin_sha256': sha(pdf), 'kind': 'book_title',
                'source_url': 'https://www.cs.cmu.edu/afs/cs/project/theo-20/www/mlbook/ch2.pdf',
                'binding': {'path': str(extraction), 'sha256': sha(extraction), 'line_start': 18, 'line_end': 18},
                'note': 'PDF is lecture slides naming the textbook in the footer. Chapter heading is not substituted for book title; no public ID invented.'}
    additions[digest(mitchell['identity'])] = mitchell
    replay_path = SIBLING / 'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json'
    replay = reader.json(replay_path, 'original historical card and critic')
    passed = next(row for row in replay['cards'] if row['card_id'] == 'card-46dbb349')
    require(not passed['prior_art']['candidates'] and not passed['prior_art']['evidence'], 'pass prior art changed')
    named = {'pass46dbb349': {'source': str(replay_path), 'sha256': sha(replay_path),
        'card_sha256': digest(passed['card']), 'critic_sha256': digest(passed['critic']),
        'status': 'mechanism_excluded_no_structured_closest_publication',
        'inspected_fields': ['card.references', 'card.known_relatives', 'card.source_transplant', 'prior_art', 'critic', 'original'],
        'reference_arxiv_ids': [r['source_id'] for r in passed['card']['references']],
        'candidate_count': 0, 'evidence_count': 0, 'critic_mapping': passed['critic']['mapping'],
        'closest_paper_invented': False}, 'mitchell': mitchell}
    for entry in additions.values():
        entry['origin_sha256'] = entry.get('origin_sha256') or reader.records[entry['origin_path']]['sha256']
        pool.add(entry['identity'])
    # Refer to v1 unresolved records rather than duplicating their long source-path lists.
    unknown = {name: {'v1_total': len(d['unresolved']), 'fully_resolved': 0,
                     'remaining_ambiguity_records': len(d['unresolved']),
                     'records_with_explicit_links': sum(r['scope'] == name for r in resolutions)}
               for name, d in scopes.items()}
    unknown['other_used_study_data'].update(fully_resolved=1, closed_no_publication_claim=1,
        remaining_ambiguity_records=len(scopes['other_used_study_data']['unresolved']) - 2)
    named['pass46dbb349']['v1_unresolved_id'] = 'cf9ab62cbc2970dc8364dcfa3c15efffe4722273e1e9cb3b5b968d444a71a9be'
    named['mitchell']['v1_unresolved_id'] = 'd530f6ca56d2ce0a5412fdb1ba339ce1e76ddf2b0beedcd156acccebfe7b13ff'
    delta = {'version': 2, 'format': 'v1-pinned-base-plus-delta',
        'base': {'path': '../exclusions.v1.json', 'sha256': sha(V1 / 'exclusions.v1.json'),
                 'stage_path': '../STAGE.v1.json', 'stage_sha256': sha(V1 / 'STAGE.v1.json')},
        'entries': list(additions.values()), 'probe_entries': probe_entries,
        'scope_status': {k: ('PENDING' if k == 'other_used_study_data' else 'complete_named_enumeration_only') for k in scopes},
        'full_known_repo_scope_complete': False, 'exhaustive_aliases': False,
        'future_acquisition_ready': False, 'status': 'PENDING', 'unknown': unknown,
        'union': {'v1_conditional_unions': base_union_count, 'v2_conditional_unions': len(pool.items),
                  'v1_aliases': len(base_aliases), 'v2_aliases': len(pool.owners)},
        'probe_source': {'path': str(probe_path), 'sha256': sha(probe_path), 'body_receipts': body_receipts},
        'mechanism_and_reference_checks_required': True,
        'prose_aliases_separate_file': 'descriptor-prose-aliases.json'}
    # Compact alias-to-union checkpoint: no duplicate v1 evidence bodies.
    groups = sorted([sorted(item['ids']) for item in pool.items.values()])
    write('alias-checkpoint.json', {'version': 2, 'groups': groups,
        'assumption': 'conditional exact normalized title/public-ID union; not semantic independence'})
    write('registry.v2.json', delta)
    write('descriptor-resolutions.json', resolutions)
    write('descriptor-prose-aliases.json', prose)
    write('named-gaps.json', named)
    write('root-receipts.json', {'cutoff': cutoff, 'roots': receipts, 'occurrences': occurrences,
                               'scope': '20 explicitly named v1 pending roots; metadata selection only'})
    write('top-level-receipts.json', {'roots': boundaries, 'cutoff': cutoff,
        'scope': 'v1 unexamined top-level directories in the two Git-registered checkouts only'})
    # Counterpart hashes are reauthenticated, not merely trusted from old manifests.
    counterpart = {p: h for r in receipts + boundaries for h, row in r.get('copy_groups', {}).items() for p in row['v1_sources']}
    verify_map(counterpart)
    write('copy-counterparts.json', counterpart)
    write('source-receipts.json', reader.records)
    after = preserved()
    verify_map(v1_snapshot)
    verify_map(stable)
    require(sha(HELD / 'harness.py') == current_harness, 'implementation changed during build; rerun against explicit revision')
    write('preservation-after.json', {**after, 'v1_unchanged': True, 'stable_frozen_unchanged': True,
        'historical_harness_change_is_authorized_separate_work': current_harness != historical_harness,
        'implementation_revision_sha256': current_harness})
    summary = {'cutoff': cutoff, 'roots_scanned': len(receipts), 'top_boundaries_scanned': len(boundaries),
        'selected_files': len(reader.records), 'copy_counterparts_verified': len(counterpart),
        'roots_with_v1_byte_copies': sum(bool(r['copy_groups']) for r in receipts),
        'roots_with_mechanisms': sum(bool(r['mechanism_ids']) for r in receipts),
        'new_entry_projections': len(additions), 'probe_identities': 12,
        'descriptor_records_with_explicit_links': len(resolutions), 'unknown': unknown,
        'ledgers_streamed_once': sum(len(r.get('ledgers', [])) for r in receipts + boundaries),
        'union': delta['union'], 'status': 'PENDING', 'future_acquisition_ready': False}
    write('BUILD-RECEIPT.json', summary)
    print(json.dumps(summary), flush=True)


def validate():
    from registry_tools import check_stage
    check_stage(HERE)
    stage = load(HERE / 'STAGE.v2.json')
    verify_map({str(HERE / n): h for n, h in stage['artifacts'].items()})
    require(sha(V1 / 'exclusions.v1.json') == stage['v1_registry_sha256'], 'v1 registry changed')
    verify_map(load(HERE / 'source-receipts.json'))
    verify_map(load(HERE / 'copy-counterparts.json'))
    verify_map(load(HERE / 'preservation-before.json')['stable_frozen_base'])
    preserved()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    validate() if args.validate else build()
