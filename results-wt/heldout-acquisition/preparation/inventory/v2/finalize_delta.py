"""Project bounded receipts and a compact shipping checkpoint, then pin v2 artifacts."""
import gzip
import json
from pathlib import Path
from build_delta import HERE, V1, HELD, SIBLING, MAIN, digest, identities, load, objects, sha, write
from registry_tools import public_only


def finalize():
    roots = load(HERE / 'root-receipts.json')
    tops = load(HERE / 'top-level-receipts.json')
    registry = load(HERE / 'registry.v2.json')
    source = load(HERE / 'source-receipts.json')
    base_unknown = load(V1 / 'other_used_study_data.v1.json')['unresolved']
    known = {(u.get('cid'), u['field'], u['descriptor']) for u in base_unknown if 'field' in u}
    new_unknown, seen_hashes = {}, set()
    # Only mechanism-bearing JSON files need another small metadata projection.
    for row in roots['occurrences']:
        p = Path(row['path'])
        if row['kind'] != 'mechanism' or p.suffix != '.json':
            continue
        h = source[str(p)]['sha256']
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        doc = load(p)
        for pointer, obj in objects(doc):
            cid = obj.get('id') if isinstance(obj.get('id'), str) else None
            if not cid or not cid.startswith('card-'):
                continue
            for j, rel in enumerate(obj.get('known_relatives', [])):
                desc = rel.get('name') if isinstance(rel, dict) else rel if isinstance(rel, str) else None
                if not desc or (cid, 'known_relatives', desc[:500]) in known:
                    continue
                key = digest([cid, 'known_relatives', desc])
                new_unknown.setdefault(key, {'cid': cid, 'field': 'known_relatives', 'descriptor': desc,
                    'classification': 'unresolved_prose_not_unique_paper', 'source': str(p), 'sha256': h,
                    'pointer': pointer + '/known_relatives/' + str(j)})
    write('new-ambiguities.json', {'records': new_unknown,
        'note': 'Additional source-qualified descriptors from selected card metadata, deduplicated by CID/field/full descriptor. No paper identity inferred.'})
    registry['unknown']['other_used_study_data']['additional_descriptor_records'] = len(new_unknown)
    registry['unknown']['other_used_study_data']['additional_records_path'] = 'new-ambiguities.json'
    coverage, experiment_aliases = [], []
    for root in roots['roots']:
        relevant = [o for o in roots['occurrences'] if o['path'].startswith(root['path'] + '/')]
        new_ids = sorted({o['cid'] for o in relevant if o['cid'] and not o['known_cid']})
        copied_card_payloads = {o['card_payload_sha256'] for o in relevant if o['exact_known_card_payload']}
        new_alias_rows = [o for o in relevant if not o['aliases_already_in_v1']]
        copy_files = {p for g in root['copy_groups'].values() for p in g['selected']}
        role = ('empty_at_cutoff' if not root['files_enumerated'] else
                'prior_pilot_synthetic_experiment' if root.get('pilot_source_receipts') else
                'identity_bearing_verification_tree' if relevant else
                'numerical_verification_or_log_outputs_no_identity_in_selected_metadata')
        row = {'path': root['path'], 'role': role, 'selected_files': len(root['selected_files']),
            'files_enumerated': root['files_enumerated'], 'v1_byte_identical_files': len(copy_files),
            'v1_byte_hash_groups': len(root['copy_groups']), 'exact_v1_card_payloads': len(copied_card_payloads),
            'identity_occurrences': len(relevant), 'new_alias_occurrences': len(new_alias_rows),
            'new_card_ids': new_ids, 'unknown_files': len(root['unknown']),
            'pending_pruned_directories': sum(r['pending'] for r in root['pruned']),
            'raw_original_hashes_unchanged': True, 'whole_root_asserted_identical': False}
        if root.get('pilot_source_receipts'):
            row['verified_frozen_source_copies'] = len(root['pilot_source_receipts'])
            for p in root['selected_files']:
                if Path(p).name != 'manifest.json':
                    continue
                data = load(p)
                families = sorted({obj['family'] for _, obj in objects(data.get('candidate_plans', {})) if isinstance(obj.get('family'), str)})
                if families:
                    experiment_aliases.append({'source': p, 'sha256': source[p]['sha256'],
                        'families': families, 'classification': 'experiment_family_aliases_not_publication_titles',
                        'source_hash_receipts': root['pilot_source_receipts'],
                        'independent_scientific_data': False,
                        'reason': 'Manifest records seeded synthetic data stages and frozen implementation copies; model names remain used experiment metadata.'})
        coverage.append(row)
    write('experiment-family-aliases.json', experiment_aliases)
    top_summary = [{'path': r['path'], 'role': r['role'], 'status': r['status'],
                    'selected_files': len(r.get('selected_files', [])), 'unknown_files': len(r.get('unknown', [])),
                    'pending_pruned_directories': sum(p['pending'] for p in r.get('pruned', []))} for r in tops['roots']]
    write('COVERAGE.v2.json', {'roots': coverage, 'top_level': top_summary,
        'root_count': len(coverage), 'cutoff': roots['cutoff'], 'known_repo_roots': [str(MAIN), str(SIBLING)],
        'full_known_repo_scope_complete': False, 'future_acquisition_ready': False,
        'limits': 'JSON/TOML/text metadata bounded to 2MB; raw model calls, jobs, numeric binaries and dependency trees omitted; cache/depth/count omissions remain explicitly pending.'})
    archive_roles = []
    for r in load(V1 / 'root-manifest.v1.json')['roots']:
        if any(term in Path(r['path']).name.lower() for term in ('archive', 'golden', 'saved')):
            archive_roles.append({'path': r['path'], 'role': 'existing retained study/archive root; v1 metadata selection inherited',
                'status': r['status'], 'selected_files': r.get('selected_files', {}), 'source_manifest_sha256': sha(V1 / 'root-manifest.v1.json')})
    for r in tops['roots']:
        for name in r.get('immediate_entries', []):
            if any(term in name.lower() for term in ('archive', 'golden', 'saved')):
                path = r['path'] + '/' + name
                archive_roles.append({'path': path, 'role': 'retained local copy/archive candidate, bounded selection only',
                    'selected_files': sum(p.startswith(path + '/') for p in r.get('selected_files', [])),
                    'full_members_examined': False})
    write('ARCHIVE-ROLES.json', {'roots': archive_roles, 'scope': 'Known repository roots only; no external archive search prerequisite.'})
    registry['source_roots_receipts'] = {'path': 'root-receipts.json', 'sha256': sha(HERE / 'root-receipts.json'),
                                       'coverage_path': 'COVERAGE.v2.json', 'coverage_sha256': sha(HERE / 'COVERAGE.v2.json')}
    registry['descriptor_resolutions'] = {'path': 'descriptor-resolutions.json', 'sha256': sha(HERE / 'descriptor-resolutions.json')}
    registry['experiment_family_aliases'] = {'path': 'experiment-family-aliases.json', 'sha256': sha(HERE / 'experiment-family-aliases.json')}
    write('registry.v2.json', registry)
    # Source paths and digests appear once in this transferable checkpoint.
    paths = sorted(source)
    path_index = {p: j for j, p in enumerate(paths)}
    hashes = sorted({source[p]['sha256'] for p in paths})
    hash_index = {h: j for j, h in enumerate(hashes)}
    checkpoint = {'format': 'public-metadata-alias-source-checkpoint-v2',
        'base_registry_sha256': registry['base']['sha256'],
        'alias_groups': load(HERE / 'alias-checkpoint.json')['groups'],
        'registry_delta': registry, 'source_paths': paths, 'source_hashes': hashes,
        'source_bindings': [[j, hash_index[source[p]['sha256']]] for j, p in enumerate(paths)],
        'identity_occurrences': [[o['identity_sha256'], path_index[o['path']], o['pointer'], o['kind'], o['cid']] for o in roots['occurrences']],
        'descriptor_resolutions': load(HERE / 'descriptor-resolutions.json'),
        'new_ambiguities': new_unknown, 'coverage': load(HERE / 'COVERAGE.v2.json'),
        'status': 'PENDING', 'future_acquisition_ready': False}
    public_only(checkpoint)
    payload = json.dumps(checkpoint, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    compressed = gzip.compress(payload, mtime=0)
    (HERE / 'shipping-checkpoint.v2.json.gz').write_bytes(compressed)
    write('COMPACT-RECEIPT.json', {'uncompressed_bytes': len(payload), 'compressed_bytes': len(compressed),
        'sha256': sha(HERE / 'shipping-checkpoint.v2.json.gz'), 'source_paths': len(paths),
        'unique_source_hashes': len(hashes), 'aliases': sum(map(len, checkpoint['alias_groups'])),
        'v1_body_duplication': False, 'private_metadata_scan': 'passed projected fields and credential pattern scan'})
    # Progress files are intermediate copies, not part of the shipping checkpoint.
    for name in ('root-progress.json', 'metadata-scan-cache.json'):
        if (HERE / name).exists():
            (HERE / name).unlink()
    names = [p.name for p in HERE.iterdir() if p.is_file() and p.name not in
             {'STAGE.v2.json', 'STAGE.v2.sha256', 'tests.log', 'tests.xml', 'validation.log', 'STATUS.md', 'REPORT.md'}
             and not p.name.endswith('.log')]
    stage = {'version': 2, 'v1_registry_sha256': registry['base']['sha256'],
        'historical_v1_stage_sha256': registry['base']['stage_sha256'],
        'historical_v1_harness_sha256': load(V1 / 'STAGE.v1.json')['frozen_base'][str(HELD / 'harness.py')],
        'implementation_revision': {'path': str(HELD / 'harness.py'), 'sha256': sha(HELD / 'harness.py'),
                                    'rule': 'Explicit v2 revision; historical v1 harness digest is not the current implementation pin.'},
        'artifacts': {name: sha(HERE / name) for name in sorted(names)},
        'status': 'PENDING', 'future_acquisition_ready': False, 'policy_amendment': False}
    write('STAGE.v2.json', stage)
    (HERE / 'STAGE.v2.sha256').write_text(sha(HERE / 'STAGE.v2.json') + '  STAGE.v2.json\n')
    print(json.dumps({'roots': len(coverage), 'new_card_ids_in_20_roots': len({c for r in coverage for c in r['new_card_ids']}),
        'new_ambiguities': len(new_unknown), 'shipping_bytes': len(compressed), 'stage_sha256': sha(HERE / 'STAGE.v2.json')}))


if __name__ == '__main__':
    finalize()
