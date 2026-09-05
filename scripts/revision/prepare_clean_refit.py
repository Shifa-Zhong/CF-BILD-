"""Prepare auditable ion-identity exclusions and species-disjoint CV folds.

Original CSV bytes and targets are preserved. No protonation, charge or
stoichiometric coefficient is inferred from a property value or model result.
The optional multivalent exclusion is a scope decision, not an assertion that
the material is chemically impossible. Use a fresh output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch  # Import before RDKit on Windows.
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cf_bild.ion_validation import check_ion_pair
from cf_bild.fragment_vocab import canonicalize_smiles, TARGET_COLUMNS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_group(cation, anion):
    cat = canonicalize_smiles(str(cation).strip())
    an = canonicalize_smiles(str(anion).strip())
    if cat is None or an is None:
        return None
    return cat + '.' + an


def read_frame(path):
    # Preserve numeric tokens and original component strings exactly.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def classify(frame):
    return [check_ion_pair(c, a) for c, a in
            frame[['new_cation', 'new_anion']].itertuples(index=False, name=None)]


def validation_assignments(non_test, old_validations):
    """Retain the modal original validation fold per canonical ion pair.

    Consolidate a split species into one fold (ties: lowest fold number).
    Only structure, original memberships and record counts are used; targets
    play no role. Every retained record is validation exactly once.
    """
    votes = defaultdict(Counter)
    retained_ids = set(non_test['ind'])
    for fold, frame in enumerate(old_validations, 1):
        for row in frame.to_dict('records'):
            if row['ind'] in retained_ids:
                group = canonical_group(row['new_cation'], row['new_anion'])
                votes[group][fold] += 1
    assignments = {}
    for group in sorted(set(non_test['group'])):
        if not votes[group]:
            raise ValueError(f'No original validation assignment for {group}')
        assignments[group] = min(votes[group], key=lambda k: (-votes[group][k], k))
    return assignments, votes


def prepare(source, output, exclude_unverified_stoichiometry=False, properties=('co2', 'vis', 'tox')):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    prepared, sources = {}, {}
    audit_rows, fold_changes, summaries = [], [], {}
    for prop in properties:
        folds = [(read_frame(source / f'train_{k}_group_{prop}.csv'),
                  read_frame(source / f'val_{k}_group_{prop}.csv')) for k in range(1, 6)]
        test = read_frame(source / f'test_group_{prop}.csv')
        complete = pd.concat(folds[0], ignore_index=True)
        if not complete['ind'].is_unique or not test['ind'].is_unique:
            raise ValueError(f'Non-unique source row identifiers: {prop}')
        expected = Counter(map(tuple, complete.to_numpy()))
        for k, (train, val) in enumerate(folds, 1):
            if Counter(map(tuple, pd.concat([train, val], ignore_index=True).to_numpy())) != expected:
                raise ValueError(f'Original non-test pool mismatch: {prop} fold {k}')
        eligible = {}
        for role, frame in [('non_test', complete), ('test', test)]:
            checks = classify(frame)
            reasons = Counter(check.reason for check in checks if not check.valid)
            if reasons['multivalent_salt_requires_explicit_stoichiometry'] and not exclude_unverified_stoichiometry:
                raise ValueError('Multivalent exclusion requires the explicit author-approved '
                                 '--exclude-unverified-stoichiometry option; no files written.')
            for row, check in zip(frame.to_dict('records'), checks):
                audit_rows.append({'property': prop, 'source_partition': role,
                    'source_record_id': row['ind'], 'cation': row['new_cation'],
                    'anion': row['new_anion'], 'canonical_group': canonical_group(row['new_cation'], row['new_anion']),
                    'decision': 'retain' if check.valid else 'exclude', **asdict(check)})
            kept = frame.loc[[check.valid for check in checks]].copy()
            kept['group'] = [canonical_group(c, a) for c, a in
                             kept[['new_cation', 'new_anion']].itertuples(index=False, name=None)]
            eligible[role] = kept
            summaries[f'{prop}_{role}'] = {'source_records': len(frame), 'retained_records': len(kept),
                'excluded_records': len(frame)-len(kept), 'retained_species': kept['group'].nunique(),
                'exclusion_reasons_records': dict(reasons)}
        non_test, kept_test = eligible['non_test'], eligible['test']
        if set(non_test['group']) & set(kept_test['group']):
            raise ValueError(f'Canonical non-test/test group overlap: {prop}; source review required')
        assignment, votes = validation_assignments(non_test, [v for _, v in folds])
        validation_count = Counter()
        for k, (old_train, old_val) in enumerate(folds, 1):
            val_mask = non_test['group'].map(assignment).eq(k)
            new_train, new_val = non_test.loc[~val_mask], non_test.loc[val_mask]
            if not len(new_train) or not len(new_val):
                raise ValueError(f'Empty repaired fold: {prop} {k}')
            assert not (set(new_train['group']) & set(new_val['group']))
            validation_count.update(new_val['ind'])
            prepared[f'train_{k}_group_{prop}.csv'] = new_train
            prepared[f'val_{k}_group_{prop}.csv'] = new_val
            old_ids = set(old_val['ind'])
            for row in non_test.to_dict('records'):
                old_is_val, new_is_val = row['ind'] in old_ids, assignment[row['group']] == k
                if old_is_val != new_is_val:
                    fold_changes.append({'property': prop, 'record_id': row['ind'], 'fold': k,
                        'canonical_group': row['group'], 'old_role': 'validation' if old_is_val else 'training',
                        'new_role': 'validation' if new_is_val else 'training'})
        assert validation_count == Counter({record: 1 for record in non_test['ind']})
        prepared[f'test_group_{prop}.csv'] = kept_test
        for name in [f'{role}_{k}_group_{prop}.csv' for k in range(1, 6) for role in ('train', 'val')]+[f'test_group_{prop}.csv']:
            sources[name] = sha256(source / name)
    output.mkdir(parents=True)
    (output / 'data').mkdir()
    (output / 'raw').mkdir()
    for name, frame in prepared.items():
        frame.to_csv(output / 'data' / name, index=False)
        shutil.copyfile(source / name, output / 'raw' / name)
        assert sha256(output / 'raw' / name) == sources[name]
        original = read_frame(source / name) if name.startswith('test_') else None
        if original is not None:
            assert set(frame['ind']).issubset(set(original['ind']))
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / 'record_curation_audit.csv', index=False)
    audit.loc[audit.decision.eq('exclude')].to_csv(output / 'excluded_records.csv', index=False)
    pd.DataFrame(fold_changes, columns=['property', 'record_id', 'fold', 'canonical_group', 'old_role', 'new_role']).to_csv(
        output / 'fold_membership_changes.csv', index=False)
    report = {'status': 'cleaned_inputs_ready', 'properties': list(properties),
        'policy': 'parseable_connected_+1_-1_pairs',
        'unverified_stoichiometry_exclusion_authorized': exclude_unverified_stoichiometry,
        'original_sources_modified': False, 'targets_modified': False, 'charges_inferred_or_repaired': False,
        'held_out_partition_resampled': False, 'cv_policy': 'modal original validation fold per canonical species; tie lowest fold',
        'test_group_overlap': 0, 'cv_group_overlap': 0, 'validation_memberships_per_record': 1,
        'partitions': summaries, 'fold_membership_change_count': len(fold_changes),
        'source_sha256': sources, 'cleaned_sha256': {name: sha256(output / 'data' / name) for name in prepared}}
    (output / 'CURATION_SUMMARY.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--exclude-unverified-stoichiometry', action='store_true')
    parser.add_argument('--properties', nargs='+', choices=['co2', 'vis', 'tox'], default=['co2', 'vis', 'tox'])
    args = parser.parse_args()
    prepare(args.source_directory, args.output_directory, args.exclude_unverified_stoichiometry, args.properties)
