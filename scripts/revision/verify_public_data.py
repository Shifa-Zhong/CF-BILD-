"""Verify released inputs and source-bearing artifacts without loading pickle."""

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(strict_splits=False):
    manifest = json.loads((ROOT / 'data/PUBLIC_DATA_RELEASE.json').read_text(encoding='utf-8'))
    for item in manifest['artifacts']:
        path = (ROOT / item['path']).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError('Manifest path is outside the repository.')
        if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Release artifact failed integrity check: ' + item['path'])
    warnings = []
    for prop in ('co2', 'vis', 'tox'):
        test = pd.read_csv(ROOT / 'data' / f'test_group_{prop}.csv')
        reference = None
        for fold in range(1, 6):
            train, val = [pd.read_csv(ROOT / 'data' / f'{role}_{fold}_group_{prop}.csv') for role in ('train', 'val')]
            sets = [set(t['group']) for t in (train, val, test)]
            if (sets[0] | sets[1]) & sets[2]:
                raise ValueError(f'Non-test/test encoded-group leakage: {prop}, fold {fold}')
            if sets[0] & sets[1]:
                warnings.append(f'{prop}, fold {fold}: {len(sets[0] & sets[1])} shared train/validation group(s)')
            current = pd.concat([train, val]).sort_values('ind').reset_index(drop=True)
            if reference is None:
                reference = current
            else:
                pd.testing.assert_frame_equal(reference, current)
        print(f'PASS {prop}: {len(reference)} non-test and {len(test)} test records; all five folds contain the same non-test record pool.')
    print('PASS:', len(manifest['artifacts']), 'exact public artifacts. This is an integrity/split test, not a chemical-identity certification.')
    for message in warnings:
        print('WARNING:', message)
    if strict_splits and warnings:
        raise ValueError('Strict species-disjoint CV check failed; see the disclosed warnings above.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-splits', action='store_true')
    main(parser.parse_args().strict_splits)
