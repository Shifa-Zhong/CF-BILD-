'''Verify exact public property records (legacy command name retained).'''

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'data' / 'restricted_data_manifest.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def property_files(data_dir):
    paths = []
    for property_name in ('co2', 'vis', 'tox'):
        paths.append(data_dir / f'test_group_{property_name}.csv')
        for fold in range(1, 6):
            paths.extend([
                data_dir / f'train_{fold}_group_{property_name}.csv',
                data_dir / f'val_{fold}_group_{property_name}.csv',
            ])
    return sorted(paths)


def metadata(path):
    frame = pd.read_csv(path)
    return {
        'bytes': path.stat().st_size,
        'rows': len(frame),
        'columns': list(frame.columns),
        'sha256': sha256(path),
    }


def build_manifest(data_dir):
    files = property_files(data_dir)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing property files:\n' + '\n'.join(missing)
        )
    return {
        'manifest_version': 2,
        'contains_experimental_values': False,
        'redistribution_note': (
            'Exact record-level inputs are publicly distributed with author authorization. '
            'The legacy filename is retained for compatibility; upstream attribution and terms apply.'
        ),
        'sources': {
            'direct_property_dataset_source': {
                'citation': 'Zhong et al., Environmental Science & Technology Letters 2024, 11, 1193-1199',
                'doi': '10.1021/acs.estlett.4c00524',
            },
            'co2_and_viscosity': {
                'name': 'NIST ILThermo (SRD 147)',
                'url': 'https://ilthermo.boulder.nist.gov/',
            },
            'toxicity': {
                'citation': (
                    'Wang, Song and Zhou, Machine Learning for Ionic Liquid '
                    'Toxicity Prediction, Processes 2021, 9, 65; IPC-81 subset'
                ),
                'doi': '10.3390/pr9010065',
            },
        },
        'files': {path.name: metadata(path) for path in files},
    }


def verify(data_dir, manifest):
    failures = []
    for name, expected in manifest['files'].items():
        path = data_dir / name
        if not path.exists():
            failures.append(f'{name}: missing')
            continue
        observed = metadata(path)
        for key in ('bytes', 'rows', 'columns', 'sha256'):
            if observed[key] != expected[key]:
                failures.append(
                    f'{name}: {key} expected {expected[key]!r}, '
                    f'observed {observed[key]!r}'
                )
    if failures:
        raise ValueError(
            'Property-data verification failed:\n' + '\n'.join(failures)
        )
    print(f'PASS: {len(manifest["files"])} public property files verified.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        '--write-manifest', action='store_true',
        help='Maintainer-only: build metadata from the exact public inputs.',
    )
    args = parser.parse_args()
    if args.write_manifest:
        payload = build_manifest(args.data_dir)
        args.manifest.write_text(
            json.dumps(payload, indent=2), encoding='utf-8'
        )
        print(args.manifest)
    else:
        payload = json.loads(args.manifest.read_text(encoding='utf-8'))
        verify(args.data_dir, payload)


if __name__ == '__main__':
    main()
