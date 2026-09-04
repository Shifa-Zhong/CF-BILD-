'''Verify author-local source-derived data without redistributing records.'''

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
            'Missing restricted files:\n' + '\n'.join(missing)
        )
    return {
        'manifest_version': 1,
        'contains_experimental_values': False,
        'redistribution_note': (
            'Record-level tables are intentionally absent. Obtain them from '
            'the original providers under their applicable terms.'
        ),
        'sources': {
            'co2_and_viscosity': {
                'name': 'NIST ILThermo (SRD 147)',
                'url': 'https://ilthermo.boulder.nist.gov/',
            },
            'toxicity': {
                'citation': (
                    'Zhao et al., Journal of Hazardous Materials 278 (2014) '
                    '320-329'
                ),
                'doi': '10.1016/j.jhazmat.2014.06.018',
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
            'Restricted-data verification failed:\n' + '\n'.join(failures)
        )
    print(f'PASS: {len(manifest["files"])} restricted files verified.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        '--write-manifest', action='store_true',
        help='Maintainer-only: build metadata from an authorized local copy.',
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
