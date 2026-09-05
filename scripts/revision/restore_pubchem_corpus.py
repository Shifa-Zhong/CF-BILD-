"""Verify compressed parts and losslessly restore the archived GRU corpus."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore(verify_only=False):
    directory = ROOT / 'data/gru/pubchem_corpus_portable'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    destination = ROOT / 'data/Pubchem dataset.txt'
    if destination.exists():
        if sha256(destination) != manifest['source_sha256']:
            raise ValueError('Existing corpus differs; it will not be overwritten.')
        if not verify_only:
            print('Existing corpus SHA-256 verified:', destination)
            return
    digest, size = hashlib.sha256(), 0
    temporary = None
    handle = None
    try:
        if not verify_only:
            handle = tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.pubchem-restore-', delete=False)
            temporary = Path(handle.name)
        for item in manifest['parts']:
            part = (directory / item['file']).resolve()
            if part.parent != directory.resolve() or sha256(part) != item['sha256']:
                raise ValueError('Invalid path or hash for corpus part.')
            part_size = 0
            with gzip.open(part, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
                    part_size += len(block)
                    if handle is not None:
                        handle.write(block)
            if part_size != item['uncompressed_bytes']:
                raise ValueError('Wrong decompressed part size.')
        if size != manifest['source_bytes'] or digest.hexdigest() != manifest['source_sha256']:
            raise ValueError('Restored corpus does not match original SHA-256.')
        if handle is not None:
            handle.close()
            handle = None
            if destination.exists():
                raise FileExistsError('Destination appeared during restoration; preserved unchanged.')
            os.replace(temporary, destination)
            temporary = None
        print('PASS: original PubChem corpus bytes and SHA-256 verified; bytes =', size)
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Stream and verify without writing a restored corpus.')
    restore(parser.parse_args().verify_only)
