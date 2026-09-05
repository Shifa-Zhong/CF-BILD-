"""Read-only verification of hash-bound scientific release artifacts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/ion_clean_refit_2026-09-05'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text(encoding='utf-8'))
    for row in manifest:
        path=(ROOT/row['path']).resolve()
        if ROOT not in path.parents:raise ValueError('Manifest path outside checkout')
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise ValueError('Changed release artifact: '+row['path'])
    for form in ['product','product_no_cross','additive','additive_no_cross','standard']:
        for prop in ['co2','vis','tox']:
            folder=RUN/'results'/form
            meta=json.loads((folder/f'fit_manifest_{prop}.json').read_text())
            sig=meta['signature']
            for name,checksum in sig['inputs'].items():
                if digest(RUN/'data'/name)!=checksum:raise ValueError('Fitted input mismatch: '+name)
            for name,checksum in meta['artifacts_sha256'].items():
                if digest(folder/name)!=checksum:raise ValueError('Fitted artifact mismatch: '+name)
            for key,name in [('optimizer_code','cf_bild/gp_cvloss.py'),('feature_code','cf_bild/fragment_vocab.py'),('runner_code','scripts/revision/run_clean_refit.py')]:
                if digest(ROOT/name)!=sig[key]:raise ValueError('Fitted implementation mismatch: '+name)
            if digest(RUN/'fragment_vocab.pkl')!=sig['vocabulary']:raise ValueError('Fitted vocabulary mismatch')
    print(f'PASS: {len(manifest)} release hashes; 15 model manifests and their exact data/code/vocabulary/artifact bindings. No pickle was loaded.')


if __name__=='__main__':main()
