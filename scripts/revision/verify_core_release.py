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
    extra=RUN/'extensions/low_parameter_2026-09-06'
    for prop in ['co2','vis','tox']:
        meta=json.loads((extra/f'fit_manifest_{prop}.json').read_text());sig=meta['signature']
        for name,value in sig['inputs'].items():
            if digest(RUN/'data'/name)!=value:raise ValueError('Comparator input mismatch: '+name)
        for name,value in sig['code'].items():
            if digest(ROOT/name)!=value:raise ValueError('Comparator implementation mismatch: '+name)
        for name,value in meta['artifacts_sha256'].items():
            if digest(extra/name)!=value:raise ValueError('Comparator artifact mismatch: '+name)
        if digest(RUN/'fragment_vocab.pkl')!=sig['vocabulary']:raise ValueError('Comparator vocabulary mismatch')
        if digest(extra/f'search_{prop}/search_checkpoint.pkl')!=meta['checkpoint_sha256']:raise ValueError('Comparator search state mismatch')
        if json.loads((extra/f'RECONSTRUCTION_{prop}.json').read_text())['status']!='passed':raise ValueError('Comparator reconstruction not verified')
    diagnostics=json.loads((RUN/'extensions/ranking_diagnostics_2026-09-06/RANKING_DIAGNOSTICS.json').read_text())
    for name,value in diagnostics['source_sha256'].items():
        if digest(RUN/'analysis'/name)!=value:raise ValueError('Frozen diagnostic source mismatch: '+name)
    print(f'PASS: {len(manifest)} release hashes; 15 original and 3 additional model bindings; frozen diagnostic inputs. No pickle was loaded.')


if __name__=='__main__':main()
