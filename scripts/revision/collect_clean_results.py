"""Collect only hash-verified curated-data fits into size-independent analysis inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cf_bild.fragment_vocab import FragmentVocabulary
from cf_bild.ion_validation import require_valid_pairs


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run):
    run = Path(run).resolve()
    results, output = run / 'results/product', run / 'analysis'
    all_metrics, hyperparameters, source_hashes = {}, {}, {}
    for prop in ('co2', 'vis', 'tox'):
        manifest = json.loads((results / f'fit_manifest_{prop}.json').read_text())
        for name, checksum in manifest['artifacts_sha256'].items():
            if digest(results / name) != checksum: raise ValueError(f'Fit artifact changed: {name}')
            source_hashes[name] = checksum
        all_metrics[prop] = json.loads((results / f'metrics_{prop}.json').read_text())
        hyperparameters[prop] = json.loads((results / f'selected_hyperparameters_{prop}.json').read_text())
    vocab = FragmentVocabulary()
    vocab.load(run / 'fragment_vocab.pkl')
    pairs = [(c, a) for c in vocab.cations for a in vocab.anions]
    require_valid_pairs(pairs, context='collected fitted candidate pool')
    cache = {}
    for key in ('latent_mu', 'latent_sigma', 'physical_mu', 'physical_sigma'):
        parts = []
        for prop in ('co2', 'vis', 'tox'):
            with np.load(results / f'candidate_predictions_{prop}.npz', allow_pickle=False) as arrays:
                part = arrays[key]
                if part.shape != (len(pairs),) or not np.all(np.isfinite(part)):
                    raise ValueError(f'Prediction array invalid: {prop}/{key}')
                parts.append(part)
        cache[key] = np.column_stack(parts)
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / 'candidate_predictions.npz', **cache)
    pd.DataFrame(pairs, columns=['cation', 'anion']).to_csv(output / 'candidate_pairs.csv', index=False)
    v = json.loads((run / 'VOCABULARY_SUMMARY.json').read_text())
    summary = {'artifact_version': 'DD-ion-clean-refit-2026-09',
        'environment': json.loads((run / 'COMPUTATIONAL_ENVIRONMENT.json').read_text()),
        'vocabulary': {'cations': v['n_cations'], 'anions': v['n_anions'], 'candidates': v['n_candidates'],
            'cation_dimensions': v['cation_dimensions'], 'anion_dimensions': v['anion_dimensions']},
        'properties': all_metrics}
    (output / 'revision_model_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'selected_hyperparameters.json').write_text(json.dumps(hyperparameters, indent=2), encoding='utf-8')
    for prop in ('co2', 'vis', 'tox'):
        for name in (f'test_predictions_{prop}.csv', f'calibration_{prop}.json'):
            shutil.copyfile(results / name, output / name)
    meta = {'property_order': ['co2', 'vis', 'tox'], 'n_candidates': len(pairs),
        'screen_temperature_k': 298.15, 'screen_pressure_kpa': 101.325,
        'co2_distribution': 'Gaussian conditioned on capacity >= 0',
        'source_fit_artifacts_sha256': source_hashes,
        'cache_sha256': digest(output / 'candidate_predictions.npz'),
        'pair_table_sha256': digest(output / 'candidate_pairs.csv'),
        'old_prediction_caches_used': False}
    (output / 'candidate_prediction_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Collected verified predictions for {len(pairs):,} charge-balanced pairs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    main(parser.parse_args().run_directory)
