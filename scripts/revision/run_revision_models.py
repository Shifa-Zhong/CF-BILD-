'''Refit the selected CF-BILD GPs on all non-test data and cache predictions.

Hyperparameters remain those selected by the original five predefined,
species-grouped CV folds. Only the final conditioning set changes from one
fold's training subset to the complete train+validation pool. The held-out
test set remains untouched.
'''

from __future__ import annotations

import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_bild.fragment_vocab import (  # noqa: E402
    FragmentVocabulary,
    load_property_datasets,
    prepare_cv_splits,
)
from cf_bild.gp_cvloss import GPCrossValidatedOptimizer  # noqa: E402
from cf_bild.predictive import (  # noqa: E402
    physical_prediction,
    regression_metrics,
    viscosity_real_space_metrics,
)


DATA_DIR = ROOT / 'data'
LEGACY_OUTPUT = ROOT / 'output'
OUTPUT_DIR = LEGACY_OUTPUT / 'revision_2026'
MODEL_DIR = OUTPUT_DIR / 'models'
HYPERPARAMETER_FILE = ROOT / 'config' / 'selected_hyperparameters.json'
PROPERTIES = ('co2', 'vis', 'tox')
SCREEN_T_K = 298.15
SCREEN_P_KPA = 101.325


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_vocabulary():
    vocab = FragmentVocabulary()
    vocab.load(LEGACY_OUTPUT / 'fragment_vocab.pkl')
    if len(vocab.cations) != 505 or len(vocab.anions) != 173:
        raise ValueError(
            f'Unexpected vocabulary size: {len(vocab.cations)} x '
            f'{len(vocab.anions)}'
        )
    return vocab


def candidate_features(vocab, property_name, start, end):
    cations = list(vocab.cation_fps)
    anions = list(vocab.anion_fps)
    n_anions = len(anions)
    indices = np.arange(start, end)
    cat_indices = indices // n_anions
    an_indices = indices % n_anions
    cat_matrix = np.asarray(
        [vocab.cation_fps[cations[index]] for index in cat_indices],
        dtype=np.float64,
    )
    an_matrix = np.asarray(
        [vocab.anion_fps[anions[index]] for index in an_indices],
        dtype=np.float64,
    )
    features = np.concatenate([cat_matrix, an_matrix], axis=1)
    if property_name in ('co2', 'vis'):
        environment = np.tile(
            np.array([[SCREEN_T_K, SCREEN_P_KPA]], dtype=np.float64),
            (end - start, 1),
        )
        features = np.concatenate([features, environment], axis=1)
    return features


def candidate_pairs(vocab):
    return [
        (cation, anion)
        for cation in vocab.cation_fps
        for anion in vocab.anion_fps
    ]


def conditional_nlpd(property_name, y, latent_mu, latent_variance):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(latent_mu, dtype=float)
    variance = np.maximum(np.asarray(latent_variance, dtype=float), 1e-16)
    value = 0.5 * (
        np.log(2.0 * np.pi * variance) + (y - mu) ** 2 / variance
    )
    if property_name == 'co2':
        from scipy.special import log_ndtr
        value += log_ndtr(mu / np.sqrt(variance))
    return float(np.mean(value))


def cpu_state_dict(module):
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def fit_property(property_name, datasets, vocab, batch_size=2000):
    start_time = time.time()
    print(f'\n=== Full non-test refit: {property_name} ===', flush=True)
    (
        cv_splits,
        x_refit,
        y_refit,
        x_test,
        y_test,
        scaler,
    ) = prepare_cv_splits(datasets, vocab, property_name)

    with HYPERPARAMETER_FILE.open(encoding='utf-8') as handle:
        selected = json.load(handle)[property_name]

    optimizer = GPCrossValidatedOptimizer(
        X_train=x_refit,
        y_train=y_refit,
        kernels=None,
        n_splits=5,
        random_state=42,
        predefined_cv_splits=cv_splits,
        compositional_kernel_dims=(
            vocab.cat_fp_length,
            vocab.an_fp_length,
        ),
        kernel_form='product',
    )
    optimizer.best_params = dict(selected['best_params'])
    optimizer.variance_scale_ = float(selected['variance_scale'])
    optimizer.refit_best_model()

    test_latent_mu, test_latent_variance = optimizer.predict(x_test)
    test_latent_mu = test_latent_mu.ravel()
    test_latent_variance = test_latent_variance.ravel()
    test_physical = physical_prediction(
        property_name, test_latent_mu, test_latent_variance
    )
    metrics = regression_metrics(y_test, test_physical)
    metrics['nlpd'] = conditional_nlpd(
        property_name, y_test, test_latent_mu, test_latent_variance
    )
    metrics['n_refit'] = int(len(y_refit))
    metrics['n_test'] = int(len(y_test))
    metrics['variance_scale'] = float(optimizer.variance_scale_)
    if property_name == 'vis':
        metrics['real_space'] = viscosity_real_space_metrics(
            y_test, test_physical['mean']
        )

    test_table = pd.DataFrame({
        'y_true': y_test,
        'latent_mu': test_latent_mu,
        'latent_std': np.sqrt(test_latent_variance),
        'pred_mean': test_physical['mean'],
        'pred_std': test_physical['std'],
        'lower_95': test_physical['lower'],
        'upper_95': test_physical['upper'],
    })
    test_table.to_csv(
        OUTPUT_DIR / f'test_predictions_{property_name}.csv', index=False
    )

    n_candidates = len(vocab.cations) * len(vocab.anions)
    latent_mu_parts = []
    latent_sigma_parts = []
    physical_mu_parts = []
    physical_sigma_parts = []
    for batch_start in range(0, n_candidates, batch_size):
        batch_end = min(batch_start + batch_size, n_candidates)
        x_raw = candidate_features(
            vocab, property_name, batch_start, batch_end
        )
        x_scaled = scaler.transform(x_raw)
        latent_mu, latent_variance = optimizer.predict(x_scaled)
        latent_mu = latent_mu.ravel()
        latent_variance = latent_variance.ravel()
        physical = physical_prediction(
            property_name, latent_mu, latent_variance
        )
        latent_mu_parts.append(latent_mu)
        latent_sigma_parts.append(np.sqrt(latent_variance))
        physical_mu_parts.append(physical['mean'])
        physical_sigma_parts.append(physical['std'])
        print(
            f'  candidates {batch_end:,}/{n_candidates:,}',
            flush=True,
        )

    artifact = {
        'artifact_version': 'DD-major-revision-2026-09',
        'property': property_name,
        'selected_hyperparameters': optimizer.best_params,
        'hyperparameter_selection': (
            'five predefined species-grouped CV folds; unchanged from the '
            'submitted analysis'
        ),
        'final_refit_pool': 'fold-1 train plus validation (complete non-test pool)',
        'n_refit': int(len(y_refit)),
        'n_test': int(len(y_test)),
        'y_train_mean': float(optimizer.y_train_mean_),
        'variance_scale': float(optimizer.variance_scale_),
        'scaler': scaler,
        'dim_cat': int(vocab.cat_fp_length),
        'dim_an': int(vocab.an_fp_length),
        'model_state': cpu_state_dict(optimizer.best_model_),
        'likelihood_state': cpu_state_dict(optimizer.best_likelihood_),
        'test_y_true': np.asarray(y_test),
        'test_latent_mu': test_latent_mu,
        'test_latent_variance': test_latent_variance,
        'test_pred_mean': test_physical['mean'],
        'test_pred_variance': test_physical['variance'],
        'selected_hyperparameter_file': (
            HYPERPARAMETER_FILE.relative_to(ROOT).as_posix()
        ),
        'selected_hyperparameter_file_sha256': sha256(HYPERPARAMETER_FILE),
    }
    with (MODEL_DIR / f'model_{property_name}.pkl').open('wb') as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metrics['elapsed_seconds'] = float(time.time() - start_time)
    print(json.dumps(metrics, indent=2), flush=True)
    result = {
        'latent_mu': np.concatenate(latent_mu_parts),
        'latent_sigma': np.concatenate(latent_sigma_parts),
        'physical_mu': np.concatenate(physical_mu_parts),
        'physical_sigma': np.concatenate(physical_sigma_parts),
        'metrics': metrics,
    }

    del optimizer, x_refit, x_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_vocabulary_provenance(vocab):
    from cf_bild.fragment_vocab import canonicalize_smiles

    provenance = {
        ('cation', smiles): {'properties': set(), 'roles': set(), 'files': set()}
        for smiles in vocab.cations
    }
    provenance.update({
        ('anion', smiles): {'properties': set(), 'roles': set(), 'files': set()}
        for smiles in vocab.anions
    })
    for path in sorted(DATA_DIR.glob('*_group_*.csv')):
        name = path.name
        property_name = name.rsplit('_', 1)[-1].replace('.csv', '')
        role = 'test' if name.startswith('test_') else (
            'validation' if name.startswith('val_') else 'training'
        )
        frame = pd.read_csv(path, usecols=['new_cation', 'new_anion'])
        for ion_type, column in (
            ('cation', 'new_cation'),
            ('anion', 'new_anion'),
        ):
            for raw_smiles in frame[column].dropna().astype(str).unique():
                smiles = canonicalize_smiles(raw_smiles.strip())
                key = (ion_type, smiles)
                if key in provenance:
                    provenance[key]['properties'].add(property_name)
                    provenance[key]['roles'].add(role)
                    provenance[key]['files'].add(name)
    rows = []
    for (ion_type, smiles), source in provenance.items():
        rows.append({
            'ion_type': ion_type,
            'canonical_smiles': smiles,
            'source_properties': ';'.join(sorted(source['properties'])),
            'source_split_roles': ';'.join(sorted(source['roles'])),
            'source_files': ';'.join(sorted(source['files'])),
        })
    table = pd.DataFrame(rows).sort_values(
        ['ion_type', 'canonical_smiles'], kind='stable'
    )
    table.to_csv(
        OUTPUT_DIR / 'fragment_vocabulary_provenance.csv', index=False
    )


def environment_versions():
    from importlib.metadata import version

    packages = (
        'numpy', 'pandas', 'scipy', 'scikit-learn', 'torch',
        'gpytorch', 'botorch', 'hyperopt', 'rdkit-pypi',
        'bit-collision-free-mf',
    )
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except Exception:
            result[package] = 'not-reported'
    result['python'] = sys.version.split()[0]
    result['cuda_available'] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name(0)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_property_datasets(DATA_DIR, n_folds=5)
    vocab = load_vocabulary()
    write_vocabulary_provenance(vocab)

    data_checksums = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(DATA_DIR.glob('*'))
        if path.is_file()
    }
    with (OUTPUT_DIR / 'data_sha256.json').open('w', encoding='utf-8') as handle:
        json.dump(data_checksums, handle, indent=2, sort_keys=True)

    results = {}
    for property_name in PROPERTIES:
        checkpoint = OUTPUT_DIR / f'candidate_predictions_{property_name}.npz'
        metrics_path = OUTPUT_DIR / f'metrics_{property_name}.json'
        if checkpoint.exists() and metrics_path.exists():
            print(f'Loading completed checkpoint for {property_name}.')
            arrays = np.load(checkpoint)
            with metrics_path.open(encoding='utf-8') as handle:
                metrics = json.load(handle)
            results[property_name] = {
                'latent_mu': arrays['latent_mu'],
                'latent_sigma': arrays['latent_sigma'],
                'physical_mu': arrays['physical_mu'],
                'physical_sigma': arrays['physical_sigma'],
                'metrics': metrics,
            }
            continue

        result = fit_property(property_name, datasets, vocab)
        np.savez_compressed(
            checkpoint,
            latent_mu=result['latent_mu'],
            latent_sigma=result['latent_sigma'],
            physical_mu=result['physical_mu'],
            physical_sigma=result['physical_sigma'],
        )
        with metrics_path.open('w', encoding='utf-8') as handle:
            json.dump(result['metrics'], handle, indent=2)
        results[property_name] = result

    cache = {
        'artifact_version': 'DD-major-revision-2026-09',
        'property_order': list(PROPERTIES),
        'candidate_cation_anion': candidate_pairs(vocab),
        'latent_mu': np.column_stack([
            results[name]['latent_mu'] for name in PROPERTIES
        ]),
        'latent_sigma': np.column_stack([
            results[name]['latent_sigma'] for name in PROPERTIES
        ]),
        'physical_mu': np.column_stack([
            results[name]['physical_mu'] for name in PROPERTIES
        ]),
        'physical_sigma': np.column_stack([
            results[name]['physical_sigma'] for name in PROPERTIES
        ]),
        'co2_posterior': 'Gaussian conditioned on capacity >= 0',
        'screen_temperature_k': SCREEN_T_K,
        'screen_pressure_kpa': SCREEN_P_KPA,
    }
    with (OUTPUT_DIR / 'predictions_87365_revision.pkl').open('wb') as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        'artifact_version': cache['artifact_version'],
        'environment': environment_versions(),
        'vocabulary': {
            'cations': len(vocab.cations),
            'anions': len(vocab.anions),
            'candidates': len(cache['candidate_cation_anion']),
            'cation_dimensions': vocab.cat_fp_length,
            'anion_dimensions': vocab.an_fp_length,
        },
        'properties': {
            name: results[name]['metrics'] for name in PROPERTIES
        },
    }
    with (OUTPUT_DIR / 'revision_model_summary.json').open(
        'w', encoding='utf-8'
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
