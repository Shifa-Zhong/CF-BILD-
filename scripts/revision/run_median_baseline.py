'''Documented non-optimized median-distance GP reference.

This replaces the submitted arbitrary fixed-lengthscale comparison. Structural
lengthscales are medians of positive pairwise Euclidean distances among unique
standardized inputs; environmental lengthscales use the analogous 1D median.
Kernel variance is Var(y), and noise variance is 1% of Var(y). The kernel
family is held equal to the CV-selected model so that the comparison focuses
on target-driven hyperparameter optimization.
'''

from __future__ import annotations

import json
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch  # Import before RDKit on Windows to avoid DLL initialization conflicts.
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_bild.fragment_vocab import (  # noqa: E402
    FragmentVocabulary,
    load_property_datasets,
    prepare_cv_splits,
)
from cf_bild.gp_cvloss import GPCrossValidatedOptimizer  # noqa: E402
from cf_bild.predictive import physical_prediction, regression_metrics  # noqa: E402


OUTPUT_DIR = ROOT / 'output' / 'revision_2026'


def median_positive_pairwise(block, seed=42, maximum=1500):
    unique = np.unique(block, axis=0)
    if len(unique) > maximum:
        indices = np.random.default_rng(seed).choice(
            len(unique), maximum, replace=False
        )
        unique = unique[indices]
    distances = pdist(unique, metric='euclidean')
    distances = distances[distances > 1e-12]
    if len(distances) == 0:
        return 1.0, int(len(unique))
    return float(np.median(distances)), int(len(unique))


def main(data_directory=None, vocabulary_path=None, selected_directory=None, output_directory=None):
    global OUTPUT_DIR
    if output_directory is not None: OUTPUT_DIR = Path(output_directory).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_property_datasets(data_directory or ROOT / 'data', n_folds=5)
    vocab = FragmentVocabulary()
    vocab.load(vocabulary_path or ROOT / 'output' / 'fragment_vocab.pkl')
    result = {
        'definition': (
            'Median positive pairwise distance among unique standardized '
            'inputs; variance=Var(y); noise=0.01*Var(y).'
        ),
        'properties': {},
    }
    for property_name in ('co2', 'vis', 'tox'):
        cv_splits, x_train, y_train, x_test, y_test, _ = prepare_cv_splits(
            datasets, vocab, property_name
        )
        dim_cat = vocab.cat_fp_length
        dim_an = vocab.an_fp_length
        ls_cat, n_cat = median_positive_pairwise(
            x_train[:, :dim_cat], seed=42
        )
        ls_an, n_an = median_positive_pairwise(
            x_train[:, dim_cat:dim_cat + dim_an], seed=43
        )
        ls_cross, n_pair = median_positive_pairwise(
            x_train[:, :dim_cat + dim_an], seed=44
        )
        if selected_directory is not None:
            selected = json.loads((Path(selected_directory) / f'selected_hyperparameters_{property_name}.json').read_text())['best_params']
        else:
            with (ROOT / 'output' / f'model_{property_name}.pkl').open('rb') as h:
                selected = pickle.load(h)['best_params']
        params = {
            'kernel_name': selected['kernel_name'],
            'ls_cat': ls_cat,
            'ls_an': ls_an,
            'ls_cross': ls_cross,
            'kernel_variance': max(float(np.var(y_train)), 1e-6),
            'likelihood_noise_variance': max(
                float(np.var(y_train)) * 0.01, 1e-6
            ),
        }
        extra = x_train.shape[1] - dim_cat - dim_an
        for index in range(extra):
            value, _ = median_positive_pairwise(
                x_train[:, dim_cat + dim_an + index:index + dim_cat + dim_an + 1],
                seed=45 + index,
            )
            params[f'ls_env_{index}'] = value

        model = GPCrossValidatedOptimizer(
            X_train=x_train,
            y_train=y_train,
            predefined_cv_splits=cv_splits,
            compositional_kernel_dims=(dim_cat, dim_an),
            kernel_form='product',
            random_state=42,
        )
        model.best_params = params
        model._calibrate_variance()
        model.refit_best_model()
        latent_mu, latent_variance = model.predict(x_test)
        prediction = physical_prediction(
            property_name, latent_mu.ravel(), latent_variance.ravel()
        )
        metrics = regression_metrics(y_test, prediction)
        result['properties'][property_name] = {
            'parameters': params,
            'unique_cation_feature_rows': n_cat,
            'unique_anion_feature_rows': n_an,
            'unique_pair_feature_rows': n_pair,
            'variance_scale': float(model.variance_scale_),
            'metrics': metrics,
        }
        print(property_name, json.dumps(
            result['properties'][property_name], indent=2
        ), flush=True)

    with (OUTPUT_DIR / 'median_heuristic_baseline.json').open(
        'w', encoding='utf-8'
    ) as handle:
        json.dump(result, handle, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-directory', type=Path)
    parser.add_argument('--vocabulary-path', type=Path)
    parser.add_argument('--selected-directory', type=Path)
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    main(args.data_directory, args.vocabulary_path, args.selected_directory, args.output_directory)
