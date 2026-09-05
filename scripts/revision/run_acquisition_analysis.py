'''Canonical acquisition analysis for the Digital Discovery revision.

The script consumes predictions from the full non-test refits, uses thresholds
and a reference point derived only from non-test targets, and evaluates
analytical q=1 EHVI against a non-empty incumbent Pareto front.
'''

from __future__ import annotations

import json
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_bild.acquisition import (  # noqa: E402
    additive_ei_scores,
    exact_q1_ehvi_scores,
    fw_aei_scores,
    incumbent_front,
    pareto_mask,
    selected_set_metrics,
    to_maximization,
    top_indices,
)
from cf_bild.fragment_vocab import canonicalize_smiles  # noqa: E402
from cf_bild.ion_validation import require_valid_pairs  # noqa: E402


DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output' / 'revision_2026'
PUBLIC_INPUTS_PATH = OUTPUT_DIR / 'acquisition_inputs_revision.json'
PROPERTIES = ('co2', 'vis', 'tox')
TARGET_COLUMNS = {
    'co2': 'CO2-exp',
    'vis': 'vis',
    'tox': 'Experimental logEC50',
}


def load_cache():
    generic = OUTPUT_DIR / 'candidate_predictions.npz'
    if generic.exists():
        with np.load(generic, allow_pickle=False) as arrays:
            cache = {name: arrays[name].copy() for name in arrays.files}
        pairs = pd.read_csv(OUTPUT_DIR / 'candidate_pairs.csv')
        cache['candidate_cation_anion'] = list(pairs[['cation', 'anion']].itertuples(index=False, name=None))
        cache['co2_posterior'] = 'Gaussian conditioned on capacity >= 0'
        for name in ('latent_mu', 'latent_sigma', 'physical_mu', 'physical_sigma'):
            if cache[name].shape != (len(pairs), 3) or not np.all(np.isfinite(cache[name])):
                raise ValueError('Candidate cache shape or finiteness mismatch')
        return cache
    with (OUTPUT_DIR / 'predictions_87365_revision.pkl').open('rb') as handle:
        return pickle.load(handle)


def non_test_frame(property_name):
    train = pd.read_csv(DATA_DIR / f'train_1_group_{property_name}.csv')
    validation = pd.read_csv(DATA_DIR / f'val_1_group_{property_name}.csv')
    return pd.concat([train, validation], ignore_index=True)


def operating_points_from_restricted_data(percentile=75):
    targets = {
        name: non_test_frame(name)[TARGET_COLUMNS[name]].to_numpy()
        for name in PROPERTIES
    }
    reference = np.array([
        targets['co2'].min(),
        -targets['vis'].max(),
        targets['tox'].min(),
    ])
    thresholds = np.array([
        np.percentile(targets['co2'], percentile),
        -np.percentile(targets['vis'], 100 - percentile),
        np.percentile(targets['tox'], percentile),
    ])
    return reference, thresholds


def restricted_tables_available():
    return all(
        (DATA_DIR / f'{role}_1_group_{property_name}.csv').exists()
        for property_name in PROPERTIES
        for role in ('train', 'val')
    )


def canonical_pair(cation, anion):
    return (
        canonicalize_smiles(str(cation).strip()),
        canonicalize_smiles(str(anion).strip()),
    )


def incumbent_indices(cache):
    represented = set()
    for property_name in PROPERTIES:
        frame = non_test_frame(property_name)
        represented.update(
            canonical_pair(cation, anion)
            for cation, anion in zip(
                frame['new_cation'], frame['new_anion']
            )
        )
    lookup = {
        canonical_pair(cation, anion): index
        for index, (cation, anion) in enumerate(
            cache['candidate_cation_anion']
        )
    }
    return np.array(sorted(
        lookup[pair] for pair in represented if pair in lookup
    ), dtype=int)


def load_or_build_public_inputs(cache):
    '''Load aggregated inputs, or rebuild them when restricted data are local.

    The public artifact contains only operating summaries and the model-derived
    incumbent Pareto coordinates. It contains no source experimental record.
    '''
    if restricted_tables_available():
        physical_mu = to_maximization(cache['physical_mu'])
        reference, _ = operating_points_from_restricted_data(75)
        incumbent_ids = incumbent_indices(cache)
        front = incumbent_front(physical_mu, reference, incumbent_ids)
        payload = {
            'artifact_version': 'DD-major-revision-2026-09',
            'content_note': (
                'Aggregated operating definitions and model-derived Pareto '
                'coordinates only; no source experimental records.'
            ),
            'reference_point_maximization_frame': reference.tolist(),
            'feasibility_thresholds_by_percentile': {
                str(percentile): operating_points_from_restricted_data(
                    percentile
                )[1].tolist()
                for percentile in (50, 60, 70, 75, 80, 90)
            },
            'n_incumbent_identities': int(len(incumbent_ids)),
            'incumbent_pareto_front_maximization_frame': front.tolist(),
        }
        with PUBLIC_INPUTS_PATH.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return payload
    with PUBLIC_INPUTS_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


def public_operating_points(public_inputs, percentile=75):
    return (
        np.asarray(
            public_inputs['reference_point_maximization_frame'], dtype=float
        ),
        np.asarray(
            public_inputs['feasibility_thresholds_by_percentile'][
                str(percentile)
            ],
            dtype=float,
        ),
    )


def candidate_table(cache, scores, indices, physical_max):
    rows = []
    local_pareto = pareto_mask(physical_max[indices])
    for rank, (index, is_pareto) in enumerate(
        zip(indices, local_pareto), start=1
    ):
        cation, anion = cache['candidate_cation_anion'][int(index)]
        rows.append({
            'rank': rank,
            'candidate_index': int(index),
            'cation': cation,
            'anion': anion,
            'il_smiles': f'{cation}.{anion}',
            'fw_aei_score': float(scores[index]),
            'co2_pred': float(cache['physical_mu'][index, 0]),
            'co2_std': float(cache['physical_sigma'][index, 0]),
            'vis_pred': float(cache['physical_mu'][index, 1]),
            'vis_std': float(cache['physical_sigma'][index, 1]),
            'tox_pred': float(cache['physical_mu'][index, 2]),
            'tox_std': float(cache['physical_sigma'][index, 2]),
            'pareto_within_selected_set': bool(is_pareto),
        })
    return pd.DataFrame(rows)


def acquisition_comparison(cache, public_inputs):
    latent_mu = to_maximization(cache['latent_mu'])
    latent_sigma = np.asarray(cache['latent_sigma'], dtype=float)
    physical_mu = to_maximization(cache['physical_mu'])
    physical_sigma = np.asarray(cache['physical_sigma'], dtype=float)
    reference, thresholds = public_operating_points(public_inputs, 75)
    front = np.asarray(
        public_inputs['incumbent_pareto_front_maximization_frame'],
        dtype=float,
    )

    methods = {
        'FW-AEI': fw_aei_scores(
            latent_mu, latent_sigma, reference, thresholds
        ),
        'Additive EI': additive_ei_scores(
            latent_mu, latent_sigma, reference
        ),
    }
    ehvi, n_cells = exact_q1_ehvi_scores(
        latent_mu, latent_sigma, reference, front
    )
    methods['Analytical q=1 EHVI'] = ehvi

    result = {
        'reference_point_maximization_frame': reference.tolist(),
        'feasibility_thresholds_maximization_frame': thresholds.tolist(),
        'n_incumbent_identities': int(
            public_inputs['n_incumbent_identities']
        ),
        'n_incumbent_pareto': int(len(front)),
        'n_ehvi_hypercells': int(n_cells),
        'co2_posterior': cache['co2_posterior'],
    }
    comparison_rows = []
    for name, scores in methods.items():
        chosen = top_indices(scores, 100)
        metrics = selected_set_metrics(
            physical_mu, physical_sigma, chosen, reference
        )
        metrics['top_indices'] = chosen.tolist()
        result[name] = metrics
        comparison_rows.append({'method': name, **{
            key: value for key, value in metrics.items()
            if key != 'top_indices'
        }})

    random_runs = []
    for seed in (42, 123, 7, 2024, 31415):
        chosen = np.random.default_rng(seed).choice(
            len(physical_mu), size=100, replace=False
        )
        metrics = selected_set_metrics(
            physical_mu, physical_sigma, chosen, reference
        )
        metrics['seed'] = seed
        random_runs.append(metrics)
    result['Random (5 seeds)'] = {
        'hypervolume_mean': float(np.mean([
            run['hypervolume'] for run in random_runs
        ])),
        'hypervolume_se': float(np.std([
            run['hypervolume'] for run in random_runs
        ], ddof=1) / np.sqrt(len(random_runs))),
        'n_pareto_mean': float(np.mean([
            run['n_pareto'] for run in random_runs
        ])),
        'mean_sigma': float(np.mean([
            run['mean_sigma'] for run in random_runs
        ])),
        'runs': random_runs,
    }
    comparison_rows.append({
        'method': 'Random (5 seeds)',
        'hypervolume': result['Random (5 seeds)']['hypervolume_mean'],
        'hypervolume_se': result['Random (5 seeds)']['hypervolume_se'],
        'n_pareto': result['Random (5 seeds)']['n_pareto_mean'],
        'mean_sigma': result['Random (5 seeds)']['mean_sigma'],
    })
    pd.DataFrame(comparison_rows).to_csv(
        OUTPUT_DIR / 'acquisition_comparison_revision.csv', index=False
    )

    fw_top = np.asarray(result['FW-AEI']['top_indices'], dtype=int)
    top_table = candidate_table(
        cache, methods['FW-AEI'], fw_top, physical_mu
    )
    top_table.to_csv(OUTPUT_DIR / 'top_candidates_revision.csv', index=False)
    top_table[top_table['pareto_within_selected_set']].to_csv(
        OUTPUT_DIR / 'pareto_candidates_revision.csv', index=False
    )
    return result, methods['FW-AEI'], physical_mu, physical_sigma


def two_vs_three_objectives(
    cache, three_scores, physical_mu, physical_sigma, public_inputs
):
    latent_mu = to_maximization(cache['latent_mu'])
    latent_sigma = np.asarray(cache['latent_sigma'], dtype=float)
    reference, thresholds = public_operating_points(public_inputs, 75)

    top_three = top_indices(three_scores, 50)
    pareto_three_mask = pareto_mask(physical_mu[top_three])
    pareto_three = set(top_three[pareto_three_mask].tolist())

    two_scores = fw_aei_scores(
        latent_mu[:, :2],
        latent_sigma[:, :2],
        reference[:2],
        thresholds[:2],
    )
    top_two = top_indices(two_scores, 50)
    pareto_two_mask = pareto_mask(physical_mu[top_two, :2])
    pareto_two = set(top_two[pareto_two_mask].tolist())

    rows = []
    for setting, top, mask, scores in (
        ('three_objective', top_three, pareto_three_mask, three_scores),
        ('two_objective', top_two, pareto_two_mask, two_scores),
    ):
        for rank, (index, is_pareto) in enumerate(zip(top, mask), start=1):
            cation, anion = cache['candidate_cation_anion'][int(index)]
            rows.append({
                'setting': setting,
                'rank': rank,
                'candidate_index': int(index),
                'score': float(scores[index]),
                'is_pareto_within_top50': bool(is_pareto),
                'cation': cation,
                'anion': anion,
                'co2_pred': float(cache['physical_mu'][index, 0]),
                'vis_pred': float(cache['physical_mu'][index, 1]),
                'tox_pred': float(cache['physical_mu'][index, 2]),
            })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / 'two_vs_three_top50_revision.csv', index=False
    )
    union = pareto_two | pareto_three
    return {
        'three_objective_top50_pareto_count': len(pareto_three),
        'two_objective_top50_pareto_count': len(pareto_two),
        'pareto_overlap_count': len(pareto_two & pareto_three),
        'pareto_union_count': len(union),
        'jaccard': len(pareto_two & pareto_three) / max(len(union), 1),
        'two_objective_pareto_mean_toxicity': float(
            cache['physical_mu'][list(pareto_two), 2].mean()
        ),
        'three_objective_pareto_mean_toxicity': float(
            cache['physical_mu'][list(pareto_three), 2].mean()
        ),
    }


def threshold_sensitivity(
    cache, physical_mu, physical_sigma, public_inputs
):
    latent_mu = to_maximization(cache['latent_mu'])
    latent_sigma = np.asarray(cache['latent_sigma'], dtype=float)
    rows = []
    reference, _ = public_operating_points(public_inputs, 75)
    for percentile in (50, 60, 70, 75, 80, 90):
        _, thresholds = public_operating_points(public_inputs, percentile)
        scores = fw_aei_scores(
            latent_mu, latent_sigma, reference, thresholds
        )
        selected = top_indices(scores, 50)
        metrics = selected_set_metrics(
            physical_mu, physical_sigma, selected, reference
        )
        rows.append({
            'threshold_percentile': percentile,
            'pareto_count_top50': metrics['n_pareto'],
            'hypervolume_top50': metrics['hypervolume'],
            'mean_fw_aei_top50': float(scores[selected].mean()),
            'mean_sigma_top50': metrics['mean_sigma'],
        })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / 'threshold_sensitivity_revision.csv', index=False
    )
    return rows


def main(data_directory=None, output_directory=None):
    global DATA_DIR, OUTPUT_DIR, PUBLIC_INPUTS_PATH
    if data_directory is not None: DATA_DIR = Path(data_directory).resolve()
    if output_directory is not None: OUTPUT_DIR = Path(output_directory).resolve()
    PUBLIC_INPUTS_PATH = OUTPUT_DIR / 'acquisition_inputs_revision.json'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    require_valid_pairs(cache['candidate_cation_anion'], context='acquisition pool')
    public_inputs = load_or_build_public_inputs(cache)
    comparison, three_scores, physical_mu, physical_sigma = (
        acquisition_comparison(cache, public_inputs)
    )
    result = {
        'acquisition_comparison': comparison,
        'two_vs_three_objective': two_vs_three_objectives(
            cache, three_scores, physical_mu, physical_sigma, public_inputs
        ),
        'threshold_sensitivity': threshold_sensitivity(
            cache, physical_mu, physical_sigma, public_inputs
        ),
    }
    with (OUTPUT_DIR / 'acquisition_analysis_revision.json').open(
        'w', encoding='utf-8'
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-directory', type=Path)
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    main(args.data_directory, args.output_directory)
