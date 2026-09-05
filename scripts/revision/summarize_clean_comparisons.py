"""Create scale-explicit acquisition and matched-pool model comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(run):
    run = Path(run).resolve()
    out = run / 'analysis'
    analyses = json.loads((out / 'acquisition_analysis_revision.json').read_text())
    with np.load(out / 'candidate_predictions.npz', allow_pickle=False) as arrays:
        sigma = arrays['physical_sigma']
        latent_mu, physical_mu = arrays['latent_mu'], arrays['physical_mu']
    comparison = analyses['acquisition_comparison']
    reference_sigma = sigma[comparison['Analytical q=1 EHVI']['top_indices']].mean(axis=0)
    rows = []
    for name in ['FW-AEI', 'Additive EI', 'Analytical q=1 EHVI', 'Random (5 seeds)']:
        if name.startswith('Random'):
            values = np.stack([sigma[np.random.default_rng(seed).choice(len(sigma), 100, replace=False)].mean(0)
                               for seed in [42, 123, 7, 2024, 31415]])
            mean = values.mean(0)
        else:
            mean = sigma[comparison[name]['top_indices']].mean(0)
        rows.append({'method': name, **{f'{p}_mean_sigma': float(value) for p, value in zip(['co2', 'vis', 'tox'], mean)},
                     **{f'{p}_ratio_to_ehvi': float(value) for p, value in zip(['co2', 'vis', 'tox'], mean/reference_sigma)}})
    pd.DataFrame(rows).to_csv(out / 'acquisition_property_uncertainty.csv', index=False)
    model_rows = []
    missing = []
    for form in ['product', 'additive', 'standard', 'product_no_cross', 'additive_no_cross']:
        for prop in ['co2', 'vis', 'tox']:
            path = run / f'results/{form}/metrics_{prop}.json'
            if not path.exists():
                missing.append(f'{form}/{prop}'); continue
            metrics = json.loads(path.read_text())
            model_rows.append({'model': form, 'property': prop, 'conditioning_pool': 'complete curated non-test pool', **metrics})
    baseline = json.loads((out / 'median_heuristic_baseline.json').read_text())
    # Earlier metadata names counted unique feature rows, not chemical IDs.
    # Relabel only that metadata; preserve every fitted parameter and metric.
    for details in baseline['properties'].values():
        for old, new in [('unique_cations', 'unique_cation_feature_rows'), ('unique_anions', 'unique_anion_feature_rows'), ('unique_ion_pairs', 'unique_pair_feature_rows')]:
            if old in details: details[new] = details.pop(old)
    (out / 'median_heuristic_baseline.json').write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    for prop, details in baseline['properties'].items():
        model_rows.append({'model': 'median_heuristic', 'property': prop,
            'conditioning_pool': 'complete curated non-test pool', **details['metrics'], 'variance_scale': details['variance_scale']})
    pd.DataFrame(model_rows).drop(columns=['real_space'], errors='ignore').to_csv(out / 'model_comparison_clean.csv', index=False)
    fw, ehvi = comparison['FW-AEI']['hypervolume'], comparison['Analytical q=1 EHVI']['hypervolume']
    summary = {'model_comparisons_complete': not missing, 'missing_model_comparisons': missing,
        'fw_aei_hypervolume': fw, 'ehvi_hypervolume': ehvi, 'fw_aei_hypervolume_change_percent': 100*(fw/ehvi-1),
        'fw_aei_uncertainty_ratios': {p: rows[0][f'{p}_ratio_to_ehvi'] for p in ['co2', 'vis', 'tox']},
        'fw_aei_has_lower_uncertainty_in_every_property': all(rows[0][f'{p}_ratio_to_ehvi'] < 1 for p in ['co2', 'vis', 'tox']),
        'candidate_co2_support': {'negative_untruncated_means': int((latent_mu[:, 0] < 0).sum()),
            'negative_reported_means': int((physical_mu[:, 0] < 0).sum()),
            'reported_means_above_one': int((physical_mu[:, 0] > 1).sum()),
            'reported_mean_min': float(physical_mu[:, 0].min()), 'reported_mean_max': float(physical_mu[:, 0].max())},
        'mean_sigma_across_mixed_property_units_used_for_claims': False}
    (out / 'clean_comparison_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    main(parser.parse_args().run_directory)
