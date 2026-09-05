"""Held-out prediction diagnostics; no test-driven fitting or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cf_bild.predictive import zero_truncated_normal_quantile


def main(run):
    run = Path(run).resolve(); out = run / 'analysis'
    summary = {'protocol': 'Descriptive diagnostics of frozen primary product-GP held-out predictions',
               'bootstrap_seed': 42, 'bootstrap_resamples': 10000, 'properties': {}}
    levels = np.linspace(0, 1, 101)
    for prop in ['co2', 'vis', 'tox']:
        file = out / f'test_predictions_{prop}.csv'
        pred = pd.read_csv(file)
        data = pd.read_csv(run / f'data/test_group_{prop}.csv')
        merged = pred.merge(data[['ind', 'group']], left_on='source_record_id', right_on='ind', validate='one_to_one')
        if len(merged) != len(pred) or len(pred) != len(data): raise ValueError('Test IDs do not match exactly')
        species = merged.groupby('group')[['y_true', 'pred_mean']].mean().reset_index()
        species.to_csv(out / f'test_species_predictions_{prop}.csv', index=False)
        actual, estimate = species.y_true.to_numpy(), species.pred_mean.to_numpy()
        rng = np.random.default_rng(42)
        draws = rng.integers(0, len(species), size=(10000, len(species)))
        truth, guess = actual[draws], estimate[draws]
        denominator = ((truth-truth.mean(1, keepdims=True))**2).sum(1)
        valid = denominator > 0
        scores = 1-((truth-guess)**2).sum(1)[valid]/denominator[valid]
        mu, sd = pred.latent_mu.to_numpy(), pred.latent_std.to_numpy()
        coverage = []
        for level in levels:
            if level == 0: coverage.append(0.0); continue
            if level == 1: coverage.append(1.0); continue
            tail = (1-level)/2
            if prop == 'co2':
                lo = zero_truncated_normal_quantile(mu, sd**2, tail)
                hi = zero_truncated_normal_quantile(mu, sd**2, 1-tail)
            else:
                lo, hi = mu+sd*norm.ppf(tail), mu+sd*norm.ppf(1-tail)
            coverage.append(float(((pred.y_true >= lo) & (pred.y_true <= hi)).mean()))
        pd.DataFrame({'nominal_coverage': levels, 'empirical_coverage': coverage}).to_csv(out / f'calibration_curve_{prop}.csv', index=False)
        summary['properties'][prop] = {'n_records': len(pred), 'n_species': len(species),
            'species_averaged_r2': float(r2_score(actual, estimate)),
            'species_averaged_rmse': float(np.sqrt(mean_squared_error(actual, estimate))),
            'species_bootstrap_r2_ci95': np.quantile(scores, [.025, .975]).tolist(),
            'bootstrap_valid_resamples': int(valid.sum()),
            'miscalibration_area_101_levels': float(np.trapz(np.abs(np.array(coverage)-levels), levels)),
            'untruncated_negative_mean_count': int((mu < 0).sum()),
            'reported_negative_mean_count': int((pred.pred_mean < 0).sum()),
            'prediction_source_sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    (out / 'prediction_diagnostics_clean.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    main(parser.parse_args().run_directory)
