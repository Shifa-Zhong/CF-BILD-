"""Retune and refit GPs on an explicitly curated, hash-identified data run.

No old fitted parameters, calibration factors or prediction caches are used.
Hyperopt search is checkpointed after each trial. The held-out test partition
is never used in selection, fold repair, stopping or variance calibration.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cf_bild.fragment_vocab import FragmentVocabulary, load_property_datasets, prepare_cv_splits
from cf_bild.gp_cvloss import GPCrossValidatedOptimizer
from cf_bild.ion_validation import require_valid_pairs
from cf_bild.predictive import physical_prediction, regression_metrics, viscosity_real_space_metrics
from run_revision_models import candidate_features, candidate_pairs, cpu_state_dict, conditional_nlpd, environment_versions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def vocabulary(run, datasets):
    path = run / 'fragment_vocab.pkl'
    vocab = FragmentVocabulary()
    if path.exists():
        vocab.load(path)
    else:
        frames = [frame for prop in ('co2', 'vis', 'tox') for frame in
                  [*datasets[prop]['folds'][0], datasets[prop]['test']]]
        vocab.extract_fragments_from_dataframes(frames)
        vocab.compute_fingerprints()
        vocab.save(path)
    pairs = candidate_pairs(vocab)
    require_valid_pairs(pairs, context='cleaned candidate pool')
    write_json(run / 'VOCABULARY_SUMMARY.json', {
        'n_cations': len(vocab.cations), 'n_anions': len(vocab.anions), 'n_candidates': len(pairs),
        'cation_dimensions': vocab.cat_fp_length, 'anion_dimensions': vocab.an_fp_length,
        'cation_nominal_bits': vocab.cfmf_cat.length, 'anion_nominal_bits': vocab.cfmf_an.length,
        'source': 'union of all retained property structures, including held-out structures but not test targets',
        'contract': '+1/-1 connected constituent ions; formally charge-balanced 1:1 pairs',
        'vocabulary_sha256': digest(path),
    })
    return vocab


def optimize_checkpointed(optimizer, directory, run_signature, max_evals, patience, seed):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'search_checkpoint.pkl'
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if checkpoint.exists():
        with checkpoint.open('rb') as handle:
            saved = pickle.load(handle)  # Only this run's own checkpoint.
        if saved['signature'] != run_signature:
            raise ValueError('Checkpoint inputs/protocol/code changed; start a new run')
        optimizer.trials = saved['trials']
        optimizer._iteration_count = len(optimizer.trials)
        rng.bit_generator.state = saved['numpy_rng_state']
        torch.set_rng_state(saved['torch_rng_state'])
        if torch.cuda.is_available() and saved['cuda_rng_state']:
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    started = time.time()

    def checkpoint_and_stop(trials, *unused):
        losses = np.asarray([trial['result']['loss'] for trial in trials.trials])
        finite = np.flatnonzero(np.isfinite(losses))
        if not len(finite):
            if len(losses) >= 5:
                raise RuntimeError('Five non-finite CV trials; diagnose before continuing')
            best_index, best_loss, best_params = None, None, None
            stop = False
        else:
            best_index = int(finite[np.argmin(losses[finite])])
            best_loss = float(losses[best_index])
            best_params = trials.trials[best_index]['result']['params']
            stop = len(losses) - 1 - best_index >= patience
        progress = {'status': 'selection_complete' if stop or len(losses) >= max_evals else 'selecting',
            'n_trials': len(losses), 'max_evals': max_evals, 'patience': patience,
            'best_trial': None if best_index is None else best_index + 1,
            'best_cv_rmse': best_loss, 'best_params': best_params,
            'process_elapsed_seconds': time.time() - started, 'test_used_in_selection': False}
        write_json(directory / 'SEARCH_PROGRESS.json', progress)
        payload = {'signature': run_signature, 'trials': trials, 'numpy_rng_state': rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temporary = checkpoint.with_suffix('.pkl.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint)
        print(f'Trial {len(losses)}: best CV RMSE={best_loss}; best trial={progress["best_trial"]}', flush=True)
        return stop, []

    already_stopped = False
    if len(optimizer.trials):
        already_stopped, _ = checkpoint_and_stop(optimizer.trials)
    if not already_stopped and len(optimizer.trials) < max_evals:
        fmin(optimizer._objective, optimizer.hyperopt_space, algo=tpe.suggest,
             max_evals=max_evals, trials=optimizer.trials, rstate=rng,
             early_stop_fn=checkpoint_and_stop, show_progressbar=False)
    optimizer.best_params = dict(optimizer.trials.best_trial['result']['params'])
    optimizer._calibrate_variance()
    optimizer.refit_best_model()
    trial_rows = [{'trial': i+1, 'cv_rmse': trial['result']['loss'],
                   'train_rmse': trial['result']['train_loss'], **trial['result']['params']}
                  for i, trial in enumerate(optimizer.trials.trials)]
    pd.DataFrame(trial_rows).to_csv(directory / 'hyperopt_trials.csv', index=False)


def fit_property(run, datasets, vocab, prop, form, max_evals, patience, seed, batch_size):
    result_dir = run / 'results' / form
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir = result_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    started = time.time()
    cv, x, y, xt, yt, scaler = prepare_cv_splits(datasets, vocab, prop)
    assert len(yt) == len(datasets[prop]['test'])
    assert len(y) == sum(len(f) for f in datasets[prop]['folds'][0])
    sources = {path.name: digest(path) for path in sorted((run / 'data').glob(f'*_group_{prop}.csv'))}
    signature = {'inputs': sources, 'vocabulary': digest(run / 'fragment_vocab.pkl'), 'form': form,
        'max_evals': max_evals, 'patience': patience, 'seed': seed,
        'optimizer_code': digest(ROOT / 'cf_bild/gp_cvloss.py'),
        'runner_code': digest(Path(__file__)), 'feature_code': digest(ROOT / 'cf_bild/fragment_vocab.py')}
    manifest_path = result_dir / f'fit_manifest_{prop}.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['signature'] != signature:
            raise ValueError('Completed fit differs from requested inputs or protocol')
        if all((result_dir / name).exists() and digest(result_dir / name) == checksum
               for name, checksum in previous['artifacts_sha256'].items()):
            print(f'Verified completed {form}/{prop}; no recomputation', flush=True)
            return json.loads((result_dir / f'metrics_{prop}.json').read_text())
        raise ValueError('Completed fit artifact checksum mismatch')
    optimizer = GPCrossValidatedOptimizer(x, y, predefined_cv_splits=cv,
        compositional_kernel_dims=(vocab.cat_fp_length, vocab.an_fp_length) if form != 'standard' else None,
        kernel_form=form, random_state=seed)
    optimize_checkpointed(optimizer, result_dir / f'search_{prop}', signature, max_evals, patience, seed)
    mu, var = optimizer.predict(xt)
    mu, var = mu.ravel(), var.ravel()
    physical = physical_prediction(prop, mu, var)
    metrics = regression_metrics(yt, physical)
    metrics.update({'nlpd': conditional_nlpd(prop, yt, mu, var), 'n_refit': len(y), 'n_test': len(yt),
        'variance_scale': float(optimizer.variance_scale_), 'n_search_trials': len(optimizer.trials),
        'best_cv_rmse': float(optimizer.trials.best_trial['result']['loss'])})
    if prop == 'vis':
        metrics['real_space'] = viscosity_real_space_metrics(yt, physical['mean'])
    table = pd.DataFrame({'source_record_id': datasets[prop]['test']['ind'].to_numpy(),
        'y_true': yt, 'latent_mu': mu, 'latent_std': np.sqrt(var), 'pred_mean': physical['mean'],
        'pred_std': physical['std'], 'lower_95': physical['lower'], 'upper_95': physical['upper']})
    table.to_csv(result_dir / f'test_predictions_{prop}.csv', index=False)
    np.savez_compressed(result_dir / f'calibration_{prop}.npz',
        absolute_residual=optimizer.calibration_residuals_, raw_std=optimizer.calibration_raw_std_)
    write_json(result_dir / f'calibration_{prop}.json', optimizer.calibration_diagnostics_)
    best = {'best_params': optimizer.best_params, 'variance_scale': float(optimizer.variance_scale_),
        'effective_noise_variance': max(float(optimizer.best_params['likelihood_noise_variance']), 1e-4),
        'environment_kernel': optimizer.best_params['kernel_name'] if prop in ('co2', 'vis') else None,
        'selection': 'new five-fold species-disjoint CV on curated non-test records only'}
    write_json(result_dir / f'selected_hyperparameters_{prop}.json', best)
    state = {**best, 'property': prop, 'kernel_form': form, 'scaler': scaler,
        'y_train_mean': float(optimizer.y_train_mean_), 'n_refit': len(y),
        'dim_cat': vocab.cat_fp_length, 'dim_an': vocab.an_fp_length,
        'model_state': cpu_state_dict(optimizer.best_model_),
        'likelihood_state': cpu_state_dict(optimizer.best_likelihood_)}
    with (model_dir / f'model_{prop}.pkl').open('wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if form == 'product':
        pieces = {key: [] for key in ('latent_mu', 'latent_sigma', 'physical_mu', 'physical_sigma')}
        count = len(vocab.cations) * len(vocab.anions)
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            raw = candidate_features(vocab, prop, start, end)
            batch_mu, batch_var = optimizer.predict(scaler.transform(raw))
            batch_mu, batch_var = batch_mu.ravel(), batch_var.ravel()
            physical_batch = physical_prediction(prop, batch_mu, batch_var)
            for key, values in [('latent_mu', batch_mu), ('latent_sigma', np.sqrt(batch_var)),
                                ('physical_mu', physical_batch['mean']), ('physical_sigma', physical_batch['std'])]:
                if not np.all(np.isfinite(values)):
                    raise ValueError('Non-finite candidate predictions')
                pieces[key].append(values)
        np.savez_compressed(result_dir / f'candidate_predictions_{prop}.npz',
                            **{key: np.concatenate(values) for key, values in pieces.items()})
    metrics['elapsed_seconds'] = time.time() - started
    write_json(result_dir / f'metrics_{prop}.json', metrics)
    artifacts = [f'metrics_{prop}.json', f'test_predictions_{prop}.csv', f'calibration_{prop}.npz',
        f'calibration_{prop}.json', f'selected_hyperparameters_{prop}.json', f'models/model_{prop}.pkl']
    if form == 'product': artifacts.append(f'candidate_predictions_{prop}.npz')
    write_json(manifest_path, {'signature': signature,
        'artifacts_sha256': {name: digest(result_dir / name) for name in artifacts}})
    print(json.dumps({'property': prop, 'form': form, 'metrics': metrics}, indent=2), flush=True)
    del optimizer, x, y, xt, cv
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--properties', nargs='+', choices=['co2', 'vis', 'tox'], default=['tox', 'co2', 'vis'])
    parser.add_argument('--kernel-forms', nargs='+', choices=['product', 'additive', 'product_no_cross', 'additive_no_cross', 'standard'], default=['product'])
    parser.add_argument('--max-evals', type=int, default=3000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=2000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run = args.run_directory.resolve()
    curation = json.loads((run / 'CURATION_SUMMARY.json').read_text())
    if curation['status'] != 'cleaned_inputs_ready': raise ValueError('Cleaned input approval is missing')
    if set(curation['properties']) != {'co2', 'vis', 'tox'}:
        raise ValueError('Complete all three property cleanups before freezing the shared vocabulary and fitting')
    for name, checksum in curation['cleaned_sha256'].items():
        if digest(run / 'data' / name) != checksum: raise ValueError(f'Cleaned input changed: {name}')
    write_json(run / 'COMPUTATIONAL_ENVIRONMENT.json', environment_versions())
    datasets = load_property_datasets(run / 'data', n_folds=5)
    vocab = vocabulary(run, datasets)
    for form in args.kernel_forms:
        for prop in args.properties:
            fit_property(run, datasets, vocab, prop, form, args.max_evals, args.patience, args.seed, args.batch_size)
    print('Requested curated-data fits completed.', flush=True)


if __name__ == '__main__':
    main()
