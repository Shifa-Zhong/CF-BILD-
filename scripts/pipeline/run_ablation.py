"""
CF-BILD Ablation Studies
A: Kernel form — product vs additive vs standard
B: GP-BT vs standard GP (with vs without CV-based hyperparameter selection)
C: Acquisition function — C-EHVI vs standard EHVI vs random
"""

import os, sys, json, time, pickle
import numpy as np
import pandas as pd
import logging
from hyperopt import STATUS_OK

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('Ablation')
logger.setLevel(logging.INFO)

# Import torch before rdkit
from bayesian_gp_cvloss import GPCrossValidatedOptimizer
from cf_bild.fragment_vocab import (
    FragmentVocabulary, load_property_datasets, prepare_cv_splits
)
from cf_bild.conservative_bo import (
    compute_c_ehvi_scores, identify_pareto_front, compute_hypervolume
)

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')


def make_early_stop_fn(patience=50):
    def fn(trials, best_loss=None, count=0):
        if len(trials) == 0:
            return False, [best_loss, count]
        cur = min(t['result']['loss'] for t in trials.trials
                  if t['result']['status'] == STATUS_OK)
        if best_loss is None or cur < best_loss - 1e-6:
            return False, [cur, 0]
        count += 1
        return (count >= patience, [best_loss, count])
    return fn


def train_and_evaluate(datasets, vocab, prop, kernel_form="product",
                       compositional=True, max_evals=3000, patience=50):
    """Train GP-BT for one property and return test metrics."""
    dim_cat = vocab.cat_fp_length
    dim_an = vocab.an_fp_length

    cv_splits, X_all, y_all, X_test, y_test, scaler = \
        prepare_cv_splits(datasets, vocab, prop)

    comp_dims = (dim_cat, dim_an) if compositional else None
    kf = kernel_form if compositional else "standard"

    optimizer = GPCrossValidatedOptimizer(
        X_train=X_all, y_train=y_all,
        kernels=None, n_splits=5, random_state=42,
        predefined_cv_splits=cv_splits,
        compositional_kernel_dims=comp_dims,
        kernel_form=kf,
    )

    early_stop = make_early_stop_fn(patience=patience)
    optimizer.optimize(max_evals=max_evals, early_stop_fn=early_stop)

    if optimizer.best_params is None:
        return {'R2': float('nan'), 'RMSE': float('nan'),
                'CI_coverage': float('nan'), 'NLPD': float('nan'),
                'CV_RMSE': float('nan')}

    trials = optimizer.get_optimization_results()
    cv_rmse = trials.best_trial['result']['loss']

    metrics = {'CV_RMSE': cv_rmse}

    if X_test is not None:
        pred_mean, pred_var = optimizer.predict(X_test)
        if pred_mean is not None:
            pred = pred_mean.flatten()
            std = np.sqrt(pred_var.flatten())

            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            metrics['R2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics['RMSE'] = np.sqrt(np.mean((y_test - pred) ** 2))

            lower = pred - 1.96 * std
            upper = pred + 1.96 * std
            metrics['CI_coverage'] = np.mean((y_test >= lower) & (y_test <= upper))

            var = pred_var.flatten()
            metrics['NLPD'] = 0.5 * np.mean(
                np.log(2 * np.pi * var) + (y_test - pred) ** 2 / var)

    return metrics, optimizer, scaler


def ablation_a_kernel_form(datasets, vocab):
    """Ablation A: product vs additive vs standard kernel."""
    logger.info("=" * 60)
    logger.info("ABLATION A: Kernel Form")
    logger.info("=" * 60)

    results = {}
    for kernel_form in ['product', 'additive', 'standard']:
        results[kernel_form] = {}
        for prop in ['co2', 'vis', 'tox']:
            logger.info(f"  {kernel_form}/{prop}...")
            compositional = kernel_form != 'standard'
            metrics, _, _ = train_and_evaluate(
                datasets, vocab, prop,
                kernel_form=kernel_form,
                compositional=compositional
            )
            results[kernel_form][prop] = metrics
            logger.info(f"    R2={metrics.get('R2', 'N/A'):.4f} "
                        f"RMSE={metrics.get('RMSE', 'N/A'):.4f} "
                        f"CV={metrics.get('CV_RMSE', 'N/A'):.4f}")

    return results


def ablation_b_gpbt_vs_standard(datasets, vocab):
    """Ablation B: GP-BT (CV-optimized) vs standard GP (default params).

    Standard GP: use median lengthscale, unit variance, median noise.
    GP-BT: full Hyperopt optimization.
    """
    logger.info("=" * 60)
    logger.info("ABLATION B: GP-BT vs Standard GP")
    logger.info("=" * 60)

    results = {'GP-BT': {}, 'Standard-GP': {}}
    dim_cat = vocab.cat_fp_length
    dim_an = vocab.an_fp_length

    for prop in ['co2', 'vis', 'tox']:
        # GP-BT (already have results from ablation A product)
        logger.info(f"  GP-BT/{prop}...")
        metrics_bt, _, _ = train_and_evaluate(
            datasets, vocab, prop, kernel_form="product", compositional=True)
        results['GP-BT'][prop] = metrics_bt

        # Standard GP: use fixed default hyperparameters (no optimization)
        logger.info(f"  Standard-GP/{prop}...")
        cv_splits, X_all, y_all, X_test, y_test, scaler = \
            prepare_cv_splits(datasets, vocab, prop)

        # Use default params: median-range lengthscale, unit variance
        default_params = {
            'kernel_name': 'RBF',
            'ls_cat': 10.0,
            'ls_an': 10.0,
            'ls_cross': 1.0,
            'kernel_variance': float(np.var(y_all)),
            'likelihood_noise_variance': float(np.var(y_all) * 0.1),
        }
        has_env = prop in ['co2', 'vis']
        if has_env:
            default_params['ls_env_0'] = 1.0
            default_params['ls_env_1'] = 1.0

        optimizer = GPCrossValidatedOptimizer(
            X_train=X_all, y_train=y_all,
            kernels=None, n_splits=5, random_state=42,
            predefined_cv_splits=cv_splits,
            compositional_kernel_dims=(dim_cat, dim_an),
            kernel_form="product",
        )
        # Manually set best_params without optimization
        optimizer.best_params = default_params
        optimizer.variance_scale_ = 1.0
        optimizer.refit_best_model()

        metrics_std = {}
        if X_test is not None:
            pred_mean, pred_var = optimizer.predict(X_test)
            if pred_mean is not None:
                pred = pred_mean.flatten()
                std = np.sqrt(pred_var.flatten())
                ss_res = np.sum((y_test - pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                metrics_std['R2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                metrics_std['RMSE'] = np.sqrt(np.mean((y_test - pred) ** 2))
                lower = pred - 1.96 * std
                upper = pred + 1.96 * std
                metrics_std['CI_coverage'] = np.mean((y_test >= lower) & (y_test <= upper))
                var = pred_var.flatten()
                metrics_std['NLPD'] = 0.5 * np.mean(
                    np.log(2 * np.pi * var) + (y_test - pred) ** 2 / var)

        results['Standard-GP'][prop] = metrics_std
        logger.info(f"    R2={metrics_std.get('R2', 'N/A'):.4f}")

    return results


def ablation_c_acquisition(models, all_stats, vocab):
    """Ablation C: C-EHVI vs standard EHVI vs random search."""
    logger.info("=" * 60)
    logger.info("ABLATION C: Acquisition Function")
    logger.info("=" * 60)

    # Load predictions from saved models
    all_mu, all_sigma = {}, {}
    cation_list = list(vocab.cation_fps.keys())
    anion_list = list(vocab.anion_fps.keys())
    n_cat, n_an = len(cation_list), len(anion_list)
    n_candidates = n_cat * n_an

    cat_fps_all = np.array([vocab.cation_fps[c] for c in cation_list], dtype=np.float64)
    an_fps_all = np.array([vocab.anion_fps[a] for a in anion_list], dtype=np.float64)

    for prop in ['co2', 'vis', 'tox']:
        optimizer = models[prop]
        scaler = all_stats[prop]['scaler']
        mu_list, sigma_list = [], []

        batch_size = 5000
        for start in range(0, n_candidates, batch_size):
            end = min(start + batch_size, n_candidates)
            cat_idx = np.arange(start, end) // n_an
            an_idx = np.arange(start, end) % n_an
            X_batch = np.concatenate([cat_fps_all[cat_idx], an_fps_all[an_idx]], axis=1)
            if prop in ['co2', 'vis']:
                env = np.full((end - start, 2), [298.15, 101.325], dtype=np.float64)
                X_batch = np.concatenate([X_batch, env], axis=1)
            X_batch = scaler.transform(X_batch)
            pm, pv = optimizer.predict(X_batch)
            mu_list.append(pm.flatten())
            sigma_list.append(np.sqrt(pv.flatten()))

        all_mu[prop] = np.concatenate(mu_list)
        all_sigma[prop] = np.concatenate(sigma_list)

    mu_matrix = np.column_stack([all_mu['co2'], all_mu['vis'], all_mu['tox']])
    sigma_matrix = np.column_stack([all_sigma['co2'], all_sigma['vis'], all_sigma['tox']])

    # Thresholds
    thresh = np.array([
        np.percentile(models['co2'].y_train_1d, 75),
        np.percentile(models['vis'].y_train_1d, 25),
        np.percentile(models['tox'].y_train_1d, 75),
    ])
    ref_point = np.array([
        models['co2'].y_train_1d.min(),
        models['vis'].y_train_1d.max(),
        models['tox'].y_train_1d.min(),
    ])
    maximize_flags = [True, False, True]

    results = {}

    # 1. C-EHVI
    scores_cehvi = compute_c_ehvi_scores(
        {'mu': mu_matrix, 'sigma': sigma_matrix}, thresh, ref_point, maximize_flags)
    top_idx = np.argsort(scores_cehvi)[::-1][:100]
    pareto_mask = identify_pareto_front(mu_matrix[top_idx], maximize_flags)
    hv = compute_hypervolume(mu_matrix[top_idx][pareto_mask], ref_point, maximize_flags)
    results['C-EHVI'] = {'n_pareto': int(pareto_mask.sum()), 'hypervolume': hv,
                          'mean_co2': float(mu_matrix[top_idx, 0].mean()),
                          'mean_sigma': float(sigma_matrix[top_idx].mean())}

    # 2. Standard EHVI (no conservative penalty)
    from scipy.stats import norm
    total_ei = np.zeros(n_candidates)
    mu_max = mu_matrix.copy()
    mu_max[:, 1] = -mu_max[:, 1]  # flip vis for maximization
    for k in range(3):
        z = (mu_max[:, k] - ref_point[k]) / (sigma_matrix[:, k] + 1e-8)
        if not maximize_flags[k]:
            z = -z
        total_ei += sigma_matrix[:, k] * (z * norm.cdf(z) + norm.pdf(z))

    top_idx_ehvi = np.argsort(total_ei)[::-1][:100]
    pareto_ehvi = identify_pareto_front(mu_matrix[top_idx_ehvi], maximize_flags)
    hv_ehvi = compute_hypervolume(mu_matrix[top_idx_ehvi][pareto_ehvi], ref_point, maximize_flags)
    results['EHVI'] = {'n_pareto': int(pareto_ehvi.sum()), 'hypervolume': hv_ehvi,
                        'mean_co2': float(mu_matrix[top_idx_ehvi, 0].mean()),
                        'mean_sigma': float(sigma_matrix[top_idx_ehvi].mean())}

    # 3. Random search
    rng = np.random.RandomState(42)
    random_idx = rng.choice(n_candidates, 100, replace=False)
    pareto_rand = identify_pareto_front(mu_matrix[random_idx], maximize_flags)
    hv_rand = compute_hypervolume(mu_matrix[random_idx][pareto_rand], ref_point, maximize_flags)
    results['Random'] = {'n_pareto': int(pareto_rand.sum()), 'hypervolume': hv_rand,
                          'mean_co2': float(mu_matrix[random_idx, 0].mean()),
                          'mean_sigma': float(sigma_matrix[random_idx].mean())}

    for method, r in results.items():
        logger.info(f"  {method}: Pareto={r['n_pareto']}, HV={r['hypervolume']:.4f}, "
                    f"mean_CO2={r['mean_co2']:.4f}, mean_sigma={r['mean_sigma']:.4f}")

    return results


def main():
    start_time = time.time()
    logger.info("CF-BILD Ablation Studies")

    # Load data
    datasets = load_property_datasets(DATA_DIR, n_folds=5)
    vocab = FragmentVocabulary(fp_radius=2)
    all_dfs = []
    for prop, data in datasets.items():
        for t, v in data['folds']:
            all_dfs.extend([t, v])
        if data['test'] is not None:
            all_dfs.append(data['test'])
    vocab.extract_fragments_from_dataframes(all_dfs)
    vocab.compute_fingerprints()

    all_results = {}

    # Ablation A: Kernel form
    logger.info("\n>>> ABLATION A: Kernel Form <<<")
    all_results['A_kernel_form'] = ablation_a_kernel_form(datasets, vocab)

    # Ablation B: GP-BT vs Standard GP
    logger.info("\n>>> ABLATION B: GP-BT vs Standard GP <<<")
    all_results['B_surrogate'] = ablation_b_gpbt_vs_standard(datasets, vocab)

    # For Ablation C, need trained models — use product kernel results
    logger.info("\n>>> ABLATION C: Acquisition Function <<<")
    # Train product models for screening
    models = {}
    all_stats = {}
    for prop in ['co2', 'vis', 'tox']:
        metrics, optimizer, scaler = train_and_evaluate(
            datasets, vocab, prop, kernel_form="product", compositional=True)
        models[prop] = optimizer
        all_stats[prop] = {'scaler': scaler}
    all_results['C_acquisition'] = ablation_c_acquisition(models, all_stats, vocab)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info("=" * 60)

    logger.info("\nA. Kernel Form:")
    logger.info(f"{'Kernel':<12} {'Property':<8} {'R²':<8} {'RMSE':<8} {'CI':<8} {'CV_RMSE':<8}")
    for kf in ['product', 'additive', 'standard']:
        for prop in ['co2', 'vis', 'tox']:
            m = all_results['A_kernel_form'][kf][prop]
            logger.info(f"{kf:<12} {prop:<8} {m.get('R2',0):<8.4f} "
                        f"{m.get('RMSE',0):<8.4f} {m.get('CI_coverage',0):<8.4f} "
                        f"{m.get('CV_RMSE',0):<8.4f}")

    logger.info("\nB. GP-BT vs Standard GP:")
    for method in ['GP-BT', 'Standard-GP']:
        for prop in ['co2', 'vis', 'tox']:
            m = all_results['B_surrogate'][method][prop]
            logger.info(f"{method:<15} {prop:<8} R2={m.get('R2',0):.4f} "
                        f"CI={m.get('CI_coverage',0):.4f}")

    logger.info("\nC. Acquisition Function:")
    for method in ['C-EHVI', 'EHVI', 'Random']:
        r = all_results['C_acquisition'][method]
        logger.info(f"{method:<10} Pareto={r['n_pareto']:<4} HV={r['hypervolume']:<10.4f} "
                    f"mean_sigma={r['mean_sigma']:.4f}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nResults saved to output/ablation_results.json")

    elapsed = time.time() - start_time
    logger.info(f"Ablation studies completed in {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
