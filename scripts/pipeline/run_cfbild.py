"""
CF-BILD: Conservative Fragment-Based Bayesian Ionic Liquid Design
Main training and optimization script.

Uses GP-BT (bayesian-gp-cvloss) with compositional kernel and
predefined group-based CV splits to prevent data leakage.

Usage:
    python run_cfbild.py
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK

# Setup paths — script lives in scripts/pipeline/, project root is two levels up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('CF-BILD')

# CRITICAL: Import torch/GPyTorch BEFORE rdkit to avoid DLL conflicts on Windows
from bayesian_gp_cvloss import GPCrossValidatedOptimizer

from cf_bild.fragment_vocab import (
    FragmentVocabulary, load_property_datasets, prepare_cv_splits, df_to_features
)
from cf_bild.conservative_bo import (
    compute_thresholds_from_training, compute_c_ehvi_scores,
    select_batch, identify_pareto_front, compute_hypervolume
)


def phase1_fragment_vocabulary():
    """Phase 1: Build fragment vocabulary from property datasets."""
    print("=" * 60)
    print("PHASE 1: Fragment Vocabulary Construction")
    print("=" * 60)

    datasets = load_property_datasets(DATA_DIR, n_folds=5)
    vocab = FragmentVocabulary(fp_radius=2)

    # Collect all DataFrames for fragment extraction
    all_dfs = []
    for prop, data in datasets.items():
        for train_df, val_df in data['folds']:
            all_dfs.extend([train_df, val_df])
        if data['test'] is not None:
            all_dfs.append(data['test'])

    vocab.extract_fragments_from_dataframes(all_dfs)
    vocab.compute_fingerprints()
    vocab.save(os.path.join(OUTPUT_DIR, 'fragment_vocab.pkl'))

    return datasets, vocab


def make_early_stop_fn(patience=50):
    """Create an early stopping function for Hyperopt.

    Stops optimization if no improvement in best loss for `patience` consecutive trials.
    """
    def early_stop_fn(trials, best_loss=None, no_progress_count=0):
        if len(trials) == 0:
            return False, [best_loss, no_progress_count]
        current_best = min(t['result']['loss'] for t in trials.trials
                           if t['result']['status'] == STATUS_OK)
        if best_loss is None or current_best < best_loss - 1e-6:
            return False, [current_best, 0]
        else:
            no_progress_count += 1
            if no_progress_count >= patience:
                logger.info(f"Early stopping: no improvement for {patience} iterations.")
                return True, [best_loss, no_progress_count]
            return False, [best_loss, no_progress_count]
    return early_stop_fn


def phase2_train_gp_models(datasets, vocab, kernel_form="product",
                           max_evals=3000, early_stop_patience=50):
    """Phase 2: Train Compositional GP-BT for each property.

    Uses predefined group-based CV splits + compositional kernel + GP-BT.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Compositional GP-BT Training")
    print("=" * 60)

    dim_cat = vocab.cat_fp_length
    dim_an = vocab.an_fp_length
    print(f"CF-MF dimensions: cation={dim_cat}, anion={dim_an}, total={dim_cat+dim_an}")

    models = {}
    all_stats = {}

    for prop in ['co2', 'vis', 'tox']:
        print(f"\n--- Training GP-BT for: {prop} (kernel_form={kernel_form}) ---")

        # Prepare predefined CV splits (StandardScaler applied per fold)
        cv_splits, X_all_train, y_all_train, X_test, y_test, scaler = \
            prepare_cv_splits(datasets, vocab, prop)

        # Store stats for later use
        y_mean = y_all_train.mean()
        y_std = y_all_train.std()
        if y_std < 1e-8:
            y_std = 1.0
        all_stats[prop] = {'y_mean': y_mean, 'y_std': y_std, 'scaler': scaler}

        print(f"    y_mean={y_mean:.4f}, y_std={y_std:.4f}")

        # Use full training data (RTX 5090 32GB VRAM can handle it)
        X_all_train_sub = X_all_train
        y_all_train_sub = y_all_train
        cv_splits_sub = cv_splits

        # All properties: compositional kernel with per-component shared lengthscale
        optimizer = GPCrossValidatedOptimizer(
            X_train=X_all_train_sub,
            y_train=y_all_train_sub,
            kernels=None,
            n_splits=5,
            random_state=42,
            predefined_cv_splits=cv_splits_sub,
            compositional_kernel_dims=(dim_cat, dim_an),
            kernel_form=kernel_form,
        )

        # Run Bayesian hyperparameter optimization
        early_stop = make_early_stop_fn(patience=early_stop_patience)
        print(f"    Running Hyperopt (max {max_evals} evals, early stop patience={early_stop_patience})...")
        best_params = optimizer.optimize(max_evals=max_evals, early_stop_fn=early_stop)

        if best_params is None:
            print(f"    WARNING: Optimization failed for {prop}")
            continue

        # Get CV results
        trials = optimizer.get_optimization_results()
        best_cv_rmse = trials.best_trial['result']['loss']
        best_train_rmse = trials.best_trial['result']['train_loss']
        print(f"    Best CV RMSE: {best_cv_rmse:.4f}")
        print(f"    Best Train RMSE: {best_train_rmse:.4f}")

        # Evaluate on test set
        if X_test is not None and len(y_test) > 0:
            pred_mean, pred_var = optimizer.predict(X_test)
            if pred_mean is not None:
                pred_mean_flat = pred_mean.flatten()
                test_rmse = np.sqrt(np.mean((y_test - pred_mean_flat) ** 2))
                ss_res = np.sum((y_test - pred_mean_flat) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

                pred_std = np.sqrt(pred_var.flatten())
                lower = pred_mean_flat - 1.96 * pred_std
                upper = pred_mean_flat + 1.96 * pred_std
                ci_coverage = np.mean((y_test >= lower) & (y_test <= upper))

                nlpd = 0.5 * np.mean(
                    np.log(2 * np.pi * pred_var.flatten()) +
                    (y_test - pred_mean_flat) ** 2 / pred_var.flatten()
                )

                print(f"\n    Test Results ({prop}):")
                print(f"      R²: {test_r2:.4f}")
                print(f"      RMSE: {test_rmse:.4f}")
                print(f"      95% CI Coverage: {ci_coverage:.4f}")
                print(f"      NLPD: {nlpd:.4f}")

                all_stats[prop]['test_r2'] = test_r2
                all_stats[prop]['test_rmse'] = test_rmse
                all_stats[prop]['ci_coverage'] = ci_coverage
                all_stats[prop]['nlpd'] = nlpd

        models[prop] = optimizer
        print(f"    Model for {prop} trained successfully.")

        # Save model, scaler, test predictions
        import pickle, torch
        save_data = {
            'best_params': optimizer.best_params,
            'y_train_mean': optimizer.y_train_mean_,
            'variance_scale': getattr(optimizer, 'variance_scale_', 1.0),
            'scaler': scaler,
            'dim_cat': dim_cat,
            'dim_an': dim_an,
        }
        if optimizer.best_model_ is not None:
            save_data['model_state'] = optimizer.best_model_.state_dict()
            save_data['likelihood_state'] = optimizer.best_likelihood_.state_dict()
        if X_test is not None and pred_mean is not None:
            save_data['test_y_true'] = y_test
            save_data['test_y_pred'] = pred_mean.flatten()
            save_data['test_y_std'] = np.sqrt(pred_var.flatten())
        with open(os.path.join(OUTPUT_DIR, f'model_{prop}.pkl'), 'wb') as f:
            pickle.dump(save_data, f)
        print(f"    Model saved to output/model_{prop}.pkl")

    # Save stats
    stats_save = {}
    for prop, s in all_stats.items():
        stats_save[prop] = {
            'y_mean': float(s['y_mean']),
            'y_std': float(s['y_std']),
        }
    with open(os.path.join(OUTPUT_DIR, 'model_stats.json'), 'w') as f:
        json.dump(stats_save, f, indent=2)

    return models, all_stats


def phase3_conservative_bo(models, all_stats, vocab):
    """Phase 3: Conservative Multi-Objective Bayesian Optimization."""
    print("\n" + "=" * 60)
    print("PHASE 3: Conservative Multi-Objective BO (C-EHVI)")
    print("=" * 60)

    # Step 1: Compute thresholds from training data
    print("\nComputing thresholds from training data...")
    thresholds_raw = {}
    for prop in ['co2', 'vis', 'tox']:
        optimizer = models[prop]
        y_train = optimizer.y_train_1d  # raw training targets

        if prop == 'co2':
            thresholds_raw[prop] = np.percentile(y_train, 75)
        elif prop == 'vis':
            thresholds_raw[prop] = np.percentile(y_train, 25)
        elif prop == 'tox':
            thresholds_raw[prop] = np.percentile(y_train, 75)

        print(f"  {prop} threshold (raw): {thresholds_raw[prop]:.4f}")

    # Step 2: Build combinatorial candidate space
    print("\nBuilding combinatorial candidate space...")
    cation_list = list(vocab.cation_fps.keys())
    anion_list = list(vocab.anion_fps.keys())
    n_cat = len(cation_list)
    n_an = len(anion_list)
    n_candidates = n_cat * n_an
    print(f"  {n_cat} cations x {n_an} anions = {n_candidates:,} candidates")

    # Build all candidate fingerprints
    cat_fps_all = np.array([vocab.cation_fps[c] for c in cation_list], dtype=np.float64)
    an_fps_all = np.array([vocab.anion_fps[a] for a in anion_list], dtype=np.float64)

    # Step 3: Predict properties for all candidates
    batch_size_eval = 5000
    n_batches = (n_candidates + batch_size_eval - 1) // batch_size_eval

    all_mu = {}
    all_sigma = {}

    for prop in ['co2', 'vis', 'tox']:
        print(f"\n  Predicting {prop} for {n_candidates:,} candidates...")
        optimizer = models[prop]

        mu_list, sigma_list = [], []
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size_eval
            end = min(start + batch_size_eval, n_candidates)

            cat_idx = np.arange(start, end) // n_an
            an_idx = np.arange(start, end) % n_an

            batch_cat_fp = cat_fps_all[cat_idx]
            batch_an_fp = an_fps_all[an_idx]
            X_batch = np.concatenate([batch_cat_fp, batch_an_fp], axis=1)

            # For CO2 and vis, append ambient T/P then scale
            if prop in ['co2', 'vis']:
                env_batch = np.full((end - start, 2), [298.15, 101.325], dtype=np.float64)
                X_batch = np.concatenate([X_batch, env_batch], axis=1)

            # Apply the same StandardScaler used during training
            scaler = all_stats[prop]['scaler']
            X_batch = scaler.transform(X_batch)

            pred_mean, pred_var = optimizer.predict(X_batch)
            mu_list.append(pred_mean.flatten())
            sigma_list.append(np.sqrt(pred_var.flatten()))

            if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
                print(f"    Batch {batch_idx+1}/{n_batches}")

        all_mu[prop] = np.concatenate(mu_list)
        all_sigma[prop] = np.concatenate(sigma_list)

    # Step 4: Compute C-EHVI scores
    print("\nComputing C-EHVI scores...")

    mu_matrix = np.column_stack([all_mu['co2'], all_mu['vis'], all_mu['tox']])
    sigma_matrix = np.column_stack([all_sigma['co2'], all_sigma['vis'], all_sigma['tox']])

    predictions = {'mu': mu_matrix, 'sigma': sigma_matrix}
    thresh_array = np.array([thresholds_raw['co2'], thresholds_raw['vis'], thresholds_raw['tox']])

    # Reference point: worst values
    ref_point = np.array([
        models['co2'].y_train_1d.min(),
        models['vis'].y_train_1d.max(),  # high vis = worst
        models['tox'].y_train_1d.min(),
    ])

    maximize_flags = [True, False, True]  # max CO2, min vis, max logEC50

    scores = compute_c_ehvi_scores(predictions, thresh_array, ref_point, maximize_flags)

    # Step 5: Select and rank top candidates
    top_k = 100
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    print(f"\nTop 20 candidates by C-EHVI score:")
    print(f"{'Rank':<5} {'Score':<12} {'CO2_pred':<12} {'CO2_std':<10} "
          f"{'Vis_pred':<12} {'Vis_std':<10} {'Tox_pred':<12} {'Tox_std':<10}")
    print("-" * 93)

    for rank, idx in enumerate(top_indices):
        cat_i = idx // n_an
        an_i = idx % n_an
        cat_smi = cation_list[cat_i]
        an_smi = anion_list[an_i]

        entry = {
            'rank': rank + 1,
            'cation': cat_smi,
            'anion': an_smi,
            'il_smiles': f"{cat_smi}.{an_smi}",
            'c_ehvi_score': float(scores[idx]),
            'co2_pred': float(all_mu['co2'][idx]),
            'co2_std': float(all_sigma['co2'][idx]),
            'vis_pred': float(all_mu['vis'][idx]),
            'vis_std': float(all_sigma['vis'][idx]),
            'tox_pred': float(all_mu['tox'][idx]),
            'tox_std': float(all_sigma['tox'][idx]),
        }
        results.append(entry)

        if rank < 20:
            print(f"{rank+1:<5} {scores[idx]:<12.6f} {entry['co2_pred']:<12.4f} "
                  f"{entry['co2_std']:<10.4f} {entry['vis_pred']:<12.4f} "
                  f"{entry['vis_std']:<10.4f} {entry['tox_pred']:<12.4f} "
                  f"{entry['tox_std']:<10.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'top_candidates.csv'), index=False)
    print(f"\nTop-{top_k} candidates saved to output/top_candidates.csv")

    # Pareto front
    top_mu = mu_matrix[top_indices]
    pareto_mask = identify_pareto_front(top_mu, maximize_flags)
    n_pareto = pareto_mask.sum()
    print(f"Pareto-optimal among top-{top_k}: {n_pareto}")

    pareto_df = results_df[pareto_mask].copy()
    pareto_df.to_csv(os.path.join(OUTPUT_DIR, 'pareto_candidates.csv'), index=False)

    if n_pareto > 0:
        pareto_points = top_mu[pareto_mask]
        hv = compute_hypervolume(pareto_points, ref_point, maximize_flags)
        print(f"Hypervolume: {hv:.6f}")

    return results_df


def main():
    print("CF-BILD: Conservative Fragment-Based Bayesian IL Design")
    print("Using GP-BT with Compositional Kernel")
    print("=" * 60)
    start_time = time.time()

    # Phase 1: Fragment vocabulary
    datasets, vocab = phase1_fragment_vocabulary()

    # Phase 2: Train GP-BT models
    # kernel_form: "product" (default), "additive" (ablation), "standard" (baseline)
    models, all_stats = phase2_train_gp_models(
        datasets, vocab,
        kernel_form="product",
        max_evals=3000,
        early_stop_patience=50
    )

    # Phase 3: C-EHVI optimization
    results = phase3_conservative_bo(models, all_stats, vocab)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"CF-BILD completed in {elapsed/60:.1f} minutes")
    print(f"Results in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
