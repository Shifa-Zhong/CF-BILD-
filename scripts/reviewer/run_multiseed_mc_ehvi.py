"""
Multi-seed MC EHVI: re-run exact MC EHVI 5 times with different RandomState seeds,
report HV mean ± SE so the 17% gap between C-EHVI and MC EHVI is qualified with
posterior-sampling stochasticity.

Reuses the GP predictions (mu, sigma matrices) from the original run_exact_ehvi.py
by extracting them once, then sweeping the MC seed.
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def compute_exact_ehvi_scores(mu, sigma, ref_point, maximize_flags, n_samples=512, seed=42):
    """Per-candidate single-shot HVI: E[max(0, HV of point - HV of empty)] under N(mu, sigma)."""
    N, K = mu.shape
    mu_max = mu.copy()
    ref_max = ref_point.copy()
    for k in range(K):
        if not maximize_flags[k]:
            mu_max[:, k] = -mu[:, k]
            ref_max[k] = -ref_point[k]

    ehvi_scores = np.zeros(N)
    rng = np.random.RandomState(seed)

    for i in range(N):
        samples = rng.normal(mu_max[i], sigma[i], size=(n_samples, K))
        improvements = np.zeros(n_samples)
        for s in range(n_samples):
            point = samples[s]
            if np.all(point > ref_max):
                improvements[s] = np.prod(point - ref_max)
        ehvi_scores[i] = improvements.mean()
        if i % 10000 == 0:
            print(f'    seed={seed} progress {i}/{N}', flush=True)
    return ehvi_scores


def identify_pareto_front(points, maximize_flags):
    N = len(points)
    pf = np.zeros(N, dtype=bool)
    pts = points.copy()
    for k in range(len(maximize_flags)):
        if not maximize_flags[k]:
            pts[:, k] = -pts[:, k]
    for i in range(N):
        dominated = False
        for j in range(N):
            if i == j: continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dominated = True; break
        if not dominated:
            pf[i] = True
    return pf


def compute_hv_for_topN(mu_matrix, sigma_matrix, top_idx, ref_max, hv_calc, maximize_flags):
    """HV of pareto-optimal subset of top_idx."""
    sub_mu = mu_matrix[top_idx]
    sub_pareto = identify_pareto_front(sub_mu, maximize_flags)
    n_pareto = int(sub_pareto.sum())
    mu_max_pareto = sub_mu[sub_pareto].copy()
    for k in range(len(maximize_flags)):
        if not maximize_flags[k]:
            mu_max_pareto[:, k] = -mu_max_pareto[:, k]
    if n_pareto == 0:
        return 0.0, 0
    hv_val = hv_calc.compute(torch.tensor(mu_max_pareto, dtype=torch.float64))
    hv = hv_val.item() if hasattr(hv_val, 'item') else float(hv_val)
    return hv, n_pareto


def main():
    print("=" * 70)
    print("Multi-seed MC EHVI sweep (item I)")
    print("=" * 70)

    # Check for cached predictions; otherwise compute fresh
    cache_path = os.path.join(OUTPUT_DIR, 'predictions_87365.pkl')

    if not os.path.exists(cache_path):
        print(f"\nNo cached predictions at {cache_path} — running fresh prediction.")
        print("Loading GP models + computing predictions on 87,365 candidates...")
        # Import torch-heavy modules only when needed
        from bayesian_gp_cvloss import GPCrossValidatedOptimizer  # noqa
        from cf_bild.fragment_vocab import (
            FragmentVocabulary, load_property_datasets, prepare_cv_splits
        )

        # We need to load the predictions from somewhere. Easiest:
        # run the full original script and then capture mu/sigma matrices.
        # As an alternative, load saved model_*.pkl and predict.
        # For now, re-run the prediction loop from run_exact_ehvi main()
        from scripts.reviewer.run_exact_ehvi import main as _orig_main
        print("Calling original run_exact_ehvi.main() to generate mu/sigma matrices...")
        # This will save its own outputs; we'll need to patch to save matrices
        raise RuntimeError(
            "Predictions cache missing. Run scripts/reviewer/run_exact_ehvi.py "
            "first to generate mu/sigma, then add a cache-save step."
        )

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    mu_matrix = cache['mu']
    sigma_matrix = cache['sigma']
    ref_point = cache['ref_point']
    maximize_flags = cache['maximize_flags']
    print(f"Loaded predictions: mu shape {mu_matrix.shape}, sigma shape {sigma_matrix.shape}")
    print(f"ref_point = {ref_point}")

    # HV calculator in maximization frame
    ref_max = ref_point.copy()
    for k in range(len(maximize_flags)):
        if not maximize_flags[k]:
            ref_max[k] = -ref_point[k]
    hv_calc = Hypervolume(ref_point=torch.tensor(ref_max, dtype=torch.float64))

    # Run MC EHVI with 5 different seeds
    seeds = [42, 123, 7, 2024, 31415]
    results = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        t0 = time.time()
        scores = compute_exact_ehvi_scores(
            mu_matrix, sigma_matrix, ref_point, maximize_flags,
            n_samples=512, seed=seed,
        )
        top_idx = np.argsort(scores)[::-1][:100]
        hv, n_pareto = compute_hv_for_topN(
            mu_matrix, sigma_matrix, top_idx, ref_max, hv_calc, maximize_flags
        )
        mean_sigma = float(sigma_matrix[top_idx].mean())
        elapsed = time.time() - t0
        results.append({'seed': seed, 'HV': hv, 'n_pareto': n_pareto,
                        'mean_sigma': mean_sigma, 'elapsed_seconds': elapsed})
        print(f"  HV = {hv:.4f}, |Pareto| = {n_pareto}, mean_sigma = {mean_sigma:.4f}, ({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df)
    print(f"\nHV: mean = {df['HV'].mean():.4f}, std = {df['HV'].std():.4f}, "
          f"SE = {df['HV'].std() / np.sqrt(len(df)):.4f}")

    out_path = os.path.join(OUTPUT_DIR, 'multiseed_mc_ehvi.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
