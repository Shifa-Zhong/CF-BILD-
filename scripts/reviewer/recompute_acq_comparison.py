"""
Recompute Fig 4a's 4 acquisition methods (C-EHVI, exact MC EHVI, additive EI,
random) from the CACHED calibrated predictions, all using identical preprocessing,
so the numbers are self-consistent and the new multi-seed SE for MC EHVI is
comparable to C-EHVI / additive-EI / random.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def identify_pareto(points_max):
    N = len(points_max)
    pf = np.ones(N, dtype=bool)
    for i in range(N):
        if not pf[i]: continue
        for j in range(N):
            if i == j: continue
            if np.all(points_max[j] >= points_max[i]) and np.any(points_max[j] > points_max[i]):
                pf[i] = False; break
    return pf


def compute_hv(points, ref_max, hv_calc, maximize_flags):
    """HV of pareto subset of points (provided in original frame)."""
    pm = points.copy()
    for k, mf in enumerate(maximize_flags):
        if not mf: pm[:, k] = -pm[:, k]
    pf = identify_pareto(pm)
    if pf.sum() == 0: return 0.0, 0
    hv_val = hv_calc.compute(torch.tensor(pm[pf], dtype=torch.float64))
    return float(hv_val.item() if hasattr(hv_val, 'item') else hv_val), int(pf.sum())


def compute_c_ehvi_score(mu, sigma, ref_max, thresh_max, maximize_flags):
    """C-EHVI = [Σₖ EIₖ] × [Πₖ Φ((μₖ−τₖ)/σₖ)]"""
    K = mu.shape[1]
    mu_max = mu.copy()
    for k, mf in enumerate(maximize_flags):
        if not mf: mu_max[:, k] = -mu_max[:, k]
    total_ei = np.zeros(mu.shape[0])
    feasibility = np.ones(mu.shape[0])
    for k in range(K):
        z_r = (mu_max[:, k] - ref_max[k]) / (sigma[:, k] + 1e-9)
        total_ei += sigma[:, k] * (z_r * norm.cdf(z_r) + norm.pdf(z_r))
        z_t = (mu_max[:, k] - thresh_max[k]) / (sigma[:, k] + 1e-9)
        feasibility *= norm.cdf(z_t)
    return total_ei * feasibility


def compute_mc_ehvi(mu, sigma, ref_max, maximize_flags, n_samples=512, seed=42):
    N, K = mu.shape
    mu_max = mu.copy()
    for k, mf in enumerate(maximize_flags):
        if not mf: mu_max[:, k] = -mu_max[:, k]
    rng = np.random.RandomState(seed)
    scores = np.zeros(N)
    for i in range(N):
        samples = rng.normal(mu_max[i], sigma[i], size=(n_samples, K))
        improvements = np.where(
            np.all(samples > ref_max, axis=1),
            np.prod(samples - ref_max, axis=1), 0.0)
        scores[i] = improvements.mean()
        if (i+1) % 20000 == 0:
            print(f'    seed={seed} {i+1}/{N}', flush=True)
    return scores


def main():
    print('Loading cached predictions...')
    with open(os.path.join(OUTPUT_DIR, 'predictions_87365.pkl'), 'rb') as f:
        cache = pickle.load(f)
    mu = cache['mu']
    sigma = cache['sigma']
    ref_point = cache['ref_point']
    maximize_flags = cache['maximize_flags']
    N, K = mu.shape
    print(f'  {N} candidates × {K} objectives')

    # Maximization frame
    ref_max = ref_point.copy()
    for k, mf in enumerate(maximize_flags):
        if not mf: ref_max[k] = -ref_point[k]
    hv_calc = Hypervolume(ref_point=torch.tensor(ref_max, dtype=torch.float64))

    # Thresholds for C-EHVI in maximization frame
    thresh_max = np.zeros(K)
    for k, mf in enumerate(maximize_flags):
        pct = 75 if mf else 25
        thresh_max[k] = np.percentile(mu[:, k], pct) * (1 if mf else -1)

    results = {}

    # ===== 1. C-EHVI =====
    print('\n1. C-EHVI')
    cehvi = compute_c_ehvi_score(mu, sigma, ref_max, thresh_max, maximize_flags)
    top = np.argsort(cehvi)[::-1][:100]
    hv, n_p = compute_hv(mu[top], ref_max, hv_calc, maximize_flags)
    results['C-EHVI'] = {'HV': hv, 'n_pareto': n_p,
                         'mean_sigma': float(sigma[top].mean()),
                         'mean_co2': float(mu[top, 0].mean())}
    print(f'  HV={hv:.4f}, |Pareto|={n_p}, mean_sigma={sigma[top].mean():.4f}')

    # ===== 2. MC EHVI (multi-seed) =====
    print('\n2. MC EHVI (5 seeds)')
    seeds = [42, 123, 7, 2024, 31415]
    mc_results = []
    for seed in seeds:
        mc = compute_mc_ehvi(mu, sigma, ref_max, maximize_flags, n_samples=512, seed=seed)
        top = np.argsort(mc)[::-1][:100]
        hv, n_p = compute_hv(mu[top], ref_max, hv_calc, maximize_flags)
        mc_results.append({'seed': seed, 'HV': hv, 'n_pareto': n_p,
                          'mean_sigma': float(sigma[top].mean()),
                          'mean_co2': float(mu[top, 0].mean())})
        print(f'  seed {seed}: HV={hv:.4f}, |Pareto|={n_p}')
    mc_df = pd.DataFrame(mc_results)
    results['Exact EHVI (MC, 5-seed)'] = {
        'HV_mean': float(mc_df['HV'].mean()),
        'HV_std': float(mc_df['HV'].std()),
        'HV_SE': float(mc_df['HV'].std() / np.sqrt(len(mc_df))),
        'HV_per_seed': mc_df.to_dict(orient='records'),
        'n_pareto_mean': float(mc_df['n_pareto'].mean()),
        'mean_sigma_mean': float(mc_df['mean_sigma'].mean()),
    }
    print(f'  MEAN HV = {mc_df["HV"].mean():.4f} ± {mc_df["HV"].std() / np.sqrt(len(mc_df)):.4f} (SE)')

    # ===== 3. Additive EI (no feasibility) =====
    print('\n3. Additive EI (no feasibility)')
    mu_max_all = mu.copy()
    for k, mf in enumerate(maximize_flags):
        if not mf: mu_max_all[:, k] = -mu_max_all[:, k]
    additive_ei = np.zeros(N)
    for k in range(K):
        z = (mu_max_all[:, k] - ref_max[k]) / (sigma[:, k] + 1e-9)
        additive_ei += sigma[:, k] * (z * norm.cdf(z) + norm.pdf(z))
    top = np.argsort(additive_ei)[::-1][:100]
    hv, n_p = compute_hv(mu[top], ref_max, hv_calc, maximize_flags)
    results['Additive EI'] = {'HV': hv, 'n_pareto': n_p,
                              'mean_sigma': float(sigma[top].mean()),
                              'mean_co2': float(mu[top, 0].mean())}
    print(f'  HV={hv:.4f}, |Pareto|={n_p}, mean_sigma={sigma[top].mean():.4f}')

    # ===== 4. Random =====
    print('\n4. Random (5 seeds for fairness)')
    rand_results = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, 100, replace=False)
        hv, n_p = compute_hv(mu[idx], ref_max, hv_calc, maximize_flags)
        rand_results.append({'seed': seed, 'HV': hv, 'n_pareto': n_p})
    rand_df = pd.DataFrame(rand_results)
    results['Random'] = {
        'HV_mean': float(rand_df['HV'].mean()),
        'HV_std': float(rand_df['HV'].std()),
        'HV_SE': float(rand_df['HV'].std() / np.sqrt(len(rand_df))),
        'mean_pareto': float(rand_df['n_pareto'].mean()),
    }
    print(f'  MEAN HV = {rand_df["HV"].mean():.4f} ± {rand_df["HV"].std() / np.sqrt(len(rand_df)):.4f}')

    # ===== Summary =====
    print('\n' + '='*70)
    print('SUMMARY (consistent calibrated-sigma re-computation)')
    print('='*70)
    cehvi_hv = results['C-EHVI']['HV']
    mc_mean = results['Exact EHVI (MC, 5-seed)']['HV_mean']
    mc_se = results['Exact EHVI (MC, 5-seed)']['HV_SE']
    addei_hv = results['Additive EI']['HV']
    rand_hv = results['Random']['HV_mean']
    rand_se = results['Random']['HV_SE']
    print(f'  C-EHVI:           HV = {cehvi_hv:.3f}')
    print(f'  Exact MC EHVI:    HV = {mc_mean:.3f} +/- {mc_se:.3f} (SE, 5 seeds)')
    print(f'  Additive EI:      HV = {addei_hv:.3f}')
    print(f'  Random:           HV = {rand_hv:.3f} +/- {rand_se:.3f} (SE, 5 seeds)')
    gap_pct = 100 * (mc_mean - cehvi_hv) / mc_mean
    print(f'  C-EHVI vs MC EHVI gap: {gap_pct:.1f}% (statistical significance: '
          f'gap = {mc_mean - cehvi_hv:.2f}, MC SE = {mc_se:.3f}, gap/SE ~ {(mc_mean-cehvi_hv)/mc_se:.0f}σ)')

    out_path = os.path.join(OUTPUT_DIR, 'acq_comparison_consistent.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
