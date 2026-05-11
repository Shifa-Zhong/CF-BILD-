"""
Layer 3: Conservative Multi-Objective Bayesian Optimization (C-EHVI)

C-EHVI = EHVI(x) * prod_k Phi((mu_k(x) - tau_k) / sigma_k(x))

where Phi is standard normal CDF, tau_k is threshold for property k.
"""

import torch
import numpy as np
from scipy.stats import norm


def compute_c_ehvi_scores(predictions, thresholds, ref_point, maximize_flags):
    """Compute Conservative EHVI scores for all candidates.

    This implements a simplified C-EHVI that works on the discrete candidate space.
    Instead of the full EHVI integral, we use a decomposition-friendly approach:
    1. Compute expected improvement contribution for each objective
    2. Multiply by feasibility probability (conservative penalty)

    Args:
        predictions: dict with keys 'mu' (N x K), 'sigma' (N x K)
            where N is num candidates, K is num objectives
        thresholds: array of shape (K,), threshold tau_k for each objective
        ref_point: array of shape (K,), reference point for hypervolume
        maximize_flags: list of bool, True if objective should be maximized

    Returns:
        scores: array of shape (N,), C-EHVI score for each candidate
    """
    mu = predictions['mu']      # (N, K)
    sigma = predictions['sigma']  # (N, K)
    N, K = mu.shape

    # Flip sign for minimization objectives so everything is "maximize"
    mu_max = mu.copy()
    sigma_max = sigma.copy()
    thresh_max = thresholds.copy()
    ref_max = ref_point.copy()

    for k in range(K):
        if not maximize_flags[k]:
            mu_max[:, k] = -mu[:, k]
            thresh_max[k] = -thresholds[k]
            ref_max[k] = -ref_point[k]

    # 1. Compute feasibility probability: prod_k Phi((mu_k - tau_k) / sigma_k)
    feasibility = np.ones(N)
    for k in range(K):
        z = (mu_max[:, k] - thresh_max[k]) / (sigma_max[:, k] + 1e-8)
        feasibility *= norm.cdf(z)

    # 2. Compute expected improvement for each objective
    # EI_k = sigma_k * [z_k * Phi(z_k) + phi(z_k)] where z_k = (mu_k - ref_k) / sigma_k
    total_ei = np.zeros(N)
    for k in range(K):
        z = (mu_max[:, k] - ref_max[k]) / (sigma_max[:, k] + 1e-8)
        ei_k = sigma_max[:, k] * (z * norm.cdf(z) + norm.pdf(z))
        total_ei += ei_k  # additive EI as proxy for EHVI

    # 3. C-EHVI = EI * feasibility
    scores = total_ei * feasibility

    return scores


def compute_thresholds_from_training(train_y_dict, stats_dict):
    """Compute thresholds from training data as specified in the plan.

    CO2 capacity: > 75th percentile (maximize)
    Viscosity: < 25th percentile (minimize, so threshold on negated = > 75th percentile of -vis)
    Toxicity (logEC50): > 75th percentile (maximize, higher EC50 = lower toxicity)

    Args:
        train_y_dict: dict mapping property name -> standardized training y values
        stats_dict: dict mapping property name -> {'y_mean', 'y_std'}

    Returns:
        thresholds: dict mapping property name -> threshold in standardized space
        thresholds_raw: dict mapping property name -> threshold in original space
    """
    thresholds = {}
    thresholds_raw = {}

    for prop in ['co2', 'vis', 'tox']:
        if prop not in train_y_dict:
            continue
        y = train_y_dict[prop]
        y_mean = stats_dict[prop]['y_mean']
        y_std = stats_dict[prop]['y_std']

        # Raw values
        y_raw = y * y_std + y_mean

        if prop == 'co2':
            # > 75th percentile (maximize)
            thresh_raw = np.percentile(y_raw, 75)
        elif prop == 'vis':
            # < 25th percentile (minimize) => in standardized, this is 25th percentile
            thresh_raw = np.percentile(y_raw, 25)
        elif prop == 'tox':
            # > 75th percentile (maximize logEC50)
            thresh_raw = np.percentile(y_raw, 75)

        thresholds[prop] = (thresh_raw - y_mean) / y_std
        thresholds_raw[prop] = thresh_raw

    return thresholds, thresholds_raw


def select_batch(predictions, thresholds, ref_point, maximize_flags,
                 batch_size=50):
    """Select top-B candidates using C-EHVI scores.

    Args:
        predictions: dict with 'mu' (N,K) and 'sigma' (N,K)
        thresholds: (K,) array
        ref_point: (K,) array
        maximize_flags: list of K bools
        batch_size: number of candidates to select

    Returns:
        indices: array of selected candidate indices
        scores: C-EHVI scores for selected candidates
    """
    scores = compute_c_ehvi_scores(predictions, thresholds, ref_point, maximize_flags)
    top_indices = np.argsort(scores)[::-1][:batch_size]

    return top_indices, scores[top_indices]


def identify_pareto_front(mu, maximize_flags):
    """Identify Pareto-optimal candidates from predicted means.

    Args:
        mu: (N, K) array of predicted means
        maximize_flags: list of K bools

    Returns:
        pareto_mask: boolean array of shape (N,), True for Pareto-optimal points
    """
    N = mu.shape[0]
    mu_max = mu.copy()
    for k in range(mu.shape[1]):
        if not maximize_flags[k]:
            mu_max[:, k] = -mu[:, k]

    pareto_mask = np.ones(N, dtype=bool)
    for i in range(N):
        if not pareto_mask[i]:
            continue
        for j in range(N):
            if i == j or not pareto_mask[j]:
                continue
            # Check if j dominates i
            if np.all(mu_max[j] >= mu_max[i]) and np.any(mu_max[j] > mu_max[i]):
                pareto_mask[i] = False
                break

    return pareto_mask


def compute_hypervolume(pareto_points, ref_point, maximize_flags):
    """Compute hypervolume indicator of the Pareto front.

    Simple 2D/3D implementation. For the 3D case, uses a sweep-line approach.
    """
    if len(pareto_points) == 0:
        return 0.0

    # Convert to maximization
    pts = pareto_points.copy()
    ref = ref_point.copy()
    for k in range(pts.shape[1]):
        if not maximize_flags[k]:
            pts[:, k] = -pts[:, k]
            ref[k] = -ref_point[k]

    K = pts.shape[1]
    if K == 2:
        return _hv_2d(pts, ref)
    elif K == 3:
        return _hv_3d(pts, ref)
    else:
        # Fallback: product of marginal improvements (rough approximation)
        hv = 0.0
        for p in pts:
            improvement = np.maximum(p - ref, 0)
            hv += np.prod(improvement)
        return hv


def _hv_2d(pts, ref):
    """Exact 2D hypervolume."""
    # Filter points dominated by ref
    valid = np.all(pts > ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    # Sort by first objective descending
    order = np.argsort(-pts[:, 0])
    pts = pts[order]

    hv = 0.0
    prev_y = ref[1]
    for p in pts:
        hv += (p[0] - ref[0]) * (p[1] - prev_y)
        prev_y = max(prev_y, p[1])
    return hv


def _hv_3d(pts, ref):
    """Approximate 3D hypervolume via slicing."""
    valid = np.all(pts > ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    # Sort by third objective
    order = np.argsort(pts[:, 2])
    pts = pts[order]

    hv = 0.0
    prev_z = ref[2]
    for i, p in enumerate(pts):
        # 2D hypervolume of points[:i+1] projected onto first two objectives
        front_2d = pts[:i+1, :2]
        hv_2d = _hv_2d(front_2d, ref[:2])
        hv += hv_2d * (p[2] - prev_z)
        prev_z = p[2]
    return hv
