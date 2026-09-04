'''Acquisition functions used in the revised CF-BILD workflow.

All objectives are represented in a maximization frame: CO2 capacity,
negative log-viscosity, and log EC50. CO2 uses a zero-truncated Gaussian
posterior. The other objectives use independent Gaussian posteriors.
'''

from __future__ import annotations

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from scipy.special import log_ndtr
from scipy.stats import norm


def to_maximization(mu):
    '''Convert [CO2, log-viscosity, logEC50] to a maximization frame.'''
    result = np.asarray(mu, dtype=float).copy()
    result[:, 1] *= -1.0
    return result


def normal_capped_improvement(mu, sigma, lower, upper):
    '''E[max(0, min(Y, upper)-lower)] for Gaussian Y.'''
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    mu = np.asarray(mu, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    a = (lower - mu) / sigma
    result = np.empty_like(mu)
    finite = np.isfinite(upper)
    if np.any(finite):
        b = (upper[finite] - mu[finite]) / sigma[finite]
        result[finite] = (
            (mu[finite] - lower[finite])
            * (norm.cdf(b) - norm.cdf(a[finite]))
            + sigma[finite] * (norm.pdf(a[finite]) - norm.pdf(b))
            + (upper[finite] - lower[finite]) * (1.0 - norm.cdf(b))
        )
    if np.any(~finite):
        z = (mu[~finite] - lower[~finite]) / sigma[~finite]
        result[~finite] = (
            (mu[~finite] - lower[~finite]) * norm.cdf(z)
            + sigma[~finite] * norm.pdf(z)
        )
    return np.maximum(result, 0.0)


def truncation_survival_at_zero(mu, sigma):
    '''P(Y >= 0) for the underlying CO2 Gaussian, calculated stably.'''
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    return np.exp(log_ndtr(mu / sigma))


def cell_width_expectation(mu, sigma, lower, upper, truncated_first=True):
    '''Expected hypercell widths under independent objective posteriors.'''
    widths = normal_capped_improvement(mu, sigma, lower, upper)
    if truncated_first:
        denominator = truncation_survival_at_zero(mu[..., 0], sigma[..., 0])
        widths[..., 0] /= np.maximum(denominator, 1e-300)
    return widths


def exact_q1_ehvi_scores(
    latent_mu,
    latent_sigma,
    reference,
    incumbent_pareto,
    batch_size=2000,
):
    '''Analytical q=1 EHVI with a non-empty incumbent Pareto front.

    The calculation is exact under independent objective posteriors, with a
    zero-truncated Gaussian for CO2 and ordinary Gaussians for the other two
    objectives.
    '''
    partitioning = NondominatedPartitioning(
        ref_point=torch.as_tensor(reference, dtype=torch.double),
        Y=torch.as_tensor(incumbent_pareto, dtype=torch.double),
    )
    bounds = partitioning.get_hypercell_bounds().cpu().numpy()
    lower, upper = bounds[0], bounds[1]
    scores = np.zeros(len(latent_mu), dtype=float)
    for start in range(0, len(latent_mu), batch_size):
        end = min(start + batch_size, len(latent_mu))
        mu_batch = np.broadcast_to(
            latent_mu[start:end, None, :], (end - start, *lower.shape)
        )
        sigma_batch = np.broadcast_to(
            latent_sigma[start:end, None, :], (end - start, *lower.shape)
        )
        lower_batch = np.broadcast_to(lower[None, :, :], mu_batch.shape)
        upper_batch = np.broadcast_to(upper[None, :, :], mu_batch.shape)
        widths = cell_width_expectation(
            mu_batch, sigma_batch, lower_batch, upper_batch
        )
        scores[start:end] = np.prod(widths, axis=2).sum(axis=1)
    return scores, int(lower.shape[0])


def additive_ei_scores(latent_mu, latent_sigma, reference):
    '''Sum of per-objective EI values relative to the fixed reference.'''
    latent_mu = np.asarray(latent_mu, dtype=float)
    latent_sigma = np.asarray(latent_sigma, dtype=float)
    lower = np.broadcast_to(reference, latent_mu.shape)
    upper = np.full_like(latent_mu, np.inf)
    contributions = cell_width_expectation(
        latent_mu, latent_sigma, lower, upper
    )
    return contributions.sum(axis=1)


def joint_feasibility_probability(latent_mu, latent_sigma, thresholds):
    '''Probability that all property thresholds are met.'''
    latent_mu = np.asarray(latent_mu, dtype=float)
    latent_sigma = np.maximum(np.asarray(latent_sigma, dtype=float), 1e-12)
    thresholds = np.asarray(thresholds, dtype=float)

    probability = np.ones(len(latent_mu), dtype=float)
    co2_tail = norm.sf(
        (thresholds[0] - latent_mu[:, 0]) / latent_sigma[:, 0]
    )
    co2_support = truncation_survival_at_zero(
        latent_mu[:, 0], latent_sigma[:, 0]
    )
    probability *= co2_tail / np.maximum(co2_support, 1e-300)
    for objective in range(1, latent_mu.shape[1]):
        z = (
            latent_mu[:, objective] - thresholds[objective]
        ) / latent_sigma[:, objective]
        probability *= norm.cdf(z)
    return np.clip(probability, 0.0, 1.0)


def fw_aei_scores(latent_mu, latent_sigma, reference, thresholds):
    '''Feasibility-weighted additive expected improvement (FW-AEI).'''
    return additive_ei_scores(
        latent_mu, latent_sigma, reference
    ) * joint_feasibility_probability(
        latent_mu, latent_sigma, thresholds
    )


def pareto_mask(points):
    '''Boolean non-dominance mask for maximization objectives.'''
    return is_non_dominated(
        torch.as_tensor(points, dtype=torch.double)
    ).cpu().numpy()


def pareto_points(points):
    '''Non-dominated subset of maximization-frame points.'''
    points = np.asarray(points, dtype=float)
    return points[pareto_mask(points)]


def incumbent_front(physical_mu, reference, indices):
    '''Build a non-empty incumbent front from represented IL identities.'''
    values = np.asarray(physical_mu, dtype=float)[np.asarray(indices, dtype=int)]
    values = values[np.all(values > np.asarray(reference, dtype=float), axis=1)]
    if len(values) == 0:
        raise ValueError('No incumbent lies strictly above the reference point.')
    return pareto_points(values)


def hypervolume(points, reference):
    '''Exact dominated hypervolume for a finite maximization-frame set.'''
    front = pareto_points(points)
    return float(Hypervolume(
        ref_point=torch.as_tensor(reference, dtype=torch.double)
    ).compute(torch.as_tensor(front, dtype=torch.double)))


def selected_set_metrics(physical_mu, physical_sigma, indices, reference):
    '''Metrics reported for a selected candidate set.'''
    indices = np.asarray(indices, dtype=int)
    selected = np.asarray(physical_mu, dtype=float)[indices]
    front = pareto_points(selected)
    return {
        'hypervolume': hypervolume(front, reference),
        'n_pareto': int(len(front)),
        'mean_sigma': float(
            np.asarray(physical_sigma, dtype=float)[indices].mean()
        ),
    }


def top_indices(scores, number):
    '''Stable descending ranking.'''
    return np.argsort(np.asarray(scores), kind='stable')[::-1][:number]
