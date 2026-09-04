'''Physical-support corrections and common predictive metrics.

The CO2-capacity GP is fitted on the measured scale, but its Gaussian
posterior is conditioned on the physically required event Y >= 0 before
reporting predictions or evaluating acquisition functions. Other properties
retain their ordinary Gaussian posterior.
'''

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri


_LOG_2PI = np.log(2.0 * np.pi)


def _as_arrays(mu, variance):
    mu = np.asarray(mu, dtype=float)
    variance = np.maximum(np.asarray(variance, dtype=float), 1e-16)
    return mu, variance, np.sqrt(variance)


def zero_truncated_normal_moments(mu, variance):
    '''Mean and variance of N(mu, variance) conditional on Y >= 0.'''
    mu, variance, sigma = _as_arrays(mu, variance)
    alpha = -mu / sigma
    log_survival = log_ndtr(-alpha)
    log_density = -0.5 * alpha * alpha - 0.5 * _LOG_2PI
    inverse_mills = np.exp(np.clip(log_density - log_survival, -745.0, 700.0))
    mean = mu + sigma * inverse_mills
    conditional_variance = variance * (
        1.0 + alpha * inverse_mills - inverse_mills * inverse_mills
    )
    conditional_variance = np.maximum(conditional_variance, 0.0)
    return mean, conditional_variance


def zero_truncated_normal_quantile(mu, variance, probability):
    '''Quantile of N(mu, variance) conditional on Y >= 0.'''
    mu, variance, sigma = _as_arrays(mu, variance)
    probability = np.asarray(probability, dtype=float)
    if np.any((probability <= 0.0) | (probability >= 1.0)):
        raise ValueError('probability must lie strictly between zero and one')
    alpha = -mu / sigma
    survival_at_zero = np.exp(log_ndtr(-alpha))
    conditional_survival = (1.0 - probability) * survival_at_zero
    z = -ndtri(np.clip(conditional_survival, 1e-300, 1.0 - 1e-16))
    return np.maximum(mu + sigma * z, 0.0)


def physical_prediction(property_name, mu, variance, interval=0.95):
    '''Return physical mean, variance, and equal-tail prediction interval.'''
    mu, variance, sigma = _as_arrays(mu, variance)
    tail = (1.0 - interval) / 2.0
    if property_name == 'co2':
        mean, physical_variance = zero_truncated_normal_moments(mu, variance)
        lower = zero_truncated_normal_quantile(mu, variance, tail)
        upper = zero_truncated_normal_quantile(mu, variance, 1.0 - tail)
    else:
        mean = mu
        physical_variance = variance
        lower = mu + sigma * ndtri(tail)
        upper = mu + sigma * ndtri(1.0 - tail)
    return {
        'mean': mean,
        'variance': physical_variance,
        'std': np.sqrt(np.maximum(physical_variance, 0.0)),
        'lower': lower,
        'upper': upper,
    }


def regression_metrics(y_true, prediction):
    '''Calculate R2, RMSE, interval coverage, and Gaussian moment NLPD.'''
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(prediction['mean'], dtype=float)
    variance = np.maximum(np.asarray(prediction['variance'], dtype=float), 1e-16)
    residual = y_true - mean
    total = np.sum((y_true - y_true.mean()) ** 2)
    return {
        'r2': float(1.0 - np.sum(residual ** 2) / total) if total > 0 else 0.0,
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'mae': float(np.mean(np.abs(residual))),
        'coverage_95': float(np.mean(
            (y_true >= prediction['lower']) & (y_true <= prediction['upper'])
        )),
        'nlpd_moment_matched': float(0.5 * np.mean(
            np.log(2.0 * np.pi * variance) + residual ** 2 / variance
        )),
    }


def viscosity_real_space_metrics(y_true_log, prediction_log):
    '''Metrics after exponentiating natural-log viscosity to Pa s.'''
    true = np.exp(np.asarray(y_true_log, dtype=float))
    predicted = np.exp(np.asarray(prediction_log, dtype=float))
    residual = true - predicted
    total = np.sum((true - true.mean()) ** 2)
    return {
        'r2': float(1.0 - np.sum(residual ** 2) / total) if total > 0 else 0.0,
        'rmse_pa_s': float(np.sqrt(np.mean(residual ** 2))),
        'mae_pa_s': float(np.mean(np.abs(residual))),
        'median_ae_pa_s': float(np.median(np.abs(residual))),
        'median_ape_percent': float(np.median(
            np.abs(residual) / np.maximum(np.abs(true), 1e-15)
        ) * 100.0),
    }
