import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm, truncnorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cf_bild.acquisition import (
    exact_q1_ehvi_scores,
    fw_aei_scores,
)
from cf_bild.fragment_vocab import (
    FragmentVocabulary,
    get_non_test_dataframe,
    load_property_datasets,
)
from cf_bild.predictive import (
    physical_prediction,
    zero_truncated_normal_moments,
    zero_truncated_normal_quantile,
)


def test_zero_truncated_moments_match_scipy():
    mu = np.array([-2.0, 0.0, 1.0])
    variance = np.array([1.0, 0.25, 4.0])
    sigma = np.sqrt(variance)
    alpha = -mu / sigma
    mean, observed_variance = zero_truncated_normal_moments(mu, variance)
    np.testing.assert_allclose(
        mean, truncnorm.mean(alpha, np.inf, loc=mu, scale=sigma)
    )
    np.testing.assert_allclose(
        observed_variance,
        truncnorm.var(alpha, np.inf, loc=mu, scale=sigma),
    )


def test_co2_prediction_is_strictly_nonnegative():
    prediction = physical_prediction(
        'co2',
        np.array([-5.0, -0.2, 0.5]),
        np.array([0.1, 0.4, 0.2]),
    )
    for key in ('mean', 'lower', 'upper'):
        assert np.all(prediction[key] >= 0.0)


def test_analytic_ehvi_matches_vectorized_monte_carlo():
    reference = np.array([0.0, -2.0])
    incumbent = np.array([[0.5, 0.0]])
    latent_mu = np.array([[0.4, -0.1]])
    latent_sigma = np.array([[0.3, 0.5]])
    analytic, _ = exact_q1_ehvi_scores(
        latent_mu, latent_sigma, reference, incumbent
    )

    rng = np.random.default_rng(42)
    n_samples = 400000
    alpha = -latent_mu[0, 0] / latent_sigma[0, 0]
    x = truncnorm.rvs(
        alpha,
        np.inf,
        loc=latent_mu[0, 0],
        scale=latent_sigma[0, 0],
        size=n_samples,
        random_state=rng,
    )
    y = rng.normal(
        latent_mu[0, 1], latent_sigma[0, 1], size=n_samples
    )
    candidate_area = x * np.maximum(y - reference[1], 0.0)
    overlap = np.minimum(x, 0.5) * np.minimum(
        np.maximum(y - reference[1], 0.0), 2.0
    )
    monte_carlo = np.mean(candidate_area - overlap)
    np.testing.assert_allclose(analytic[0], monte_carlo, rtol=0.015)


def test_fw_aei_is_finite_and_nonnegative():
    mu = np.array([[0.2, 1.0], [-2.0, -1.0]])
    sigma = np.array([[0.1, 0.5], [1.0, 2.0]])
    scores = fw_aei_scores(
        mu,
        sigma,
        reference=np.array([0.0, -3.0]),
        thresholds=np.array([0.1, 0.0]),
    )
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_complete_non_test_pool_counts_and_fold_identity():
    if not Path('data/train_1_group_co2.csv').exists():
        raise unittest.SkipTest(
            'Restricted source-derived tables are not distributed publicly.'
        )
    datasets = load_property_datasets('data', n_folds=5)
    expected = {'co2': 12503, 'vis': 12374, 'tox': 302}
    for property_name, count in expected.items():
        frame = get_non_test_dataframe(datasets, property_name)
        assert len(frame) == count


def test_revision_cache_contract():
    with open(
        'output/revision_2026/predictions_87365_revision.pkl', 'rb'
    ) as handle:
        cache = pickle.load(handle)
    assert cache['property_order'] == ['co2', 'vis', 'tox']
    assert cache['physical_mu'].shape == (87365, 3)
    assert cache['physical_sigma'].shape == (87365, 3)
    assert np.all(cache['physical_mu'][:, 0] >= 0.0)
    assert np.all(cache['physical_sigma'] >= 0.0)


def main():
    tests = [
        test_zero_truncated_moments_match_scipy,
        test_co2_prediction_is_strictly_nonnegative,
        test_analytic_ehvi_matches_vectorized_monte_carlo,
        test_fw_aei_is_finite_and_nonnegative,
        test_complete_non_test_pool_counts_and_fold_identity,
        test_revision_cache_contract,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')


if __name__ == '__main__':
    main()
