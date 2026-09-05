"""Small distribution and acquisition regressions, independent of data paths."""
import sys
from pathlib import Path
import torch
import numpy as np
from scipy.stats import truncnorm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cf_bild.predictive import physical_prediction,zero_truncated_normal_moments
from cf_bild.acquisition import exact_q1_ehvi_scores,fw_aei_scores


def test_zero_truncated_moments_match_scipy():
    mu=np.array([-2.,0.,1.]);variance=np.array([1.,.25,4.]);sd=np.sqrt(variance)
    mean,var=zero_truncated_normal_moments(mu,variance)
    np.testing.assert_allclose(mean,truncnorm.mean(-mu/sd,np.inf,loc=mu,scale=sd))
    np.testing.assert_allclose(var,truncnorm.var(-mu/sd,np.inf,loc=mu,scale=sd))


def test_co2_prediction_is_strictly_nonnegative():
    result=physical_prediction('co2',np.array([-5.,-.2,.5]),np.array([.1,.4,.2]))
    for name in ['mean','lower','upper']:assert (result[name]>=0).all()


def test_analytic_ehvi_matches_vectorized_monte_carlo():
    reference=np.array([0.,-2.]);incumbent=np.array([[.5,0.]])
    mu=np.array([[.4,-.1]]);sd=np.array([[.3,.5]])
    analytic,_=exact_q1_ehvi_scores(mu,sd,reference,incumbent)
    rng=np.random.default_rng(42);n=400000
    x=truncnorm.rvs(-mu[0,0]/sd[0,0],np.inf,loc=mu[0,0],scale=sd[0,0],size=n,random_state=rng)
    y=rng.normal(mu[0,1],sd[0,1],size=n)
    volume=x*np.maximum(y-reference[1],0.)
    overlap=np.minimum(x,.5)*np.minimum(np.maximum(y-reference[1],0.),2.)
    np.testing.assert_allclose(analytic[0],np.mean(volume-overlap),rtol=.015)


def test_fw_aei_is_finite_and_nonnegative():
    score=fw_aei_scores(np.array([[.2,1.],[-2.,-1.]]),np.array([[.1,.5],[1.,2.]]),
        reference=np.array([0.,-3.]),thresholds=np.array([.1,0.]))
    assert np.isfinite(score).all() and (score>=0).all()
