"""Run scientific regression tests without manuscript or plotting dependencies."""
import sys
from pathlib import Path
import runpy
import unittest
import torch
import numpy as np
import pandas as pd
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
RUN=ROOT/'runs/ion_clean_refit_2026-09-05'


def main():
    for name in ['test_shared_kernel_baseline.py','test_ranking_diagnostics.py']:
        runpy.run_path(str(ROOT/'tests'/name))['main']()
    numerical=runpy.run_path(str(ROOT/'tests/test_numerical_methods.py'))
    for name in ['test_zero_truncated_moments_match_scipy','test_co2_prediction_is_strictly_nonnegative',
                 'test_analytic_ehvi_matches_vectorized_monte_carlo','test_fw_aei_is_finite_and_nonnegative']:
        numerical[name]();print('PASS',name)
    families=runpy.run_path(str(ROOT/'tests/test_ion_families.py'))
    for name,fn in families.items():
        if name.startswith('test_'):fn();print('PASS',name)
    import test_clean_refit
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_clean_refit))
    if not result.wasSuccessful():raise AssertionError('Curation/checkpoint tests failed')
    expected={'co2':(10867,1080),'vis':(11101,1344),'tox':(300,31)}
    for prop,(nt,te) in expected.items():
        data=[pd.read_csv(RUN/f'data/{r}_1_group_{prop}.csv') for r in ['train','val']]
        assert sum(map(len,data))==nt
        assert len(pd.read_csv(RUN/f'data/test_group_{prop}.csv'))==te
    pairs=pd.read_csv(RUN/'analysis/candidate_pairs.csv')
    assert len(pairs)==67488 and pairs.cation.nunique()==444 and pairs.anion.nunique()==152
    with np.load(RUN/'analysis/candidate_predictions.npz',allow_pickle=False) as cache:
        for key in ['latent_mu','latent_sigma','physical_mu','physical_sigma']:
            assert cache[key].shape==(67488,3) and np.isfinite(cache[key]).all()
        assert (cache['physical_mu'][:,0]>=0).all() and (cache['physical_sigma']>=0).all()
    for form in ['product','product_no_cross','additive','additive_no_cross','standard']:
        for prop in expected:
            metadata=json.loads((RUN/f'results/{form}/metrics_{prop}.json').read_text())
            assert (metadata['n_refit'],metadata['n_test'])==expected[prop]
    top=pd.read_csv(RUN/'analysis/stability_screening_revision.csv')
    assert top.overall_status.value_counts().to_dict()=={'Fail':63,'Pass':35,'Caution':2}
    assert top.tier3_flag.sum()==0
    post=pd.read_csv(RUN/'analysis/post_filter_top10_revision.csv')
    assert post['rank'].iloc[0]==5 and post.overall_status.eq('Pass').all()
    u=pd.read_csv(RUN/'analysis/acquisition_property_uncertainty.csv').set_index('method')
    assert u.loc['FW-AEI','co2_ratio_to_ehvi']<1
    assert u.loc['FW-AEI','vis_ratio_to_ehvi']>1 and u.loc['FW-AEI','tox_ratio_to_ehvi']>1
    print('PASS curated counts, posterior cache, 15 matched-data fits, stability and property-specific uncertainty contracts')


if __name__=='__main__':main()
