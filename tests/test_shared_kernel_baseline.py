"""Small CPU tests for the additional low-dimensional comparator."""
import sys
from pathlib import Path
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cf_bild.shared_kernel_baseline import SharedStructuralGP


def main():
    for env in [0,2]:
        x=np.random.default_rng(2).normal(size=(12,5+env));y=np.sin(x[:,0])
        opt=SharedStructuralGP(x,y,compositional_kernel_dims=(2,3),kernel_form='single_structural_shared',device=torch.device('cpu'))
        expected={'ls_structure','kernel_variance','likelihood_noise_variance','kernel_name'}|{f'ls_env_{i}' for i in range(env)}
        assert set(opt.hyperopt_space)==expected
        p={'ls_structure':3.,'kernel_variance':2.,'likelihood_noise_variance':.01,'kernel_name':'RBF',**{f'ls_env_{i}':1. for i in range(env)}}
        k=opt._build_kernel(p);xx=torch.tensor(x,dtype=torch.float32)
        actual=k(xx).to_dense().detach().numpy()
        distance=((x[:,None,:5]-x[None,:,:5])**2).sum(axis=-1)/9
        if env:distance+=((x[:,None,5:]-x[None,:,5:])**2).sum(axis=-1)
        np.testing.assert_allclose(actual,2*np.exp(-.5*distance),rtol=2e-6,atol=2e-6)
        assert np.linalg.eigvalsh(actual).min()>-1e-5
        # A permutation across the old cation/anion boundary cannot change a single structural kernel.
        perm=[4,1,2,3,0]+list(range(5,5+env))
        np.testing.assert_allclose(k(xx[:,perm]).to_dense().detach().numpy(),actual,rtol=2e-6,atol=2e-6)
    print('PASS shared structural kernel formula, environmental separation, parameter count and ion-boundary invariance')


if __name__=='__main__':main()
