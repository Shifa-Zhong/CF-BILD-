"""Low-parameter non-compositional reference; frozen primary GP code is reused."""
import numpy as np
import torch
from hyperopt import hp
from cf_bild.gp_cvloss import GPCrossValidatedOptimizer, _make_base_kernel


class SharedStructuralGP(GPCrossValidatedOptimizer):
    """One structural lengthscale, with the same separate T/P kernel as CF-BILD.

    The structural kernel sees the entire concatenated count fingerprint.
    It has no cation/anion decomposition. Selection, calibration, likelihood,
    numerical solves and final conditioning are inherited unchanged.
    """
    def _get_default_data_dependent_space(self):
        space=super()._get_default_data_dependent_space()
        space.pop('ls_cat');space.pop('ls_an')
        space.pop('ls_cross',None)
        x=np.asarray(self.X_train)
        std=np.std(x,axis=0);positive=std[std>1e-9]
        typical=np.median(positive) if len(positive) else 1.0
        lo,hi=self._lengthscale_bounds or (max(1e-3,typical*.01),max(max(1e-3,typical*.01)*10,typical*100))
        space['ls_structure']=hp.loguniform('ls_structure',np.log(lo),np.log(hi))
        return space

    def _build_kernel(self,params,dtype=torch.float32):
        n_struct=sum(self.compositional_kernel_dims)
        structural=_make_base_kernel(params['kernel_name'],float(params['ls_structure']),
            float(params['kernel_variance']),active_dims=list(range(n_struct)),dtype=dtype)
        if self.extra_dims:
            environment=_make_base_kernel(params['kernel_name'],
                [float(params[f'ls_env_{i}']) for i in range(self.extra_dims)],1.0,
                active_dims=list(range(n_struct,self.num_features)),dtype=dtype)
            return structural*environment
        return structural
