"""Reconstruct trusted saved GP posteriors and compare their test predictions.

Pickle executes code: use only trusted, hash-verified artifacts. This command
does not retune hyperparameters or change the archived fitted outputs.
"""
import argparse
import gc
import hashlib
import json
import pickle
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cf_bild.fragment_vocab import FragmentVocabulary,load_property_datasets,prepare_cv_splits
from cf_bild.gp_cvloss import GPCrossValidatedOptimizer
from cf_bild.predictive import physical_prediction


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run,forms,properties):
    run=Path(run).resolve();results=[]
    vocab=FragmentVocabulary()
    expected=json.loads((run/'VOCABULARY_SUMMARY.json').read_text())['vocabulary_sha256']
    if sha(run/'fragment_vocab.pkl')!=expected:raise ValueError('Vocabulary hash differs')
    vocab.load(run/'fragment_vocab.pkl')
    datasets=load_property_datasets(run/'data',n_folds=5)
    for form in forms:
        for prop in properties:
            directory=run/'results'/form
            manifest=json.loads((directory/f'fit_manifest_{prop}.json').read_text())
            for name,digest in manifest['artifacts_sha256'].items():
                if sha(directory/name)!=digest:raise ValueError('Artifact hash mismatch: '+name)
            for name,digest in manifest['signature']['inputs'].items():
                if sha(run/'data'/name)!=digest:raise ValueError('Input hash mismatch: '+name)
            with (directory/f'models/model_{prop}.pkl').open('rb') as handle:state=pickle.load(handle)
            cv,x,y,xt,yt,scaler=prepare_cv_splits(datasets,vocab,prop)
            np.testing.assert_allclose(scaler.mean_,state['scaler'].mean_,rtol=0,atol=1e-12)
            np.testing.assert_allclose(scaler.scale_,state['scaler'].scale_,rtol=0,atol=1e-12)
            optimizer=GPCrossValidatedOptimizer(x,y,predefined_cv_splits=cv,
                compositional_kernel_dims=(vocab.cat_fp_length,vocab.an_fp_length) if form!='standard' else None,
                kernel_form=form,random_state=42)
            optimizer.best_params=state['best_params']
            # LOVE's Lanczos variance cache depends on RNG state. Restore the
            # final search checkpoint and replay the same five calibrations;
            # parameter states alone do not capture that numerical cache.
            checkpoint_path=directory/f'search_{prop}/search_checkpoint.pkl'
            with checkpoint_path.open('rb') as handle:checkpoint=pickle.load(handle)
            if checkpoint['signature']!=manifest['signature']:raise ValueError('Search signature differs')
            torch.set_rng_state(checkpoint['torch_rng_state'])
            if torch.cuda.is_available() and checkpoint['cuda_rng_state']:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            optimizer._calibrate_variance()
            np.testing.assert_allclose(optimizer.variance_scale_,state['variance_scale'],rtol=1e-6,atol=1e-7)
            with np.load(directory/f'calibration_{prop}.npz',allow_pickle=False) as recorded:
                np.testing.assert_allclose(optimizer.calibration_residuals_,recorded['absolute_residual'],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(optimizer.calibration_raw_std_,recorded['raw_std'],rtol=1e-5,atol=1e-6)
            optimizer.refit_best_model()
            optimizer.best_model_.load_state_dict(state['model_state'])
            optimizer.best_likelihood_.load_state_dict(state['likelihood_state'])
            mu,var=optimizer.predict(xt);mu,var=mu.ravel(),var.ravel()
            pred=physical_prediction(prop,mu,var)
            reference=pd.read_csv(directory/f'test_predictions_{prop}.csv')
            errors={}
            for name,actual in [('latent_mu',mu),('latent_std',np.sqrt(var)),('pred_mean',pred['mean']),('pred_std',pred['std'])]:
                baseline=reference[name].to_numpy()
                errors[name+'_maximum_absolute_difference']=float(np.abs(actual-baseline).max())
                np.testing.assert_allclose(actual,baseline,rtol=1e-5,atol=1e-6)
            results.append({'form':form,'property':prop,'passed':True,**errors})
            print(json.dumps(results[-1]),flush=True)
            del optimizer,cv,x,y,xt
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    payload={'status':'passed','trusted_pickle_required':True,'comparison_rtol':1e-5,'comparison_atol':1e-6,
             'reconstruction':'restore end-of-search RNG state, replay five-fold calibration, condition on full pool, then predict; no retuning',
             'numerical_note':'GPyTorch fast_pred_var uses a Lanczos variance approximation; saved parameters alone omit its random numerical cache',
             'checks':results}
    (run/'analysis/SAVED_MODEL_RECONSTRUCTION.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--kernel-forms',nargs='+',default=['product'])
    parser.add_argument('--properties',nargs='+',choices=['co2','vis','tox'],default=['co2','vis','tox'])
    args=parser.parse_args();main(args.run_directory,args.kernel_forms,args.properties)
