"""Fit an additive, isolated low-parameter comparator without changing primary fits."""
import argparse
import gc
import json
import logging
from pathlib import Path
import pickle
import sys
import time
import torch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cf_bild.fragment_vocab import FragmentVocabulary,load_property_datasets,prepare_cv_splits
from cf_bild.shared_kernel_baseline import SharedStructuralGP
from cf_bild.predictive import physical_prediction,regression_metrics,viscosity_real_space_metrics
from run_clean_refit import digest,write_json,optimize_checkpointed
from run_revision_models import conditional_nlpd,cpu_state_dict,environment_versions


def signature(run,prop,args):
    return {'inputs':{p.name:digest(p) for p in sorted((run/'data').glob(f'*_group_{prop}.csv'))},
        'vocabulary':digest(run/'fragment_vocab.pkl'),'form':'single_structural_shared',
        'max_evals':args.max_evals,'patience':args.patience,'seed':args.seed,
        'code':{name:digest(ROOT/name) for name in ['cf_bild/gp_cvloss.py','cf_bild/shared_kernel_baseline.py',
            'cf_bild/fragment_vocab.py','scripts/revision/run_clean_refit.py','scripts/revision/run_shared_kernel_baseline.py']},
        'continuous_parameters':5 if prop in ('co2','vis') else 3,
        'screening_models_changed':False,'test_used_in_selection':False}


def fit_or_verify(run,out,datasets,vocab,prop,args):
    meta=out/f'fit_manifest_{prop}.json';sig=signature(run,prop,args)
    if meta.exists():
        old=json.loads(meta.read_text())
        assert old['signature']==sig,'Existing fit has different input/protocol/code'
        assert all(digest(out/name)==checksum for name,checksum in old['artifacts_sha256'].items())
        if not args.verify_only:
            print('Verified completed comparator '+prop,flush=True);return
    elif args.verify_only:raise ValueError('Missing completed comparator '+prop)
    cv,x,y,xt,yt,scaler=prepare_cv_splits(datasets,vocab,prop)
    optimizer=SharedStructuralGP(x,y,predefined_cv_splits=cv,
        compositional_kernel_dims=(vocab.cat_fp_length,vocab.an_fp_length),
        kernel_form='single_structural_shared',random_state=args.seed)
    started=time.time()
    if args.verify_only:
        # Trusted release pickle only, after signature/artifact integrity checks.
        checkpoint=out/f'search_{prop}/search_checkpoint.pkl'
        assert digest(checkpoint)==old['checkpoint_sha256']
        saved=pickle.loads(checkpoint.read_bytes());assert saved['signature']==sig
        state=pickle.loads((out/f'models/model_{prop}.pkl').read_bytes())
        optimizer.best_params=state['best_params']
        torch.set_rng_state(saved['torch_rng_state'])
        if torch.cuda.is_available() and saved['cuda_rng_state']:torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        optimizer._calibrate_variance();optimizer.refit_best_model()
        np.testing.assert_allclose(optimizer.variance_scale_,state['variance_scale'],rtol=1e-6,atol=1e-7)
        optimizer.best_model_.load_state_dict(state['model_state'])
        optimizer.best_likelihood_.load_state_dict(state['likelihood_state'])
    else:
        optimize_checkpointed(optimizer,out/f'search_{prop}',sig,args.max_evals,args.patience,args.seed)
    mu,var=optimizer.predict(xt);mu,var=mu.ravel(),var.ravel()
    pred=physical_prediction(prop,mu,var)
    table=pd.DataFrame({'source_record_id':datasets[prop]['test']['ind'].to_numpy(),'y_true':yt,
        'latent_mu':mu,'latent_std':np.sqrt(var),'pred_mean':pred['mean'],'pred_std':pred['std'],
        'lower_95':pred['lower'],'upper_95':pred['upper']})
    if args.verify_only:
        recorded=pd.read_csv(out/f'test_predictions_{prop}.csv');errors={}
        for col in table:
            np.testing.assert_allclose(table[col],recorded[col],rtol=1e-5,atol=1e-6)
            errors[col]=float(np.abs(table[col]-recorded[col]).max())
        write_json(out/f'RECONSTRUCTION_{prop}.json',{'status':'passed','rtol':1e-5,'atol':1e-6,
            'maximum_absolute_differences':errors,'test_predictions_sha256':digest(out/f'test_predictions_{prop}.csv')})
        print('PASS numerical reconstruction '+prop,flush=True)
    else:
        metrics=regression_metrics(yt,pred)
        metrics.update({'nlpd':conditional_nlpd(prop,yt,mu,var),'n_refit':len(y),'n_test':len(yt),
            'variance_scale':float(optimizer.variance_scale_),'n_search_trials':len(optimizer.trials),
            'best_cv_rmse':float(optimizer.trials.best_trial['result']['loss']),
            'elapsed_seconds':time.time()-started,'continuous_parameters':sig['continuous_parameters']})
        if prop=='vis':metrics['real_space']=viscosity_real_space_metrics(yt,pred['mean'])
        table.to_csv(out/f'test_predictions_{prop}.csv',index=False)
        write_json(out/f'metrics_{prop}.json',metrics)
        selected={'best_params':optimizer.best_params,'variance_scale':float(optimizer.variance_scale_)}
        write_json(out/f'selected_hyperparameters_{prop}.json',selected)
        write_json(out/f'calibration_{prop}.json',optimizer.calibration_diagnostics_)
        np.savez_compressed(out/f'calibration_{prop}.npz',absolute_residual=optimizer.calibration_residuals_,raw_std=optimizer.calibration_raw_std_)
        state={**selected,'scaler':scaler,'property':prop,'n_refit':len(y),
            'dim_cat':vocab.cat_fp_length,'dim_an':vocab.an_fp_length,
            'model_state':cpu_state_dict(optimizer.best_model_),
            'likelihood_state':cpu_state_dict(optimizer.best_likelihood_)}
        (out/'models').mkdir(exist_ok=True)
        with (out/f'models/model_{prop}.pkl').open('wb') as f:pickle.dump(state,f,protocol=pickle.HIGHEST_PROTOCOL)
        artifacts=[f'metrics_{prop}.json',f'test_predictions_{prop}.csv',f'selected_hyperparameters_{prop}.json',
            f'calibration_{prop}.json',f'calibration_{prop}.npz',f'models/model_{prop}.pkl']
        write_json(meta,{'signature':sig,'artifacts_sha256':{name:digest(out/name) for name in artifacts},
            'checkpoint_sha256':digest(out/f'search_{prop}/search_checkpoint.pkl')})
        print(json.dumps({'property':prop,'metrics':metrics}),flush=True)
    del optimizer,cv,x,y,xt;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path)
    parser.add_argument('--properties',nargs='+',choices=['co2','vis','tox'],default=['tox','co2','vis'])
    parser.add_argument('--max-evals',type=int,default=3000)
    parser.add_argument('--patience',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();run=args.run_directory.resolve()
    out=(args.output_directory or run/'extensions/low_parameter_2026-09-06').resolve();out.mkdir(parents=True,exist_ok=True)
    curation=json.loads((run/'CURATION_SUMMARY.json').read_text())
    assert all(digest(run/'data'/name)==value for name,value in curation['cleaned_sha256'].items())
    if not args.verify_only:write_json(out/'COMPUTATIONAL_ENVIRONMENT.json',environment_versions())
    logging.basicConfig(level=logging.WARNING)
    datasets=load_property_datasets(run/'data',n_folds=5)
    vocab=FragmentVocabulary();vocab.load(run/'fragment_vocab.pkl')
    for prop in args.properties:fit_or_verify(run,out,datasets,vocab,prop,args)


if __name__=='__main__':main()
