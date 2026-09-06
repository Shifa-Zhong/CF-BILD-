"""Frozen-posterior scoring diagnostics; no refitting or replacement of the primary ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import torch
import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cf_bild.acquisition import (to_maximization,cell_width_expectation,joint_feasibility_probability,
    top_indices,fw_aei_scores,additive_ei_scores,exact_q1_ehvi_scores)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--timing-repeats',type=int,default=0)
    args=parser.parse_args();run=args.run_directory.resolve();src=run/'analysis'
    out=run/'extensions/ranking_diagnostics_2026-09-06';out.mkdir(parents=True,exist_ok=True)
    names=['candidate_predictions.npz','candidate_pairs.csv','acquisition_inputs_revision.json',
        'acquisition_analysis_revision.json','top_candidates_revision.csv']
    before={name:digest(src/name) for name in names}
    with np.load(src/'candidate_predictions.npz',allow_pickle=False) as a:
        mu=to_maximization(a['latent_mu']);sd=a['latent_sigma'].copy()
    settings=json.loads((src/'acquisition_inputs_revision.json').read_text())
    comparison=json.loads((src/'acquisition_analysis_revision.json').read_text())['acquisition_comparison']
    ref=np.asarray(settings['reference_point_maximization_frame']);th=np.asarray(settings['feasibility_thresholds_by_percentile']['75'])
    width=cell_width_expectation(mu,sd,np.broadcast_to(ref,mu.shape),np.full_like(mu,np.inf))
    marginal=norm.sf((th-mu)/sd)
    marginal[:,0]=np.exp(log_ndtr((mu[:,0]-th[0])/sd[:,0])-log_ndtr(mu[:,0]/sd[:,0]))
    joint=joint_feasibility_probability(mu,sd,th)
    np.testing.assert_allclose(marginal.prod(axis=1),joint,rtol=1e-12,atol=1e-15)
    score=width.sum(axis=1)*joint;base=top_indices(score,100)
    assert np.array_equal(base,np.asarray(comparison['FW-AEI']['top_indices']))
    upper=np.exp(log_ndtr((mu[:,0]-1)/sd[:,0])-log_ndtr(mu[:,0]/sd[:,0]))
    shares=width/width.sum(axis=1,keepdims=True)
    cases=[('Original',[1,1,1]),('CO2 x0.5',[.5,1,1]),('CO2 x2',[2,1,1]),
        ('Viscosity x0.5',[1,.5,1]),('Viscosity x2',[1,2,1]),
        ('Toxicity x0.5',[1,1,.5]),('Toxicity x2',[1,1,2]),('CO2 percentage scale',[100,1,1])]
    iqrs=[]
    for prop,col in [('co2','CO2-exp'),('vis','vis'),('tox','Experimental logEC50')]:
        v=np.concatenate([pd.read_csv(run/f'data/{role}_1_group_{prop}.csv')[col].to_numpy() for role in ['train','val']])
        iqrs.append(float(np.percentile(v,75)-np.percentile(v,25)))
    assert min(iqrs)>0
    cases.append(('Non-test IQR normalization',(1/np.asarray(iqrs)).tolist()))
    rows=[];selections=[]
    for name,weights in cases:
        alt=top_indices((width*np.asarray(weights)).sum(axis=1)*joint,100)
        overlap=len(set(base)&set(alt));top10=len(set(base[:10])&set(alt[:10]))
        rows.append({'case':name,'co2_weight':weights[0],'vis_weight':weights[1],'tox_weight':weights[2],
            'top100_overlap':overlap,'top100_jaccard':overlap/(200-overlap),'top10_overlap':top10,
            'mean_joint_feasibility':float(joint[alt].mean())})
        selections.extend({'case':name,'rank':i+1,'candidate_index':int(j)} for i,j in enumerate(alt))
    pd.DataFrame(rows).to_csv(out/'weight_sensitivity.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'weight_selection_indices.csv',index=False)
    methods={name:[np.asarray(comparison[name]['top_indices'])] for name in ['FW-AEI','Additive EI','Analytical q=1 EHVI']}
    methods['Random (5 seeds)']=[np.random.default_rng(seed).choice(len(mu),100,replace=False) for seed in [42,123,7,2024,31415]]
    method_rows=[]
    for name,groups in methods.items():
        means=np.asarray([joint[g].mean() for g in groups])
        item={'method':name,'n_selected_per_run':100,'n_runs':len(groups),
            'mean_joint_feasibility':float(means.mean()),
            'joint_feasibility_mean_se':float(means.std(ddof=1)/np.sqrt(len(means))) if len(means)>1 else 0.,
            'mean_co2_above_one_probability':float(np.mean([upper[g].mean() for g in groups]))}
        for k,prop in enumerate(['co2','vis','tox']):
            item[f'mean_{prop}_feasibility']=float(np.mean([marginal[g,k].mean() for g in groups]))
            item[f'mean_{prop}_additive_ei_share']=float(np.mean([shares[g,k].mean() for g in groups]))
        method_rows.append(item)
    pd.DataFrame(method_rows).to_csv(out/'method_preference_probabilities.csv',index=False)
    upper_summary={}
    for name,indices in [('all_candidates',np.arange(len(mu))),('FW_AEI_top100',base)]:
        u=upper[indices]
        upper_summary[name]={'mean_probability':float(u.mean()),'maximum_probability':float(u.max()),
            'count_above_1pct_probability':int((u>.01).sum()),'n_candidates':len(indices)}
    np.savez_compressed(out/'candidate_diagnostics.npz',marginal_preference_probability=marginal,
        joint_preference_probability=joint,marginal_ei_contribution=width,co2_above_one_probability=upper)
    payload={'scope':'Post hoc descriptive sensitivity analyses of frozen main-model predictions; no test-driven model replacement',
        'source_sha256':before,'reference':ref.tolist(),'thresholds':th.tolist(),'non_test_target_iqrs':iqrs,
        'weight_cases':rows,'methods':method_rows,'upper_support':upper_summary,
        'interpretation':'Probabilities are model-based satisfaction of data-defined preferences, not empirical process success. EI shares are score arithmetic, not physical or causal importance.',
        'scale_note':'Only positive objective scales are varied; applying the same transform to each mean, sigma, reference and threshold leaves feasibility unchanged and multiplies its EI. IQRs use non-test targets only.',
        'random_seeds':[42,123,7,2024,31415]}
    write(out/'RANKING_DIAGNOSTICS.json',payload)
    print(json.dumps({'weights':rows,'methods':method_rows,'upper_support':upper_summary}),flush=True)
    if args.timing_repeats:
        assert args.timing_repeats>=3,'Use at least three complete timing repetitions'
        front=np.asarray(settings['incumbent_pareto_front_maximization_frame'])
        functions={'FW-AEI':lambda:fw_aei_scores(mu,sd,ref,th),
            'Additive EI':lambda:additive_ei_scores(mu,sd,ref),
            'Analytical q=1 EHVI':lambda:exact_q1_ehvi_scores(mu,sd,ref,front)[0]}
        times={name:[] for name in functions}
        # One unreported full-pool warm-up per implementation; all predictions are in memory.
        for name,fn in functions.items():
            v=fn();assert np.isfinite(v).all()
            assert np.array_equal(top_indices(v,100),np.asarray(comparison[name]['top_indices']))
            print('Timing warm-up checked: '+name,flush=True)
        order=list(functions)
        for repeat in range(args.timing_repeats):
            for name in order[repeat%3:]+order[:repeat%3]:
                start=time.perf_counter();v=functions[name]();elapsed=time.perf_counter()-start
                times[name].append(elapsed);print(f'Timing {repeat+1} {name}: {elapsed:.6f} s',flush=True)
        write(out/'SCORING_TIMINGS.json',{'processor':platform.processor(),'platform':platform.platform(),
            'python':platform.python_version(),'n_candidates':len(mu),'n_incumbent_pareto':len(front),
            'n_hypercells':comparison['n_ehvi_hypercells'],'repeats':args.timing_repeats,
            'includes':'full-pool acquisition evaluation; EHVI includes front partitioning',
            'excludes':'GP fitting/prediction, input/output, Pareto assessment and stability filtering',
            'warning':'Wall time depends on this implementation, hardware and incumbent front; not a general algorithmic speed guarantee.',
            'methods':{name:{'seconds':values,'median_seconds':float(np.median(values)),'min_seconds':min(values),'max_seconds':max(values)} for name,values in times.items()}})
    assert before=={name:digest(src/name) for name in names},'Frozen sources must not change'


if __name__=='__main__':main()
