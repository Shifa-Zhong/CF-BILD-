"""
Cache GP predictions on all 87,365 candidates without re-running TPE.
Reuses best_params from saved model_*.pkl and the refit_best_model() path.
"""
import os, sys, pickle, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import torch  # before rdkit
from bayesian_gp_cvloss import GPCrossValidatedOptimizer
from cf_bild.fragment_vocab import FragmentVocabulary, load_property_datasets, prepare_cv_splits

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def main():
    print('Loading datasets...')
    datasets = load_property_datasets(os.path.join(BASE_DIR, 'data'), n_folds=5)
    print('Loading fragment vocab...')
    with open(os.path.join(OUTPUT_DIR, 'fragment_vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)

    vocab = FragmentVocabulary(fp_radius=2)
    # Rebuild vocab the same way run_exact_ehvi does
    all_dfs = []
    for prop, data_dict in datasets.items():
        for t, v in data_dict['folds']:
            all_dfs.extend([t, v])
        if data_dict['test'] is not None:
            all_dfs.append(data_dict['test'])
    vocab.extract_fragments_from_dataframes(all_dfs)
    vocab.compute_fingerprints()
    dim_cat = vocab.cat_fp_length
    dim_an = vocab.an_fp_length
    print(f'  cat_dim={dim_cat}, an_dim={dim_an}')

    # Candidate matrix
    cation_list = list(vocab_data['cation_fps'].keys())
    anion_list = list(vocab_data['anion_fps'].keys())
    n_cat, n_an = len(cation_list), len(anion_list)
    n_candidates = n_cat * n_an
    cat_fps = np.array([vocab_data['cation_fps'][c] for c in cation_list], dtype=np.float64)
    an_fps = np.array([vocab_data['anion_fps'][a] for a in anion_list], dtype=np.float64)
    print(f'  candidates: {n_candidates}')

    all_mu = {}
    all_sigma = {}
    for prop in ['co2', 'vis', 'tox']:
        print(f'\n=== {prop} ===')
        t0 = time.time()
        # Load saved best params
        with open(os.path.join(OUTPUT_DIR, f'model_{prop}.pkl'), 'rb') as f:
            saved = pickle.load(f)
        best_params = saved['best_params']
        var_scale = float(saved['variance_scale'])
        scaler = saved['scaler']

        # Prepare same splits so optimizer has the right training data shape
        cv_splits, X_all, y_all, X_test, y_test, scaler2 = prepare_cv_splits(
            datasets, vocab, prop)

        optimizer = GPCrossValidatedOptimizer(
            X_train=X_all, y_train=y_all,
            kernels=None, n_splits=5, random_state=42,
            predefined_cv_splits=cv_splits,
            compositional_kernel_dims=(dim_cat, dim_an),
            kernel_form="product",
        )
        # Inject best_params (skip TPE)
        optimizer.best_params = best_params
        optimizer.refit_best_model()
        print(f'  refit done ({time.time()-t0:.1f}s)')

        # Build candidate matrix
        mu_list, sigma_list = [], []
        batch_size = 5000
        for start in range(0, n_candidates, batch_size):
            end = min(start + batch_size, n_candidates)
            cat_idx = np.arange(start, end) // n_an
            an_idx = np.arange(start, end) % n_an
            X_batch = np.concatenate([cat_fps[cat_idx], an_fps[an_idx]], axis=1)
            if prop in ('co2', 'vis'):
                env = np.full((end - start, 2), [298.15, 101.325], dtype=np.float64)
                X_batch = np.concatenate([X_batch, env], axis=1)
            X_batch = scaler.transform(X_batch)
            pm, pv = optimizer.predict(X_batch)
            mu_list.append(pm.flatten())
            sigma_list.append(np.sqrt(pv.flatten()))

        mu = np.concatenate(mu_list)
        sigma = np.concatenate(sigma_list) * np.sqrt(var_scale)  # apply post-hoc calibration
        all_mu[prop] = mu
        all_sigma[prop] = sigma
        print(f'  predict done ({time.time()-t0:.1f}s total)')
        print(f'  mu: [{mu.min():.3f}, {mu.max():.3f}], '
              f'sigma (cal): [{sigma.min():.3f}, {sigma.max():.3f}], scale={var_scale:.3f}')

    mu_matrix = np.column_stack([all_mu['co2'], all_mu['vis'], all_mu['tox']])
    sigma_matrix = np.column_stack([all_sigma['co2'], all_sigma['vis'], all_sigma['tox']])
    ref_point = np.array([
        mu_matrix[:, 0].min(),
        mu_matrix[:, 1].max(),
        mu_matrix[:, 2].min(),
    ], dtype=np.float64)
    maximize_flags = [True, False, True]

    out = {
        'mu': mu_matrix, 'sigma': sigma_matrix,
        'ref_point': ref_point, 'maximize_flags': maximize_flags,
        'candidate_cation_anion': [(c, a) for c in cation_list for a in anion_list],
    }
    cache_path = os.path.join(OUTPUT_DIR, 'predictions_87365.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'\nSaved {cache_path}')
    print(f'mu_matrix shape: {mu_matrix.shape}')
    print(f'ref_point: {ref_point}')


if __name__ == '__main__':
    main()
