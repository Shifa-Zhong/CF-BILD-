"""
Drop-k_cross ablation: refit GP with kernel = k_cat·k_an (no k_cross term),
under the same TPE budget and species-grouped CV splits used for the main
results. Compare to product (k_cat·k_an + k_cross) and additive (k_cat + k_an + k_cross).

Implementation: runtime monkey-patch of bayesian_gp_cvloss.optimizer._build_compositional_kernel
to add two new kernel_form options:
  - "product_no_cross"  -> ProductKernel(k_cat, k_an)
  - "additive_no_cross" -> k_cat + k_an
"""
import os, sys, json, time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# Import torch BEFORE rdkit on Windows
import torch  # noqa: F401
import gpytorch
from bayesian_gp_cvloss import optimizer as bgc_opt

# ==========================================================================
# Monkey-patch: extend _build_compositional_kernel with no-cross variants
# ==========================================================================
_original_build = bgc_opt._build_compositional_kernel

def _make_base_kernel(kernel_name, lengthscale, variance, active_dims=None, dtype=torch.float32):
    return bgc_opt._make_base_kernel(kernel_name, lengthscale, variance, active_dims=active_dims, dtype=dtype)

def _build_compositional_kernel_extended(kernel_name, params, dim_cat, dim_an,
                                          kernel_form, num_features, dtype=torch.float32):
    """Extended kernel builder supporting product_no_cross / additive_no_cross."""
    if kernel_form in ("product", "additive"):
        return _original_build(kernel_name, params, dim_cat, dim_an,
                                kernel_form, num_features, dtype=dtype)

    total_dims = num_features or (dim_cat + dim_an)
    extra_dims = total_dims - dim_cat - dim_an
    cat_dims = list(range(dim_cat))
    an_dims = list(range(dim_cat, dim_cat + dim_an))
    env_dims = list(range(dim_cat + dim_an, total_dims)) if extra_dims > 0 else []

    ls_cat = float(params['ls_cat'])
    ls_an = float(params['ls_an'])
    variance = float(params['kernel_variance'])

    # For 2-component kernel, split variance differently
    var_component = variance ** 0.5 if kernel_form == "product_no_cross" else variance / 2.0

    k_cat = _make_base_kernel(kernel_name, ls_cat, var_component, active_dims=cat_dims, dtype=dtype)
    k_an = _make_base_kernel(kernel_name, ls_an, var_component, active_dims=an_dims, dtype=dtype)

    if kernel_form == "product_no_cross":
        k_struct = gpytorch.kernels.ProductKernel(k_cat, k_an)
    elif kernel_form == "additive_no_cross":
        k_struct = k_cat + k_an
    else:
        raise ValueError(f"Unknown kernel_form: {kernel_form}")

    if extra_dims > 0:
        env_ls = [float(params.get(f'ls_env_{i}', 1.0)) for i in range(len(env_dims))]
        k_env = _make_base_kernel(kernel_name, env_ls, 1.0, active_dims=env_dims, dtype=dtype)
        return gpytorch.kernels.ProductKernel(k_struct, k_env)
    return k_struct

# Install monkey-patch
bgc_opt._build_compositional_kernel = _build_compositional_kernel_extended
print("Monkey-patched bayesian_gp_cvloss._build_compositional_kernel")

# ==========================================================================
# Reuse the train_and_evaluate machinery from run_ablation
# ==========================================================================
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts', 'pipeline'))
import run_ablation as ab
import logging
ab.logger.setLevel(logging.INFO)

from cf_bild.fragment_vocab import (
    FragmentVocabulary, load_property_datasets, prepare_cv_splits
)

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')


def main():
    print("=" * 70)
    print("DROP-k_cross ABLATION")
    print("=" * 70)

    # Load datasets + fragment vocab
    print("\nLoading fragment vocabulary...")
    import pickle
    with open(os.path.join(OUTPUT_DIR, 'fragment_vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)

    vocab = FragmentVocabulary()
    vocab.cation_smiles_to_idx = vocab_data['cations']
    vocab.anion_smiles_to_idx = vocab_data['anions']
    vocab.cation_fps = vocab_data['cation_fps']
    vocab.anion_fps = vocab_data['anion_fps']
    vocab.cat_fp_length = vocab_data['cat_fp_length']
    vocab.an_fp_length = vocab_data['an_fp_length']
    vocab.fp_radius = vocab_data['fp_radius']
    vocab.cfmf_cat = vocab_data['cfmf_cat']
    vocab.cfmf_an = vocab_data['cfmf_an']

    print(f"  Cations: {len(vocab.cation_smiles_to_idx)}, "
          f"Anions: {len(vocab.anion_smiles_to_idx)}")

    print("\nLoading property datasets...")
    datasets = load_property_datasets(DATA_DIR, n_folds=5)
    for prop in ['co2', 'vis', 'tox']:
        d = datasets.get(prop, {})
        print(f"  {prop}: keys={list(d.keys())[:5]}")

    # Run the two no-cross ablations × 3 properties = 6 GP fits
    results = {}
    for kernel_form in ['product_no_cross', 'additive_no_cross']:
        results[kernel_form] = {}
        for prop in ['co2', 'vis', 'tox']:
            print(f"\n--- {kernel_form} / {prop} ---")
            t0 = time.time()
            try:
                metrics, _, _ = ab.train_and_evaluate(
                    datasets, vocab, prop,
                    kernel_form=kernel_form,
                    compositional=True,
                    max_evals=3000,
                    patience=50,
                )
                elapsed = time.time() - t0
                metrics['elapsed_seconds'] = elapsed
                results[kernel_form][prop] = metrics
                r2 = metrics.get('R2', float('nan'))
                rmse = metrics.get('RMSE', float('nan'))
                cv = metrics.get('CV_RMSE', float('nan'))
                print(f"  R²={r2:.4f}  RMSE={rmse:.4f}  CV_RMSE={cv:.4f}  ({elapsed:.0f}s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                results[kernel_form][prop] = {'error': str(e)}

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'drop_cross_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # ===== Compare to existing product/additive baselines =====
    abl_path = os.path.join(OUTPUT_DIR, 'ablation_results.json')
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            existing = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        print(f'{"Kernel form":<26} {"R²(CO2)":<10} {"R²(Vis)":<10} {"R²(Tox)":<10}')
        print("-" * 60)
        for form in ['product', 'additive', 'standard']:
            row = existing['A_kernel_form'][form]
            r2_co2 = row['co2'].get('R2', float('nan'))
            r2_vis = row['vis'].get('R2', float('nan'))
            r2_tox = row['tox'].get('R2', float('nan'))
            print(f'{form:<26} {r2_co2:<10.4f} {r2_vis:<10.4f} {r2_tox:<10.4f}')
        for form in ['product_no_cross', 'additive_no_cross']:
            r = results[form]
            r2_co2 = r.get('co2', {}).get('R2', float('nan'))
            r2_vis = r.get('vis', {}).get('R2', float('nan'))
            r2_tox = r.get('tox', {}).get('R2', float('nan'))
            print(f'{form:<26} {r2_co2:<10.4f} {r2_vis:<10.4f} {r2_tox:<10.4f}')

        # ΔR² product vs product_no_cross
        print("\nΔR² (product − product_no_cross):")
        prod = existing['A_kernel_form']['product']
        no = results.get('product_no_cross', {})
        for prop in ['co2', 'vis', 'tox']:
            d = prod[prop]['R2'] - no.get(prop, {}).get('R2', float('nan'))
            print(f'  {prop}:  Δ = {d:+.4f}')


if __name__ == '__main__':
    main()
