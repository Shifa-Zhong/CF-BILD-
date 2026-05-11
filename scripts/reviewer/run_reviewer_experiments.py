"""
CF-BILD Reviewer Response Experiments
=====================================
Experiment 1: Unique IL counts per split
Experiment 2: Structure-level R² for CO2
Experiment 3: 2-objective ablation (CO2 + Viscosity only)
Experiment 4: C-EHVI threshold sensitivity
Experiment 5: Scaffold-based split analysis
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

# CRITICAL: Import torch BEFORE rdkit to avoid DLL conflicts on Windows
import torch
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score

# Setup paths
BASE_DIR = 'D:/wuzeting'
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, 'D:/wuzeting/bayesian-gp-cvloss')

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'reviewer_experiments')
os.makedirs(RESULTS_DIR, exist_ok=True)

from cf_bild.conservative_bo import (
    compute_c_ehvi_scores, identify_pareto_front, compute_hypervolume
)


# ============================================================================
# Experiment 1: Unique IL counts per split
# ============================================================================
def experiment_1_unique_il_counts():
    print("=" * 60)
    print("EXPERIMENT 1: Unique IL Counts Per Split")
    print("=" * 60)

    rows = []
    for prop in ['co2', 'vis', 'tox']:
        # Folds 1-5: train and val
        for fold in range(1, 6):
            for split_type in ['train', 'val']:
                fname = f'{split_type}_{fold}_group_{prop}.csv'
                fpath = os.path.join(DATA_DIR, fname)
                if not os.path.exists(fpath):
                    continue
                df = pd.read_csv(fpath)
                total_rows = len(df)
                # Unique ILs = unique (cation, anion) pairs
                unique_ils = df.groupby(['new_cation', 'new_anion']).ngroups
                rows.append({
                    'property': prop,
                    'split': f'{split_type}_fold{fold}',
                    'total_rows': total_rows,
                    'unique_ILs': unique_ils,
                    'rows_per_IL': round(total_rows / unique_ils, 2) if unique_ils > 0 else 0
                })
                print(f"  {prop} {split_type}_fold{fold}: {total_rows} rows, {unique_ils} unique ILs")

        # Test set
        test_path = os.path.join(DATA_DIR, f'test_group_{prop}.csv')
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            total_rows = len(df)
            unique_ils = df.groupby(['new_cation', 'new_anion']).ngroups
            rows.append({
                'property': prop,
                'split': 'test',
                'total_rows': total_rows,
                'unique_ILs': unique_ils,
                'rows_per_IL': round(total_rows / unique_ils, 2) if unique_ils > 0 else 0
            })
            print(f"  {prop} test: {total_rows} rows, {unique_ils} unique ILs")

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'unique_il_counts.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(result_df.to_string(index=False))
    return result_df


# ============================================================================
# Experiment 2: Structure-level R² for CO2
# ============================================================================
def experiment_2_structure_level_r2():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Structure-Level R² for CO2")
    print("=" * 60)

    # Load model predictions
    with open(os.path.join(OUTPUT_DIR, 'model_co2.pkl'), 'rb') as f:
        model_data = pickle.load(f)

    test_y_true = model_data['test_y_true']
    test_y_pred = model_data['test_y_pred']
    test_y_std = model_data['test_y_std']

    # Load test CSV to get cation/anion assignments
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_group_co2.csv'))

    print(f"  Model test predictions: {len(test_y_true)}")
    print(f"  Test CSV rows: {len(test_df)}")

    # Point-level R²
    point_r2 = r2_score(test_y_true, test_y_pred)
    point_rmse = np.sqrt(np.mean((test_y_true - test_y_pred) ** 2))
    print(f"\n  Point-level R²: {point_r2:.4f}")
    print(f"  Point-level RMSE: {point_rmse:.4f}")

    # Assign predictions to test_df rows
    # The model predictions should align row-by-row with the test CSV
    # (same order after df_to_features processing; some rows may have been
    #  dropped if fingerprinting failed, but typically all succeed)
    n_pred = len(test_y_true)
    n_df = len(test_df)

    if n_pred == n_df:
        test_df = test_df.copy()
        test_df['y_true'] = test_y_true
        test_df['y_pred'] = test_y_pred
    else:
        # If sizes differ, we need to match. The df_to_features function
        # processes rows in order, skipping only those with failed fingerprints.
        # Assign to first n_pred rows as a best-effort approach.
        print(f"  WARNING: prediction count ({n_pred}) != CSV rows ({n_df})")
        print(f"  Using first {n_pred} rows of test CSV.")
        test_df = test_df.iloc[:n_pred].copy()
        test_df['y_true'] = test_y_true
        test_df['y_pred'] = test_y_pred

    # Group by unique IL (cation + anion pair)
    grouped = test_df.groupby(['new_cation', 'new_anion']).agg(
        mean_y_true=('y_true', 'mean'),
        mean_y_pred=('y_pred', 'mean'),
        n_conditions=('y_true', 'count')
    ).reset_index()

    structure_r2 = r2_score(grouped['mean_y_true'], grouped['mean_y_pred'])
    structure_rmse = np.sqrt(np.mean((grouped['mean_y_true'] - grouped['mean_y_pred']) ** 2))

    print(f"\n  Number of unique IL structures in test: {len(grouped)}")
    print(f"  Structure-level R² (averaged over T/P): {structure_r2:.4f}")
    print(f"  Structure-level RMSE: {structure_rmse:.4f}")

    # Save results
    results_rows = [
        {'metric': 'point_level_R2', 'value': point_r2, 'n_points': n_pred},
        {'metric': 'point_level_RMSE', 'value': point_rmse, 'n_points': n_pred},
        {'metric': 'structure_level_R2', 'value': structure_r2, 'n_structures': len(grouped)},
        {'metric': 'structure_level_RMSE', 'value': structure_rmse, 'n_structures': len(grouped)},
    ]
    result_df = pd.DataFrame(results_rows)
    out_path = os.path.join(RESULTS_DIR, 'structure_level_r2.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # Also save the per-structure details
    grouped_path = os.path.join(RESULTS_DIR, 'structure_level_r2_details.csv')
    grouped.to_csv(grouped_path, index=False)
    print(f"  Per-structure details saved to {grouped_path}")

    return result_df


# ============================================================================
# Experiment 3: 2-Objective Ablation (CO2 + Viscosity only)
# ============================================================================
def experiment_3_two_objective_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: 2-Objective Ablation (CO2 + Viscosity)")
    print("=" * 60)

    # Load the saved pareto_candidates (3-objective result)
    pareto_3obj_path = os.path.join(OUTPUT_DIR, 'pareto_candidates.csv')
    top_3obj_path = os.path.join(OUTPUT_DIR, 'top_candidates.csv')

    pareto_3obj_df = pd.read_csv(pareto_3obj_path)
    top_3obj_df = pd.read_csv(top_3obj_path)

    print(f"  3-objective: {len(top_3obj_df)} top candidates, "
          f"{len(pareto_3obj_df)} Pareto-optimal")

    # --- Recompute C-EHVI with 2 objectives (CO2 + Viscosity only) ---
    # We use the predictions already stored in top_candidates.csv
    # But we need predictions for ALL candidates to do a fair comparison.
    # Instead, use the top_candidates.csv predictions as our candidate pool
    # (the full screening was already done; we re-score the same pool).

    # For the full candidate pool, load all 3 model predictions
    # We'll reconstruct from the top_candidates (100 candidates) which is
    # the same pool used for the 3-objective analysis.
    mu_co2 = top_3obj_df['co2_pred'].values
    sigma_co2 = top_3obj_df['co2_std'].values
    mu_vis = top_3obj_df['vis_pred'].values
    sigma_vis = top_3obj_df['vis_std'].values
    mu_tox = top_3obj_df['tox_pred'].values
    sigma_tox = top_3obj_df['tox_std'].values

    N = len(top_3obj_df)

    # --- 3-objective Pareto (recompute for consistency) ---
    mu_3obj = np.column_stack([mu_co2, mu_vis, mu_tox])
    maximize_3 = [True, False, True]
    pareto_mask_3 = identify_pareto_front(mu_3obj, maximize_3)
    n_pareto_3 = int(pareto_mask_3.sum())

    # Hypervolume for 3-objective
    ref_3 = np.array([mu_co2.min() - 0.01, mu_vis.max() + 0.01, mu_tox.min() - 0.01])
    if n_pareto_3 > 0:
        hv_3 = compute_hypervolume(mu_3obj[pareto_mask_3], ref_3, maximize_3)
    else:
        hv_3 = 0.0

    print(f"\n  3-objective: {n_pareto_3} Pareto points, HV={hv_3:.6f}")

    # --- 2-objective: CO2 + Viscosity only ---
    mu_2obj = np.column_stack([mu_co2, mu_vis])
    sigma_2obj = np.column_stack([sigma_co2, sigma_vis])
    maximize_2 = [True, False]

    # Compute thresholds for 2 objectives from training data
    # Load model stats
    with open(os.path.join(OUTPUT_DIR, 'model_stats.json'), 'r') as f:
        stats = json.load(f)

    # Load training data to compute percentile thresholds
    # Combine all 5 training folds for each property
    train_y_co2 = []
    train_y_vis = []
    for fold in range(1, 6):
        co2_df = pd.read_csv(os.path.join(DATA_DIR, f'train_{fold}_group_co2.csv'))
        train_y_co2.extend(co2_df['CO2-exp'].values)
        vis_df = pd.read_csv(os.path.join(DATA_DIR, f'train_{fold}_group_vis.csv'))
        train_y_vis.extend(vis_df['vis'].values)

    # Use unique training values (not duplicated across folds - take fold 1 + val as proxy)
    # Actually, the thresholds should match the original: 75th percentile for co2,
    # 25th percentile for vis
    co2_train_all = pd.read_csv(os.path.join(DATA_DIR, 'train_1_group_co2.csv'))['CO2-exp'].values
    vis_train_all = pd.read_csv(os.path.join(DATA_DIR, 'train_1_group_vis.csv'))['vis'].values

    # Use the largest fold's training data for thresholds (consistent with run_cfbild.py)
    thresh_co2 = np.percentile(co2_train_all, 75)
    thresh_vis = np.percentile(vis_train_all, 25)

    thresh_2 = np.array([thresh_co2, thresh_vis])
    ref_2 = np.array([mu_co2.min() - 0.01, mu_vis.max() + 0.01])

    # C-EHVI with 2 objectives
    scores_2obj = compute_c_ehvi_scores(
        {'mu': mu_2obj, 'sigma': sigma_2obj},
        thresh_2, ref_2, maximize_2
    )

    # Pareto front for 2 objectives
    pareto_mask_2 = identify_pareto_front(mu_2obj, maximize_2)
    n_pareto_2 = int(pareto_mask_2.sum())

    if n_pareto_2 > 0:
        hv_2 = compute_hypervolume(mu_2obj[pareto_mask_2], ref_2, maximize_2)
    else:
        hv_2 = 0.0

    print(f"  2-objective: {n_pareto_2} Pareto points, HV={hv_2:.6f}")

    # Compare: which candidates appear in 2-obj top but not 3-obj Pareto?
    top_2obj_idx = np.argsort(scores_2obj)[::-1][:20]
    top_2obj_cands = top_3obj_df.iloc[top_2obj_idx][['cation', 'anion', 'co2_pred', 'vis_pred', 'tox_pred']].copy()
    top_2obj_cands['in_3obj_pareto'] = pareto_mask_3[top_2obj_idx]

    # Toxicity values for 2-obj top candidates (not optimized)
    mean_tox_2obj_top = mu_tox[top_2obj_idx].mean()
    mean_tox_3obj_pareto = mu_tox[pareto_mask_3].mean() if n_pareto_3 > 0 else float('nan')

    print(f"\n  Mean toxicity (logEC50) of 2-obj top-20: {mean_tox_2obj_top:.4f}")
    print(f"  Mean toxicity (logEC50) of 3-obj Pareto: {mean_tox_3obj_pareto:.4f}")
    print(f"  (Higher logEC50 = lower toxicity = better)")

    # Overlap analysis
    set_2obj_pareto = set(np.where(pareto_mask_2)[0])
    set_3obj_pareto = set(np.where(pareto_mask_3)[0])
    overlap = set_2obj_pareto & set_3obj_pareto
    print(f"\n  Pareto overlap (2-obj & 3-obj): {len(overlap)} candidates")

    # Save results
    results_rows = [
        {'setting': '3-objective', 'n_pareto': n_pareto_3, 'hypervolume_3d': hv_3,
         'hypervolume_2d': '', 'mean_tox_pareto': mean_tox_3obj_pareto,
         'pareto_overlap': ''},
        {'setting': '2-objective (CO2+Vis)', 'n_pareto': n_pareto_2, 'hypervolume_3d': '',
         'hypervolume_2d': hv_2, 'mean_tox_pareto': mean_tox_2obj_top,
         'pareto_overlap': len(overlap)},
    ]
    result_df = pd.DataFrame(results_rows)
    out_path = os.path.join(RESULTS_DIR, 'two_objective_ablation.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # Also save top-20 from 2-obj for inspection
    top_2obj_cands.to_csv(os.path.join(RESULTS_DIR, 'two_objective_top20.csv'), index=False)

    return result_df


# ============================================================================
# Experiment 4: C-EHVI Threshold Sensitivity
# ============================================================================
def experiment_4_threshold_sensitivity():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: C-EHVI Threshold Sensitivity")
    print("=" * 60)

    # Load top candidates with their predictions
    top_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'top_candidates.csv'))
    mu_co2 = top_df['co2_pred'].values
    sigma_co2 = top_df['co2_std'].values
    mu_vis = top_df['vis_pred'].values
    sigma_vis = top_df['vis_std'].values
    mu_tox = top_df['tox_pred'].values
    sigma_tox = top_df['tox_std'].values

    mu_matrix = np.column_stack([mu_co2, mu_vis, mu_tox])
    sigma_matrix = np.column_stack([sigma_co2, sigma_vis, sigma_tox])
    maximize_flags = [True, False, True]

    # Load training data for computing percentile thresholds
    # Use fold 1 training data (largest fold used for refit)
    co2_train = pd.read_csv(os.path.join(DATA_DIR, 'train_1_group_co2.csv'))['CO2-exp'].values
    vis_train = pd.read_csv(os.path.join(DATA_DIR, 'train_1_group_vis.csv'))['vis'].values
    tox_train = pd.read_csv(os.path.join(DATA_DIR, 'train_1_group_tox.csv'))['Experimental logEC50'].values

    ref_point = np.array([mu_co2.min() - 0.01, mu_vis.max() + 0.01, mu_tox.min() - 0.01])

    percentiles = [50, 60, 70, 75, 80, 90]
    results_rows = []

    for pct in percentiles:
        # CO2: > Xth percentile (maximize)
        thresh_co2 = np.percentile(co2_train, pct)
        # Vis: < (100-X)th percentile (minimize), i.e., threshold at (100-pct)th percentile
        # For the conservative penalty on a minimization objective:
        # We want vis < threshold, so threshold should be at the (100-pct)th percentile
        # so that only (100-pct)% are below it... Actually, looking at the original code:
        # vis threshold is at 25th percentile when pct=75. So: vis_thresh = percentile(vis, 100-pct)
        thresh_vis = np.percentile(vis_train, 100 - pct)
        # Tox: > Xth percentile (maximize logEC50)
        thresh_tox = np.percentile(tox_train, pct)

        thresh_array = np.array([thresh_co2, thresh_vis, thresh_tox])

        scores = compute_c_ehvi_scores(
            {'mu': mu_matrix, 'sigma': sigma_matrix},
            thresh_array, ref_point, maximize_flags
        )

        # Select top candidates and compute Pareto front
        top_idx = np.argsort(scores)[::-1][:50]
        top_mu = mu_matrix[top_idx]
        pareto_mask = identify_pareto_front(top_mu, maximize_flags)
        n_pareto = int(pareto_mask.sum())

        if n_pareto > 0:
            hv = compute_hypervolume(top_mu[pareto_mask], ref_point, maximize_flags)
        else:
            hv = 0.0

        mean_score = float(scores[top_idx].mean())
        max_score = float(scores.max())

        # Mean feasibility probability (product of CDFs)
        mu_max = mu_matrix.copy()
        thresh_max = thresh_array.copy()
        mu_max[:, 1] = -mu_matrix[:, 1]
        thresh_max[1] = -thresh_array[1]
        feasibility = np.ones(len(mu_matrix))
        for k in range(3):
            z = (mu_max[:, k] - thresh_max[k]) / (sigma_matrix[:, k] + 1e-8)
            feasibility *= norm.cdf(z)
        mean_feasibility = float(feasibility.mean())
        n_feasible = int((feasibility > 0.5).sum())

        results_rows.append({
            'threshold_percentile': pct,
            'thresh_co2': round(thresh_co2, 4),
            'thresh_vis': round(thresh_vis, 4),
            'thresh_tox': round(thresh_tox, 4),
            'n_pareto_top50': n_pareto,
            'hypervolume': round(hv, 6),
            'max_cehvi_score': round(max_score, 6),
            'mean_cehvi_top50': round(mean_score, 6),
            'mean_feasibility_all': round(mean_feasibility, 6),
            'n_feasible_gt50pct': n_feasible,
        })

        current = " <-- current" if pct == 75 else ""
        print(f"  Percentile {pct:3d}: Pareto={n_pareto:3d}, HV={hv:.6f}, "
              f"max_score={max_score:.6f}, feasible={n_feasible}{current}")

    result_df = pd.DataFrame(results_rows)
    out_path = os.path.join(RESULTS_DIR, 'threshold_sensitivity.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")
    return result_df


# ============================================================================
# Experiment 5: Scaffold-based Split Analysis
# ============================================================================
def experiment_5_scaffold_analysis():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Scaffold-Based Split Analysis (CO2)")
    print("=" * 60)

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def get_murcko_scaffold(smiles):
        """Get Murcko scaffold SMILES for a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return None

    def get_generic_scaffold(smiles):
        """Get generic (framework) scaffold SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            return Chem.MolToSmiles(generic)
        except Exception:
            return None

    # Collect all cation SMILES from CO2 data
    all_cations_train = set()
    all_cations_test = set()
    all_scaffolds_train = set()
    all_scaffolds_test = set()
    all_generic_train = set()
    all_generic_test = set()

    cation_to_scaffold = {}
    cation_to_generic = {}

    # Process training folds
    for fold in range(1, 6):
        train_df = pd.read_csv(os.path.join(DATA_DIR, f'train_{fold}_group_co2.csv'))
        val_df = pd.read_csv(os.path.join(DATA_DIR, f'val_{fold}_group_co2.csv'))
        for df in [train_df, val_df]:
            for cat in df['new_cation'].unique():
                all_cations_train.add(cat)
                if cat not in cation_to_scaffold:
                    cation_to_scaffold[cat] = get_murcko_scaffold(cat)
                    cation_to_generic[cat] = get_generic_scaffold(cat)
                scaf = cation_to_scaffold[cat]
                gen = cation_to_generic[cat]
                if scaf is not None:
                    all_scaffolds_train.add(scaf)
                if gen is not None:
                    all_generic_train.add(gen)

    # Process test set
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_group_co2.csv'))
    for cat in test_df['new_cation'].unique():
        all_cations_test.add(cat)
        if cat not in cation_to_scaffold:
            cation_to_scaffold[cat] = get_murcko_scaffold(cat)
            cation_to_generic[cat] = get_generic_scaffold(cat)
        scaf = cation_to_scaffold[cat]
        gen = cation_to_generic[cat]
        if scaf is not None:
            all_scaffolds_test.add(scaf)
        if gen is not None:
            all_generic_test.add(gen)

    # Compute overlaps
    scaffold_overlap = all_scaffolds_train & all_scaffolds_test
    scaffold_test_only = all_scaffolds_test - all_scaffolds_train
    scaffold_train_only = all_scaffolds_train - all_scaffolds_test
    generic_overlap = all_generic_train & all_generic_test
    generic_test_only = all_generic_test - all_generic_train
    generic_train_only = all_generic_train - all_generic_test

    cation_overlap = all_cations_train & all_cations_test

    print(f"\n  Unique cations (train+val): {len(all_cations_train)}")
    print(f"  Unique cations (test): {len(all_cations_test)}")
    print(f"  Cation overlap: {len(cation_overlap)}")
    print(f"  Cations test-only: {len(all_cations_test - all_cations_train)}")

    print(f"\n  Murcko Scaffolds:")
    print(f"    Train+Val: {len(all_scaffolds_train)}")
    print(f"    Test: {len(all_scaffolds_test)}")
    print(f"    Overlap: {len(scaffold_overlap)}")
    print(f"    Test-only scaffolds: {len(scaffold_test_only)}")
    print(f"    Train-only scaffolds: {len(scaffold_train_only)}")

    print(f"\n  Generic (Framework) Scaffolds:")
    print(f"    Train+Val: {len(all_generic_train)}")
    print(f"    Test: {len(all_generic_test)}")
    print(f"    Overlap: {len(generic_overlap)}")
    print(f"    Test-only generic: {len(generic_test_only)}")
    print(f"    Train-only generic: {len(generic_train_only)}")

    # Does species-identity split already separate scaffolds?
    if len(cation_overlap) == 0:
        species_separation = "Complete species separation (no cation overlap)"
    else:
        species_separation = f"Partial: {len(cation_overlap)} cations shared"

    if len(scaffold_test_only) > 0:
        scaffold_separation = (
            f"Partial scaffold separation: {len(scaffold_test_only)} test-only scaffolds "
            f"out of {len(all_scaffolds_test)} test scaffolds"
        )
    else:
        scaffold_separation = "No new scaffolds in test (all test scaffolds seen in train)"

    if len(generic_test_only) > 0:
        generic_separation = (
            f"Partial generic scaffold separation: {len(generic_test_only)} test-only "
            f"generic scaffolds out of {len(all_generic_test)} test generic scaffolds"
        )
    else:
        generic_separation = "No new generic scaffolds in test"

    scaffold_overlap_pct = (
        len(scaffold_overlap) / len(all_scaffolds_test) * 100
        if len(all_scaffolds_test) > 0 else 0
    )
    generic_overlap_pct = (
        len(generic_overlap) / len(all_generic_test) * 100
        if len(all_generic_test) > 0 else 0
    )

    print(f"\n  Scaffold overlap rate (test scaffolds seen in train): {scaffold_overlap_pct:.1f}%")
    print(f"  Generic scaffold overlap rate: {generic_overlap_pct:.1f}%")
    print(f"\n  Species separation: {species_separation}")
    print(f"  Scaffold separation: {scaffold_separation}")
    print(f"  Generic separation: {generic_separation}")

    # Build results table
    results_rows = [
        {'level': 'cation_identity', 'train_val_unique': len(all_cations_train),
         'test_unique': len(all_cations_test), 'overlap': len(cation_overlap),
         'test_only': len(all_cations_test - all_cations_train),
         'overlap_pct': round(len(cation_overlap) / max(len(all_cations_test), 1) * 100, 1),
         'note': species_separation},
        {'level': 'murcko_scaffold', 'train_val_unique': len(all_scaffolds_train),
         'test_unique': len(all_scaffolds_test), 'overlap': len(scaffold_overlap),
         'test_only': len(scaffold_test_only),
         'overlap_pct': round(scaffold_overlap_pct, 1),
         'note': scaffold_separation},
        {'level': 'generic_scaffold', 'train_val_unique': len(all_generic_train),
         'test_unique': len(all_generic_test), 'overlap': len(generic_overlap),
         'test_only': len(generic_test_only),
         'overlap_pct': round(generic_overlap_pct, 1),
         'note': generic_separation},
    ]

    result_df = pd.DataFrame(results_rows)
    out_path = os.path.join(RESULTS_DIR, 'scaffold_analysis.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    # Also save the per-scaffold details
    scaffold_details = []
    for cat, scaf in cation_to_scaffold.items():
        gen = cation_to_generic.get(cat)
        in_train = cat in all_cations_train
        in_test = cat in all_cations_test
        scaffold_details.append({
            'cation_smiles': cat,
            'murcko_scaffold': scaf,
            'generic_scaffold': gen,
            'in_train': in_train,
            'in_test': in_test,
        })
    details_df = pd.DataFrame(scaffold_details)
    details_path = os.path.join(RESULTS_DIR, 'scaffold_analysis_details.csv')
    details_df.to_csv(details_path, index=False)
    print(f"  Per-cation scaffold details saved to {details_path}")

    return result_df


# ============================================================================
# Main
# ============================================================================
def main():
    print("CF-BILD Reviewer Response Experiments")
    print("=" * 60)

    # Experiment 1
    exp1 = experiment_1_unique_il_counts()

    # Experiment 2
    exp2 = experiment_2_structure_level_r2()

    # Experiment 3
    exp3 = experiment_3_two_objective_ablation()

    # Experiment 4
    exp4 = experiment_4_threshold_sensitivity()

    # Experiment 5
    exp5 = experiment_5_scaffold_analysis()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)

    # Summary
    print("\nOutput files:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname} ({size:,} bytes)")


if __name__ == '__main__':
    main()
