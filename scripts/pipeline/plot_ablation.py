"""Ablation result visualization + Final scoring + Feature importance."""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Load ablation results
with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'r') as f:
    abl = json.load(f)

# ========== ABLATION A: Kernel Form Bar Chart ==========
print("Plotting Ablation A...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
kernels = ['product', 'additive', 'standard']
kernel_labels = ['Product\n(k_cat·k_an + k_cross)', 'Additive\n(k_cat+k_an+k_cross)', 'Standard\n(single RBF)']
colors = ['#2196F3', '#4CAF50', '#FF9800']
metrics_list = ['R2', 'RMSE', 'CI_coverage']
metric_labels = ['R²', 'RMSE', '95% CI Coverage']

for i, (metric, mlabel) in enumerate(zip(metrics_list, metric_labels)):
    ax = axes[i]
    x = np.arange(3)
    width = 0.25
    for j, prop in enumerate(['co2', 'vis', 'tox']):
        vals = [abl['A_kernel_form'][k][prop].get(metric, 0) for k in kernels]
        ax.bar(x + j * width, vals, width, label=prop.upper(), color=colors[j], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(kernel_labels, fontsize=8)
    ax.set_ylabel(mlabel)
    ax.set_title(f'Ablation A: {mlabel}')
    ax.legend(fontsize=8)
    if metric == 'R2':
        ax.set_ylim([0, 1])
    if metric == 'CI_coverage':
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% target')
        ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_a_kernel.png'), dpi=300, bbox_inches='tight')
print("  Saved ablation_a_kernel.png")

# ========== ABLATION B: GP-BT vs Standard GP ==========
print("Plotting Ablation B...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
methods = ['GP-BT', 'Standard-GP']
props = ['co2', 'vis', 'tox']
prop_labels = ['CO₂', 'Viscosity', 'Toxicity']

# R²
ax = axes[0]
x = np.arange(3)
width = 0.35
for j, method in enumerate(methods):
    vals = [abl['B_surrogate'][method][p].get('R2', 0) for p in props]
    ax.bar(x + j * width, vals, width, label=method,
           color=['#2196F3', '#FF5722'][j], alpha=0.8)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(prop_labels)
ax.set_ylabel('R²')
ax.set_title('Ablation B: R² Comparison')
ax.legend()
ax.set_ylim([0, 1])

# CI Coverage
ax = axes[1]
for j, method in enumerate(methods):
    vals = [abl['B_surrogate'][method][p].get('CI_coverage', 0) for p in props]
    ax.bar(x + j * width, vals, width, label=method,
           color=['#2196F3', '#FF5722'][j], alpha=0.8)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(prop_labels)
ax.set_ylabel('95% CI Coverage')
ax.set_title('Ablation B: Calibration Comparison')
ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
ax.legend()
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_b_surrogate.png'), dpi=300, bbox_inches='tight')
print("  Saved ablation_b_surrogate.png")

# ========== ABLATION C: Acquisition Function ==========
print("Plotting Ablation C...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
acq_methods = ['C-EHVI', 'EHVI', 'Random']
acq_colors = ['#2196F3', '#FF9800', '#9E9E9E']

# Hypervolume
ax = axes[0]
hvs = [abl['C_acquisition'][m]['hypervolume'] for m in acq_methods]
ax.bar(acq_methods, hvs, color=acq_colors, alpha=0.8)
ax.set_ylabel('Hypervolume')
ax.set_title('Hypervolume')
for i, v in enumerate(hvs):
    ax.text(i, v + 0.3, f'{v:.2f}', ha='center', fontsize=9)

# Pareto count
ax = axes[1]
paretos = [abl['C_acquisition'][m]['n_pareto'] for m in acq_methods]
ax.bar(acq_methods, paretos, color=acq_colors, alpha=0.8)
ax.set_ylabel('Pareto-optimal Count')
ax.set_title('Pareto Front Size')
for i, v in enumerate(paretos):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=9)

# Mean sigma (uncertainty)
ax = axes[2]
sigmas = [abl['C_acquisition'][m]['mean_sigma'] for m in acq_methods]
ax.bar(acq_methods, sigmas, color=acq_colors, alpha=0.8)
ax.set_ylabel('Mean Predictive σ')
ax.set_title('Candidate Uncertainty')
for i, v in enumerate(sigmas):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_c_acquisition.png'), dpi=300, bbox_inches='tight')
print("  Saved ablation_c_acquisition.png")

# ========== FINAL COMPOSITE SCORING ==========
print("\nComputing final composite scores...")
top = pd.read_csv(os.path.join(OUTPUT_DIR, 'top_candidates_tiered.csv'))

# Score = C-EHVI * w_tier
# w_tier: Tier 1=1.0, Tier 2=0.7, Tier 3=0.3
tier_weights = {1: 1.0, 2: 0.7, 3: 0.3}
top['w_tier'] = top['tier'].map(tier_weights)
top['composite_score'] = top['c_ehvi_score'] * top['w_tier']
top = top.sort_values('composite_score', ascending=False).reset_index(drop=True)
top['final_rank'] = range(1, len(top) + 1)

top.to_csv(os.path.join(OUTPUT_DIR, 'final_ranked_candidates.csv'), index=False)
print(f"  Saved final_ranked_candidates.csv")

print("\n  Top 10 Final Candidates:")
print(f"  {'Rank':<5} {'Tier':<5} {'Score':<8} {'CO2':<8} {'Vis':<8} {'Tox':<8} {'IL SMILES'}")
print("  " + "-" * 80)
for _, r in top.head(10).iterrows():
    print(f"  {int(r.final_rank):<5} {int(r.tier):<5} {r.composite_score:<8.4f} "
          f"{r.co2_pred:<8.4f} {r.vis_pred:<8.4f} {r.tox_pred:<8.4f} {r.il_smiles[:50]}")

# ========== FEATURE IMPORTANCE via Lengthscale Analysis ==========
print("\nAnalyzing feature importance from kernel lengthscales...")
fig, ax = plt.subplots(figsize=(10, 5))

# Load model params for each property
importance_data = []
for prop in ['co2', 'vis', 'tox']:
    pkl_path = os.path.join(OUTPUT_DIR, f'model_{prop}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        params = data['best_params']
        # Smaller lengthscale = more important (feature varies more in relevant ways)
        # For compositional kernel: ls_cat, ls_an, ls_cross
        ls_cat = params.get('ls_cat', None)
        ls_an = params.get('ls_an', None)
        ls_cross = params.get('ls_cross', None)
        ls_env_0 = params.get('ls_env_0', None)
        ls_env_1 = params.get('ls_env_1', None)

        importance_data.append({
            'property': prop.upper(),
            'kernel': params.get('kernel_name', '?'),
            'ls_cat': ls_cat,
            'ls_an': ls_an,
            'ls_cross': ls_cross,
            'ls_T': ls_env_0,
            'ls_P': ls_env_1,
            'noise': params.get('likelihood_noise_variance', None),
            'kernel_var': params.get('kernel_variance', None),
        })

if importance_data:
    df_imp = pd.DataFrame(importance_data)

    # Plot lengthscales (inverse = importance)
    components = ['ls_cat', 'ls_an', 'ls_cross']
    comp_labels = ['Cation FP', 'Anion FP', 'Cross (cat||an)']
    x = np.arange(len(importance_data))
    width = 0.25
    colors_comp = ['#2196F3', '#4CAF50', '#FF9800']

    for j, (comp, clabel) in enumerate(zip(components, comp_labels)):
        vals = [1.0 / (row[comp] + 1e-6) if row[comp] is not None else 0
                for _, row in df_imp.iterrows()]
        ax.bar(x + j * width, vals, width, label=clabel, color=colors_comp[j], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([d['property'] for d in importance_data])
    ax.set_ylabel('Importance (1/lengthscale)')
    ax.set_title('Component Importance from Kernel Lengthscales')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    print("  Saved feature_importance.png")

    # Print best hyperparameters
    print("\n  Best Hyperparameters per Property:")
    print(f"  {'Prop':<6} {'Kernel':<20} {'ls_cat':<10} {'ls_an':<10} {'ls_cross':<10} "
          f"{'ls_T':<10} {'ls_P':<10} {'noise':<12} {'kvar':<12}")
    for _, row in df_imp.iterrows():
        print(f"  {row['property']:<6} {row['kernel']:<20} "
              f"{row['ls_cat']:<10.4f} {row['ls_an']:<10.4f} {row['ls_cross']:<10.4f} "
              f"{str(row.get('ls_T', 'N/A')):<10} {str(row.get('ls_P', 'N/A')):<10} "
              f"{row['noise']:<12.6f} {row['kernel_var']:<12.6f}")

print("\nAll done!")
