"""Generate visualization plots for CF-BILD results.
Loads saved model predictions from pkl files — no retraining needed.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')

prop_names = {'co2': 'CO₂ Capacity', 'vis': 'Viscosity (log)', 'tox': 'Toxicity (logEC50)'}
prop_units = {'co2': 'mol/mol', 'vis': 'log(Pa·s)', 'tox': 'logEC50'}
target_cols = {'co2': 'CO2-exp', 'vis': 'vis', 'tox': 'Experimental logEC50'}

# Load saved predictions
all_results = {}
for prop in ['co2', 'vis', 'tox']:
    pkl_path = os.path.join(OUTPUT_DIR, f'model_{prop}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        all_results[prop] = {
            'y_true': data['test_y_true'],
            'y_pred': data['test_y_pred'],
            'y_std': data['test_y_std'],
        }
        print(f"Loaded {prop} predictions from {pkl_path}")
    else:
        print(f"WARNING: {pkl_path} not found — run run_cfbild.py first")

if len(all_results) == 0:
    print("No saved predictions found. Exiting.")
    sys.exit(1)

# ========== PLOT 1: Predicted vs Experimental ==========
print("\nPlotting predicted vs experimental...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, prop in enumerate(['co2', 'vis', 'tox']):
    ax = axes[i]
    if prop not in all_results:
        ax.set_title(f'{prop_names[prop]}\nNo data')
        continue
    r = all_results[prop]
    y_true, y_pred = r['y_true'], r['y_pred']

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Experimental {prop_units[prop]}')
    ax.set_ylabel(f'Predicted {prop_units[prop]}')
    ax.set_title(f'{prop_names[prop]}\nR²={r2:.4f}  RMSE={rmse:.4f}')
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pred_vs_exp.png'), dpi=300, bbox_inches='tight')
print("  Saved pred_vs_exp.png")

# ========== PLOT 2: Calibration Plot ==========
print("Plotting calibration curves...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
expected_coverages = np.arange(0.05, 1.0, 0.05)

for i, prop in enumerate(['co2', 'vis', 'tox']):
    ax = axes[i]
    if prop not in all_results:
        ax.set_title(f'{prop_names[prop]}\nNo data')
        continue
    r = all_results[prop]
    y_true, y_pred, y_std = r['y_true'], r['y_pred'], r['y_std']

    observed = []
    for ec in expected_coverages:
        z = norm.ppf(0.5 + ec / 2)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        observed.append(coverage)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Ideal')
    ax.plot(expected_coverages, observed, 'o-', color='steelblue',
            markersize=4, label='Observed')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title(f'{prop_names[prop]}')
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
print("  Saved calibration_plot.png")

# ========== PLOT 3: Pareto Front ==========
print("Plotting Pareto front...")
pareto = pd.read_csv(os.path.join(OUTPUT_DIR, 'pareto_candidates.csv'))
top = pd.read_csv(os.path.join(OUTPUT_DIR, 'top_candidates.csv'))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
pairs = [
    ('co2_pred', 'vis_pred', 'CO₂ Capacity', 'Viscosity log(Pa·s)'),
    ('co2_pred', 'tox_pred', 'CO₂ Capacity', 'Toxicity logEC50'),
    ('vis_pred', 'tox_pred', 'Viscosity log(Pa·s)', 'Toxicity logEC50'),
]
for i, (xc, yc, xl, yl) in enumerate(pairs):
    ax = axes[i]
    ax.scatter(top[xc], top[yc], alpha=0.3, s=15, color='gray', label='Top-100')
    ax.scatter(pareto[xc], pareto[yc], s=30, color='red', zorder=5, label='Pareto')
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f'{xl} vs {yl}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pareto_front.png'), dpi=300, bbox_inches='tight')
print("  Saved pareto_front.png")

# ========== PLOT 4: Candidate Quality Distribution ==========
print("Plotting candidate distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for i, (prop, col) in enumerate([('co2', 'co2_pred'), ('vis', 'vis_pred'), ('tox', 'tox_pred')]):
    ax = axes[i]
    train_df = pd.read_csv(f'data/train_2_group_{prop}.csv')
    train_vals = train_df[target_cols[prop]].values

    ax.hist(train_vals, bins=50, alpha=0.5, density=True, color='steelblue', label='Training data')
    ax.hist(top[col].values, bins=20, alpha=0.5, density=True, color='red', label='Top-100 candidates')
    ax.set_xlabel(prop_units[prop])
    ax.set_ylabel('Density')
    ax.set_title(f'{prop_names[prop]}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'candidate_distribution.png'), dpi=300, bbox_inches='tight')
print("  Saved candidate_distribution.png")

print("\nAll 4 plots saved to output/")
