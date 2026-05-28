# CF-BILD: Conservative Fragment-Based Bayesian Ionic Liquid Design

Trustworthy multi-objective ionic liquid design via compositional Gaussian processes with conservative Bayesian optimization.

This repository contains only the **scientific code** (training, ablation, screening, and reviewer-experiment scripts) used to produce the results reported in the manuscript. Manuscript-editing scripts, experimental data files, raw datasets, and trained model checkpoints are not included; please contact the authors for access to those artifacts.

## Overview

CF-BILD replaces unconstrained generative-model exploration with a closed-loop, uncertainty-aware optimization over a chemically grounded combinatorial IL design space. The framework has four layers:

1. **Fragment Vocabulary** — Decompose ILs into cation/anion fragments and encode each ion with a collision-free Morgan fingerprint (CF-MF).
2. **Compositional GP-BT Surrogate** — Multi-property prediction with a compositional kernel (`k = k_cat·k_an + k_cross`); GP-BT (Bayesian-tuned cross-validated GP) for calibrated uncertainty.
3. **Conservative Multi-Objective Acquisition (C-EHVI)** — Decomposable expected-improvement acquisition × joint feasibility penalty; uncertainty-penalized ranking over the full candidate pool.
4. **Combinatorial Stability Filter** — Three-tier SMARTS-based post-filter (proton transfer / liquidity heuristics / chemical reactivity) implemented in RDKit.

## Repository Layout

```
CF-BILD/
├── cf_bild/                          # Core framework (importable Python package)
│   ├── __init__.py
│   ├── fragment_vocab.py             # CF-MF fingerprints, data loading, scaler
│   └── conservative_bo.py            # C-EHVI acquisition, Pareto front, hypervolume
│
├── scripts/
│   ├── pipeline/                     # Main scientific pipeline
│   │   ├── run_cfbild.py             # Phases 1–3: vocab + GP-BT training + C-EHVI screening
│   │   ├── run_ablation.py           # Ablations A (kernel), B (surrogate), C (acquisition)
│   │   ├── run_ablation_d.py         # Ablation D: fragment-constrained vs generative
│   │   ├── plot_results.py           # GP performance + calibration figures
│   │   └── plot_ablation.py          # Ablation figures + composite tier scoring
│   └── reviewer/                     # Reviewer-response experiments
│       ├── run_reviewer_experiments.py    # Scaffold / two-objective / threshold sensitivity / structure-level R²
│       ├── run_exact_ehvi.py              # Exact Monte-Carlo EHVI (BoTorch) benchmark
│       ├── run_stability_screening.py     # SMARTS three-tier stability filter
│       ├── run_drop_cross_ablation.py     # Drop-k_cross ablation (product / additive, no cross term)
│       ├── cache_predictions.py           # Cache GP predictions on full 87,365-candidate pool
│       ├── recompute_acq_comparison.py    # Consistent re-comparison of C-EHVI / MC EHVI / Add EI / Random
│       ├── run_multiseed_mc_ehvi.py       # 5-seed MC EHVI (HV mean ± SE; statistical-significance test)
│       └── run_xtb_cosmo.py               # GFN2-xTB + COSMO-RS oracle (independent verification)
│
└── VAE/                              # Generative-model baseline (Ablation D only)
    ├── train_model.py                # Train SMILES language model
    ├── sample-molecules.py           # Generate candidate ILs
    ├── models.py                     # GRU / RNN architectures
    ├── datasets.py                   # SMILES dataset loader
    ├── functions.py                  # Sampling utilities
    ├── augment-SMILES.py             # SMILES augmentation
    ├── clean-SMILES.py               # SMILES cleaning
    ├── convert-DeepSMILES.py         # DeepSMILES conversion
    ├── calculate_outcomes.py         # Post-sampling diagnostics
    └── Training generative models.ipynb
```

## Environment Setup

Tested on Python 3.10, NVIDIA RTX 5090 + CUDA 12.8.

```bash
python -m venv venv
source venv/Scripts/activate          # Windows (Git Bash); use venv/bin/activate on Linux/Mac

# Core scientific stack
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install gpytorch botorch hyperopt rdkit pandas scikit-learn matplotlib numpy

# Two locally-developed packages (install from their respective sources):
pip install <path-to>/bit_collision_free_MF        # CF-MF fingerprint package
pip install <path-to>/bayesian-gp-cvloss           # GP-BT (GPyTorch backend)
```

Both `bit_collision_free_MF` and the GPyTorch port of `bayesian-gp-cvloss` are required as dependencies. Contact the authors for the source distributions.

**Important**: Always import `torch` *before* `rdkit` to avoid DLL conflicts on Windows.

## Reproducing the Reported Results

The scripts below assume:
- A `data/` directory with the property-specific cross-validation splits (`train_{1-5}_group_{co2,vis,tox}.csv`, `val_{1-5}_group_{co2,vis,tox}.csv`, `test_group_{co2,vis,tox}.csv`).
- An `output/` directory for generated artifacts.

### Full pipeline (≈ 60 minutes on RTX 5090)
```bash
python scripts/pipeline/run_cfbild.py
```
Runs Phase 1 (fragment vocabulary), Phase 2 (per-property GP-BT training with TPE), and Phase 3 (C-EHVI scoring across all 87,365 candidates). Produces `output/fragment_vocab.pkl`, `output/model_{co2,vis,tox}.pkl`, `output/top_candidates.csv`, `output/pareto_candidates.csv`, and `output/model_stats.json`.

### Ablation studies
```bash
python scripts/pipeline/run_ablation.py      # A (kernel) / B (surrogate) / C (acquisition)
python scripts/pipeline/run_ablation_d.py    # D: vs SMILES generative baseline
```

### Plotting + composite tier scoring (consumes saved models)
```bash
python scripts/pipeline/plot_results.py      # GP performance + calibration figures
python scripts/pipeline/plot_ablation.py     # Ablation figures + final composite ranking
```

### Reviewer experiments (independent of the main pipeline)
```bash
python scripts/reviewer/run_exact_ehvi.py            # Exact MC-EHVI vs C-EHVI
python scripts/reviewer/run_reviewer_experiments.py  # Scaffold / two-obj / threshold-sensitivity / structure-level R²
python scripts/reviewer/run_stability_screening.py   # SMARTS three-tier filter on top-100
python scripts/reviewer/run_drop_cross_ablation.py   # Drop-k_cross ablation (product/additive, no cross term)
python scripts/reviewer/cache_predictions.py         # Cache GP predictions on 87,365 candidates (skips TPE)
python scripts/reviewer/recompute_acq_comparison.py  # Consistent C-EHVI/MC EHVI/Add EI/Random comparison
python scripts/reviewer/run_multiseed_mc_ehvi.py     # 5-seed MC EHVI (HV ± SE for significance test)
python scripts/reviewer/run_xtb_cosmo.py             # xTB + COSMO-RS oracle (slow)
```

The four newer scripts (`run_drop_cross_ablation.py`, `cache_predictions.py`, `recompute_acq_comparison.py`, `run_multiseed_mc_ehvi.py`) implement the additional analyses requested during peer review: the drop-k_cross architectural ablation, a fast prediction cache that skips TPE re-optimization, a consistent calibrated-σ re-comparison of all four acquisition methods, and a multi-seed MC EHVI sweep that quantifies Monte-Carlo standard error on the headline 17% C-EHVI vs MC EHVI hypervolume gap.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fingerprint | CF-MF (radius = 2, zero-columns removed) | Collision-free; 1,369 (cation) + 902 (anion) dimensions after zero removal |
| GP backend | GPyTorch (not GPflow) | Windows GPU support, autodiff, batch inference |
| Lengthscale | Per-component shared (one ls per kernel block) | 5–7-dim search space vs > 6,800 with full ARD |
| Kernel form | Compositional (`k_cat·k_an + k_cross`) | Encodes cation-anion product structure explicitly |
| Preprocessing | StandardScaler on all features per-fold | Required for GP with mixed-magnitude features |
| CV splits | 5-fold by IL species (no leakage) | Same species cannot appear in both train and test |
| Variance calibration | Post-hoc multiplicative (CV-residual based) | Corrects raw GP σ over/under-confidence |
| Max TPE evals | 3,000 with early-stop patience 50 | Empirically sufficient for 5–7-dim Bayesian-opt search |
| Stability filter | RDKit SMARTS (3 tiers) | Inspectable, no learned model required |

## Results Summary (test set, species-level 5-fold split)

| Property | R² | RMSE | 95% CI Coverage | NLPD |
|----------|-----|------|-----------------|------|
| CO₂ capacity | 0.889 | 0.072 | 97.5% | −1.22 |
| Viscosity | 0.853 | 0.617 | 95.7% | 0.76 |
| Toxicity (logEC₅₀) | 0.537 | 0.612 | 100% | 1.41 |

Bootstrap 95% CIs (paired 10,000-resample): CO₂ structure-level R² = 0.754 [0.38, 0.91] (N = 21 unique IL species); toxicity R² = 0.537 [0.23, 0.80] (N = 32 test records).

Ablation summaries:
- **Kernel form**: compositional (product / additive) R² ≈ 0.85–0.89 vs single-RBF R² < 0.26 for CO₂; the drop-k_cross variant (`k_cat · k_an` alone, no cross term) retains test-set R² of 0.869 / 0.854 / 0.549 (ΔR² ≤ 0.020 vs full compositional kernel across all three properties), establishing `k_cat · k_an` as the empirically-validated minimum-parameter architecture.
- **Surrogate**: GP-BT R² = 0.89 vs default-hyperparameter standard GP R² = 0.39 (CO₂).
- **Acquisition (consistent calibrated-σ recomputation)**: C-EHVI HV = 40.79 vs exact MC-EHVI HV = 49.16 ± 0.25 (5 seeds) vs additive-EI HV = 31.17 vs random HV = 27.34 ± 1.22 (5 seeds); 17% C-EHVI vs MC-EHVI gap is > 30× the MC standard error → highly statistically significant. C-EHVI mean predictive σ = 1.28 vs MC-EHVI 1.32.
- **Search space**: 100% of CF-BILD candidates have both ions in training-data-related vocabulary; 0% for an unconstrained GRU SMILES generator. SMARTS stability filter rejects 68.6% of the GRU subsample as Fail vs 54.0% of the C-EHVI-ranked CF-BILD top-100.

Stability filter on top-100 C-EHVI candidates: 37 Pass / 9 Caution / 54 Fail; the pre-filter top-1 (trimethylammonium glycinate, `C[NH+](C)C` + `NCC(=O)[O-]`) is rejected by Tier I (proton transfer), and the post-filter top-1 becomes 1-ethyl-3-methylimidazolium prolinate (`CCn1cc[n+](C)c1` + `O=C([O-])C1CCCN1`, original C-EHVI rank 3) — a novel cation-anion combination not in the training set but belonging to the experimentally well-studied family of imidazolium-amino-acid ILs for CO₂ capture.

## Citation

If you use this code, please cite the corresponding manuscript:
```
@article{CFBILD2026,
  author = {[Author 1] and [Author 2] and [Author 3]},
  title  = {Fragment-Constrained Ionic Liquid Screening with a Compositional
            Gaussian Process Surrogate and an Uncertainty-Penalized
            Multi-Objective Acquisition Function},
  journal = {[journal]},
  year   = {2026},
}
```

## License

[MIT — to be added]
