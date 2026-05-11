"""
Ablation D: Fragment-constrained (CF-BILD) vs VAE free-generation.

Compares:
1. Chemical validity rate
2. Structural complexity
3. Uniqueness
4. Overlap with known ILs (synthesizability proxy)
5. GP predictability (fraction with finite predictions)
6. Pareto front quality (if GP can predict)
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# torch before rdkit
from bayesian_gp_cvloss import GPCrossValidatedOptimizer
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def analyze_validity(df, name):
    """Analyze chemical validity of generated ILs."""
    n_total = len(df)
    valid_cat, valid_an, valid_both = 0, 0, 0
    unique_smiles = set()
    mol_weights_cat, mol_weights_an = [], []

    for _, row in df.iterrows():
        cat_smi = str(row['cation']).strip()
        an_smi = str(row['anion']).strip()

        cat_mol = Chem.MolFromSmiles(cat_smi)
        an_mol = Chem.MolFromSmiles(an_smi)

        cat_ok = cat_mol is not None
        an_ok = an_mol is not None

        if cat_ok:
            valid_cat += 1
            mol_weights_cat.append(Descriptors.MolWt(cat_mol))
        if an_ok:
            valid_an += 1
            mol_weights_an.append(Descriptors.MolWt(an_mol))
        if cat_ok and an_ok:
            valid_both += 1
            # Canonical SMILES for uniqueness
            canon = Chem.MolToSmiles(cat_mol) + "." + Chem.MolToSmiles(an_mol)
            unique_smiles.add(canon)

    stats = {
        'name': name,
        'total': n_total,
        'valid_cation': valid_cat,
        'valid_anion': valid_an,
        'valid_both': valid_both,
        'validity_rate': valid_both / n_total if n_total > 0 else 0,
        'unique_valid': len(unique_smiles),
        'uniqueness_rate': len(unique_smiles) / valid_both if valid_both > 0 else 0,
        'mean_mw_cat': np.mean(mol_weights_cat) if mol_weights_cat else 0,
        'mean_mw_an': np.mean(mol_weights_an) if mol_weights_an else 0,
        'median_mw_cat': np.median(mol_weights_cat) if mol_weights_cat else 0,
        'median_mw_an': np.median(mol_weights_an) if mol_weights_an else 0,
        'mw_cat_list': mol_weights_cat,
        'mw_an_list': mol_weights_an,
    }
    return stats


def check_known_overlap(df, known_ions):
    """Check how many generated ions are in known fragment vocabulary."""
    cat_known, an_known, both_known = 0, 0, 0
    for _, row in df.iterrows():
        cat_mol = Chem.MolFromSmiles(str(row['cation']).strip())
        an_mol = Chem.MolFromSmiles(str(row['anion']).strip())
        if cat_mol is None or an_mol is None:
            continue
        cat_canon = Chem.MolToSmiles(cat_mol)
        an_canon = Chem.MolToSmiles(an_mol)
        ck = cat_canon in known_ions
        ak = an_canon in known_ions
        if ck:
            cat_known += 1
        if ak:
            an_known += 1
        if ck and ak:
            both_known += 1
    return cat_known, an_known, both_known


def main():
    print("=" * 60)
    print("ABLATION D: CF-BILD (Fragment) vs VAE (Free Generation)")
    print("=" * 60)

    # Load VAE data
    vae_df = pd.read_csv('D:/wuzeting/VAE/generate_result.csv')
    print(f"\nVAE generated: {len(vae_df)} ILs")

    # Load CF-BILD candidates
    cfbild_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'top_candidates.csv'))
    print(f"CF-BILD top candidates: {len(cfbild_df)} ILs")

    # Build CF-BILD full space info
    vocab_path = os.path.join(OUTPUT_DIR, 'fragment_vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    known_cations = set(vocab_data['cations'].keys())
    known_anions = set(vocab_data['anions'].keys())
    known_ions = known_cations | known_anions
    n_cfbild_space = len(known_cations) * len(known_anions)

    print(f"\nCF-BILD fragment space: {len(known_cations)} cations × {len(known_anions)} anions = {n_cfbild_space:,}")

    # ===== 1. Validity Analysis =====
    print("\n--- 1. Chemical Validity ---")

    # VAE validity (sample 10000 for speed if too large)
    if len(vae_df) > 20000:
        vae_sample = vae_df.sample(20000, random_state=42)
        print(f"  Sampling 20000 from {len(vae_df)} VAE candidates for analysis")
    else:
        vae_sample = vae_df

    vae_stats = analyze_validity(vae_sample, "VAE")

    # CF-BILD: construct full space candidates (sample)
    # All CF-BILD candidates are valid by construction
    cfbild_stats = {
        'name': 'CF-BILD',
        'total': n_cfbild_space,
        'valid_both': n_cfbild_space,
        'validity_rate': 1.0,
        'unique_valid': n_cfbild_space,
        'uniqueness_rate': 1.0,
    }

    print(f"\n  {'Metric':<25} {'VAE':<15} {'CF-BILD':<15}")
    print("  " + "-" * 55)
    print(f"  {'Total candidates':<25} {vae_stats['total']:<15} {cfbild_stats['total']:<15,}")
    print(f"  {'Valid (both ions)':<25} {vae_stats['valid_both']:<15} {cfbild_stats['valid_both']:<15,}")
    print(f"  {'Validity rate':<25} {vae_stats['validity_rate']:<15.4f} {cfbild_stats['validity_rate']:<15.4f}")
    print(f"  {'Unique valid':<25} {vae_stats['unique_valid']:<15} {cfbild_stats['unique_valid']:<15,}")
    print(f"  {'Uniqueness rate':<25} {vae_stats['uniqueness_rate']:<15.4f} {cfbild_stats['uniqueness_rate']:<15.4f}")

    # ===== 2. Known Ion Overlap (Synthesizability) =====
    print("\n--- 2. Known Ion Overlap (Synthesizability Proxy) ---")
    vae_cat_known, vae_an_known, vae_both_known = check_known_overlap(vae_sample, known_ions)
    vae_valid = vae_stats['valid_both']

    print(f"  VAE: cation known={vae_cat_known}/{vae_valid} ({100*vae_cat_known/max(vae_valid,1):.1f}%), "
          f"anion known={vae_an_known}/{vae_valid} ({100*vae_an_known/max(vae_valid,1):.1f}%), "
          f"both known={vae_both_known}/{vae_valid} ({100*vae_both_known/max(vae_valid,1):.1f}%)")
    print(f"  CF-BILD: both known=100% by construction")

    # ===== 3. Molecular Complexity =====
    print("\n--- 3. Molecular Complexity ---")
    print(f"  VAE cation MW: mean={vae_stats['mean_mw_cat']:.1f}, median={vae_stats['median_mw_cat']:.1f}")
    print(f"  VAE anion MW: mean={vae_stats['mean_mw_an']:.1f}, median={vae_stats['median_mw_an']:.1f}")

    # CF-BILD ion MW
    cfbild_cat_mw = []
    for smi in known_cations:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            cfbild_cat_mw.append(Descriptors.MolWt(mol))
    cfbild_an_mw = []
    for smi in known_anions:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            cfbild_an_mw.append(Descriptors.MolWt(mol))

    print(f"  CF-BILD cation MW: mean={np.mean(cfbild_cat_mw):.1f}, median={np.median(cfbild_cat_mw):.1f}")
    print(f"  CF-BILD anion MW: mean={np.mean(cfbild_an_mw):.1f}, median={np.median(cfbild_an_mw):.1f}")

    # ===== PLOT =====
    print("\n--- Generating comparison plots ---")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Plot 1: Validity comparison
    ax = axes[0, 0]
    methods = ['VAE', 'CF-BILD']
    validity = [vae_stats['validity_rate'] * 100, 100]
    bars = ax.bar(methods, validity, color=['#FF5722', '#2196F3'], alpha=0.8)
    ax.set_ylabel('Validity Rate (%)')
    ax.set_title('Chemical Validity')
    ax.set_ylim([0, 110])
    for bar, v in zip(bars, validity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}%', ha='center', fontsize=10)

    # Plot 2: Uniqueness
    ax = axes[0, 1]
    uniqueness = [vae_stats['uniqueness_rate'] * 100, 100]
    bars = ax.bar(methods, uniqueness, color=['#FF5722', '#2196F3'], alpha=0.8)
    ax.set_ylabel('Uniqueness Rate (%)')
    ax.set_title('Unique Candidates')
    ax.set_ylim([0, 110])
    for bar, v in zip(bars, uniqueness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}%', ha='center', fontsize=10)

    # Plot 3: Known ion overlap
    ax = axes[0, 2]
    overlap_pct = [100 * vae_both_known / max(vae_valid, 1), 100]
    bars = ax.bar(methods, overlap_pct, color=['#FF5722', '#2196F3'], alpha=0.8)
    ax.set_ylabel('Both Ions Known (%)')
    ax.set_title('Synthesizability (Known Ions)')
    ax.set_ylim([0, 110])
    for bar, v in zip(bars, overlap_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}%', ha='center', fontsize=10)

    # Plot 4: Cation MW distribution
    ax = axes[1, 0]
    ax.hist(vae_stats['mw_cat_list'], bins=50, alpha=0.5, density=True,
            color='#FF5722', label='VAE')
    ax.hist(cfbild_cat_mw, bins=50, alpha=0.5, density=True,
            color='#2196F3', label='CF-BILD')
    ax.set_xlabel('Molecular Weight')
    ax.set_ylabel('Density')
    ax.set_title('Cation MW Distribution')
    ax.legend()
    ax.set_xlim([0, 800])

    # Plot 5: Anion MW distribution
    ax = axes[1, 1]
    ax.hist(vae_stats['mw_an_list'], bins=50, alpha=0.5, density=True,
            color='#FF5722', label='VAE')
    ax.hist(cfbild_an_mw, bins=50, alpha=0.5, density=True,
            color='#2196F3', label='CF-BILD')
    ax.set_xlabel('Molecular Weight')
    ax.set_ylabel('Density')
    ax.set_title('Anion MW Distribution')
    ax.legend()
    ax.set_xlim([0, 1000])

    # Plot 6: Summary table as text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"Summary Comparison\n"
        f"{'─' * 35}\n"
        f"{'Metric':<22} {'VAE':<10} {'CF-BILD':<10}\n"
        f"{'─' * 35}\n"
        f"{'Total':<22} {vae_stats['total']:<10} {n_cfbild_space:<10,}\n"
        f"{'Validity':<22} {vae_stats['validity_rate']*100:<9.1f}% {'100.0':<9}%\n"
        f"{'Uniqueness':<22} {vae_stats['uniqueness_rate']*100:<9.1f}% {'100.0':<9}%\n"
        f"{'Both ions known':<22} {100*vae_both_known/max(vae_valid,1):<9.1f}% {'100.0':<9}%\n"
        f"{'Cat MW (median)':<22} {vae_stats['median_mw_cat']:<9.1f} {np.median(cfbild_cat_mw):<9.1f}\n"
        f"{'An MW (median)':<22} {vae_stats['median_mw_an']:<9.1f} {np.median(cfbild_an_mw):<9.1f}\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_d_vae_comparison.png'),
                dpi=300, bbox_inches='tight')
    print("  Saved ablation_d_vae_comparison.png")

    # Save results
    results = {
        'VAE': {
            'total': vae_stats['total'],
            'sampled': len(vae_sample),
            'valid_both': vae_stats['valid_both'],
            'validity_rate': vae_stats['validity_rate'],
            'unique_valid': vae_stats['unique_valid'],
            'uniqueness_rate': vae_stats['uniqueness_rate'],
            'both_ions_known': vae_both_known,
            'both_ions_known_rate': vae_both_known / max(vae_valid, 1),
            'median_mw_cat': vae_stats['median_mw_cat'],
            'median_mw_an': vae_stats['median_mw_an'],
        },
        'CF-BILD': {
            'total': n_cfbild_space,
            'valid_both': n_cfbild_space,
            'validity_rate': 1.0,
            'unique_valid': n_cfbild_space,
            'uniqueness_rate': 1.0,
            'both_ions_known': n_cfbild_space,
            'both_ions_known_rate': 1.0,
            'median_mw_cat': float(np.median(cfbild_cat_mw)),
            'median_mw_an': float(np.median(cfbild_an_mw)),
        }
    }

    import json
    with open(os.path.join(OUTPUT_DIR, 'ablation_d_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("  Saved ablation_d_results.json")

    print("\nAblation D complete!")


if __name__ == '__main__':
    main()
