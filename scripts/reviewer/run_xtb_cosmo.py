"""
xTB-COSMO Validation for CF-BILD
Step 1: Oracle qualification (known ILs with experimental CO2 data)
Step 2: Candidate verification (top Pareto candidates)

Runs xtb via WSL Ubuntu.
"""
import os, sys, subprocess, json, pickle, tempfile, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bayesian-gp-cvloss'))

# torch before rdkit
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

OUTPUT_DIR = 'output'
XTB_CMD = "$HOME/miniconda3/bin/xtb"


def smiles_to_xyz(smiles, name="mol"):
    """Generate 3D coordinates from SMILES using RDKit ETKDG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        # Try with random coords
        params.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    lines = [f"{n_atoms}", name]
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{symbol}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
    return "\n".join(lines)


def run_xtb_solvation(smiles, name="mol", solvent="water", charge=0):
    """Run xtb GFN2 optimization + ALPB solvation via WSL.

    Returns dict with energy, solvation_energy, or None on failure.
    """
    xyz_content = smiles_to_xyz(smiles, name)
    if xyz_content is None:
        return None

    # Write xyz to temp file accessible from WSL
    wsl_tmpdir = f"/tmp/xtb_{name}_{os.getpid()}"

    try:
        # Create dir and write file via WSL
        subprocess.run(
            ['wsl', '-d', 'Ubuntu', '--', 'bash', '-c',
             f'mkdir -p {wsl_tmpdir}'],
            capture_output=True, timeout=10
        )

        # Write xyz file
        xyz_escaped = xyz_content.replace("'", "'\\''")
        subprocess.run(
            ['wsl', '-d', 'Ubuntu', '--', 'bash', '-c',
             f"echo '{xyz_escaped}' > {wsl_tmpdir}/mol.xyz"],
            capture_output=True, timeout=10
        )

        # Run xtb: optimize geometry + ALPB solvation
        cmd = (f"cd {wsl_tmpdir} && "
               f"{XTB_CMD} mol.xyz --gfn 2 --opt --alpb {solvent} "
               f"--chrg {charge} --uhf 0 2>&1")
        result = subprocess.run(
            ['wsl', '-d', 'Ubuntu', '--', 'bash', '-c', cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )
        output = result.stdout.decode('utf-8', errors='replace')

        # Parse total energy and solvation energy
        energy = None
        solv_energy = None
        for line in output.split('\n'):
            if 'TOTAL ENERGY' in line:
                try:
                    energy = float(line.split()[-3])
                except (IndexError, ValueError):
                    pass
            if 'Born' in line and 'free energy' in line.lower():
                try:
                    solv_energy = float(line.split()[-3])
                except (IndexError, ValueError):
                    pass

        # Also run gas phase for solvation energy difference
        cmd_gas = (f"cd {wsl_tmpdir} && "
                   f"{XTB_CMD} mol.xyz --gfn 2 --opt "
                   f"--chrg {charge} --uhf 0 2>&1")
        result_gas = subprocess.run(
            ['wsl', '-d', 'Ubuntu', '--', 'bash', '-c', cmd_gas],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )
        gas_energy = None
        for line in result_gas.stdout.decode('utf-8', errors='replace').split('\n'):
            if 'TOTAL ENERGY' in line:
                try:
                    gas_energy = float(line.split()[-3])
                except (IndexError, ValueError):
                    pass

        return {
            'energy_solv': energy,
            'energy_gas': gas_energy,
            'delta_solv': (energy - gas_energy) if energy and gas_energy else None,
        }

    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  Error for {name}: {e}")
        return None
    finally:
        # Cleanup
        subprocess.run(
            ['wsl', '-d', 'Ubuntu', '--', 'bash', '-c',
             f'rm -rf {wsl_tmpdir}'],
            capture_output=True, timeout=10
        )


def compute_co2_solvation_in_il(cation_smi, anion_smi, il_name="IL"):
    """Compute CO2 solvation energy in an IL environment.

    Simplified approach: compute solvation energy of CO2 using ALPB
    with water as proxy solvent, then compare IL ions' electronic
    properties as descriptors for CO2 affinity.

    More rigorous: compute interaction energy E(IL+CO2) - E(IL) - E(CO2).
    """
    # 1. Optimize cation
    cat_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(cation_smi))
    cat_result = run_xtb_solvation(cation_smi, f"{il_name}_cat", charge=cat_charge)

    # 2. Optimize anion
    an_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(anion_smi))
    an_result = run_xtb_solvation(anion_smi, f"{il_name}_an", charge=an_charge)

    if cat_result is None or an_result is None:
        return None

    # 3. CO2 in gas phase
    co2_gas = run_xtb_solvation("O=C=O", f"{il_name}_co2", charge=0)

    if co2_gas is None:
        return None

    # Use anion's solvation energy as proxy for CO2 affinity
    # (anion-CO2 interaction is the primary absorption mechanism)
    return {
        'cat_energy': cat_result['energy_solv'],
        'an_energy': an_result['energy_solv'],
        'co2_energy': co2_gas['energy_gas'],
        'an_delta_solv': an_result['delta_solv'],
        'cat_delta_solv': cat_result['delta_solv'],
        # Proxy for CO2 affinity: anion solvation + electrostatic descriptor
        'co2_affinity_proxy': an_result['delta_solv'] if an_result['delta_solv'] else None,
    }


def step1_oracle_qualification():
    """Validate xTB descriptors against experimental CO2 data."""
    print("=" * 60)
    print("STEP 1: xTB Oracle Qualification")
    print("=" * 60)

    # Load test set CO2 data
    test_df = pd.read_csv('data/test_group_co2.csv')

    # Get unique ILs — average CO2 per IL across all conditions
    unique_ils = test_df.groupby(['new_cation', 'new_anion']).agg(
        co2_mean=('CO2-exp', 'mean')
    ).reset_index()
    unique_ils.columns = ['new_cation', 'new_anion', 'CO2-exp']
    sample = unique_ils  # Use all unique ILs in test set
    print(f"  Unique ILs in test set: {len(sample)}")

    print(f"Computing xTB for {len(sample)} ILs...")

    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        cat = str(row['new_cation']).strip()
        an = str(row['new_anion']).strip()
        co2_exp = float(row['CO2-exp'])

        print(f"  [{i+1}/{len(sample)}] {cat[:30]}...{an[:30]}...", end=" ")

        xtb_result = compute_co2_solvation_in_il(cat, an, f"oracle_{i}")

        if xtb_result and xtb_result['co2_affinity_proxy'] is not None:
            results.append({
                'cation': cat, 'anion': an,
                'co2_exp': co2_exp,
                'an_delta_solv': xtb_result['an_delta_solv'],
                'cat_delta_solv': xtb_result['cat_delta_solv'],
            })
            print(f"OK (dG_an={xtb_result['an_delta_solv']:.4f})")
        else:
            print("FAILED")

    if len(results) < 5:
        print(f"Only {len(results)} successful — insufficient for qualification")
        return results

    df_res = pd.DataFrame(results)

    # Compute correlations
    from scipy.stats import spearmanr, pearsonr
    r_spearman, p_spearman = spearmanr(df_res['co2_exp'], df_res['an_delta_solv'])
    r_pearson, _ = pearsonr(df_res['co2_exp'], df_res['an_delta_solv'])
    r2 = r_pearson ** 2

    print(f"\nOracle Qualification Results ({len(results)} ILs):")
    print(f"  Pearson R2 = {r2:.4f}")
    print(f"  Spearman rho = {r_spearman:.4f} (p={p_spearman:.4e})")
    print(f"  Qualified: R2>0.6: {'YES' if r2 > 0.6 else 'NO'}, "
          f"rho>0.7: {'YES' if abs(r_spearman) > 0.7 else 'NO'}")

    df_res.to_csv(os.path.join(OUTPUT_DIR, 'xtb_oracle_qualification.csv'), index=False)
    return results


def step2_candidate_verification():
    """Verify top Pareto candidates with xTB."""
    print("\n" + "=" * 60)
    print("STEP 2: Candidate Verification")
    print("=" * 60)

    pareto = pd.read_csv(os.path.join(OUTPUT_DIR, 'pareto_candidates.csv'))
    top_n = min(20, len(pareto))  # Verify top 20 Pareto candidates
    print(f"Computing xTB for top {top_n} Pareto candidates...")

    results = []
    for i, (_, row) in enumerate(pareto.head(top_n).iterrows()):
        cat = str(row['cation']).strip()
        an = str(row['anion']).strip()

        print(f"  [{i+1}/{top_n}] {cat[:30]}...{an[:30]}...", end=" ")

        xtb_result = compute_co2_solvation_in_il(cat, an, f"cand_{i}")

        if xtb_result and xtb_result['co2_affinity_proxy'] is not None:
            results.append({
                'rank': int(row['rank']),
                'cation': cat, 'anion': an,
                'co2_pred': float(row['co2_pred']),
                'vis_pred': float(row['vis_pred']),
                'tox_pred': float(row['tox_pred']),
                'an_delta_solv': xtb_result['an_delta_solv'],
                'cat_delta_solv': xtb_result['cat_delta_solv'],
            })
            print(f"OK")
        else:
            print("FAILED")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'xtb_candidate_verification.csv'), index=False)

    if len(results) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(df_res['co2_pred'], df_res['an_delta_solv'])
        print(f"\nGP-xTB Agreement (Spearman rho): {rho:.4f} (p={p:.4e})")

    print(f"Results saved to output/xtb_candidate_verification.csv")
    return results


if __name__ == '__main__':
    print("xTB-COSMO Validation for CF-BILD")
    print("Using GFN2-xTB with ALPB solvation via WSL\n")

    results_oracle = step1_oracle_qualification()
    results_cand = step2_candidate_verification()

    print("\nxTB-COSMO validation complete!")
