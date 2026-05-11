#!/usr/bin/env python
"""
Combinatorial Stability Screening for CF-BILD Top 100 Candidates.

Three-tier filter:
  Tier I   - Proton transfer stability
  Tier II  - Melting point screening (empirical score 0-6)
  Tier III - Chemical reactivity flags

Classifies each candidate as Pass / Caution / Fail.
"""

import sys
sys.path.insert(0, 'D:/wuzeting')
sys.path.insert(0, 'D:/wuzeting/bayesian-gp-cvloss')

# Import torch BEFORE rdkit to avoid DLL conflicts on Windows
import torch  # noqa: F401

import os
import re
import csv
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def mol_from_smiles(smi):
    """Parse SMILES, return RDKit mol or None."""
    if not smi or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)


def has_substructure(mol, smarts):
    """Check whether *mol* contains a substructure given as SMARTS."""
    if mol is None:
        return False
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return False
    return mol.HasSubstructMatch(pat)


# ---------------------------------------------------------------------------
# Tier I  -  Proton Transfer Stability
# ---------------------------------------------------------------------------

def is_protic_cation(smi, mol):
    """
    A cation is *protic* if it bears a transferable proton on N or O.
    Key patterns:
      - [NH+]   trialkylammonium  (e.g. trimethylammonium C[NH+](C)C)
      - [NH2+]  dialkylammonium
      - [NH3+]  alkylammonium
      - [nH+]   protonated aromatic N  (e.g. imidazolium ring with [nH+])
    We look for any N-H or O-H on a positively charged atom.
    """
    if mol is None:
        return False
    # SMARTS for N carrying at least one H and a positive charge
    protic_patterns = [
        '[NH+]',     # secondary / tertiary ammonium with 1H
        '[NH2+]',    # primary ammonium with 2H
        '[NH3+]',    # ammonium with 3H
        '[nH+]',     # protonated aromatic N with H (imidazolium-like)
        '[nH+;r5]',  # protonated aromatic N in 5-ring
        '[nH+;r6]',  # protonated aromatic N in 6-ring
    ]
    for pat in protic_patterns:
        if has_substructure(mol, pat):
            return True
    return False


def is_strongly_basic_anion(smi, mol):
    """
    Strongly basic anions that accept a proton readily:
      - Carboxylate  [O-] attached to C=O  (R-COO-)
      - Amino acid anion  (has both COO- and NH2)
      - Simple alkoxide [O-] not in sulfonyl/phosphoryl context
    We intentionally exclude very weak bases like [NTf2]-, saccharinate, etc.
    """
    if mol is None:
        return False
    # Carboxylate: C(=O)[O-]
    if has_substructure(mol, '[CX3](=O)[O-]'):
        return True
    # Simple alkoxide
    if has_substructure(mol, '[#6][O-]') and not has_substructure(mol, '[S,P](=O)[O-]'):
        return True
    # Formate O=C[O-]
    if has_substructure(mol, '[CH1](=O)[O-]'):
        return True
    return False


def is_amino_acid_anion(smi, mol):
    """Check if anion is an amino acid type (has both COO- and a free NH2)."""
    if mol is None:
        return False
    return has_substructure(mol, '[CX3](=O)[O-]') and has_substructure(mol, '[NX3;H2]')


def tier1_check(cat_smi, an_smi):
    """
    Tier I: Proton transfer stability.
    Returns (flagged: bool, reason: str).
    """
    cat_mol = mol_from_smiles(cat_smi)
    an_mol = mol_from_smiles(an_smi)

    protic = is_protic_cation(cat_smi, cat_mol)
    basic = is_strongly_basic_anion(an_smi, an_mol)

    if protic and basic:
        reasons = []
        reasons.append("Protic cation")
        if is_amino_acid_anion(an_smi, an_mol):
            reasons.append("amino acid anion (COO-/NH2)")
        else:
            reasons.append("strongly basic anion (carboxylate/alkoxide)")
        return True, "; ".join(reasons)
    return False, ""


# ---------------------------------------------------------------------------
# Tier II  -  Melting Point Screening  (score 0-6, higher = more liquid-like)
# ---------------------------------------------------------------------------

def tier2_score_calc(cat_smi, an_smi):
    """
    Empirical melting-point likelihood score (0-6).
    +1 for each favourable factor:
      (i)   asymmetric cation
      (ii)  diffuse / delocalised charge on cation
      (iii) flexible side chains on cation
      (iv)  large anion
      (v)   weak H-bonding capacity
      (vi)  low symmetry (combined ion pair)
    """
    cat_mol = mol_from_smiles(cat_smi)
    an_mol = mol_from_smiles(an_smi)
    score = 0

    if cat_mol is None or an_mol is None:
        return 0

    # (i) Asymmetric cation --------------------------------------------------
    # Heuristic: count distinct substituent heavy-atom counts around the charged N/P/S.
    # If they are not all the same  ->  asymmetric.
    charged_atoms = [a for a in cat_mol.GetAtoms()
                     if a.GetFormalCharge() > 0]
    asymmetric = False
    for ca in charged_atoms:
        nbr_sizes = []
        for nbr in ca.GetNeighbors():
            # BFS to count fragment size excluding the charged centre
            visited = {ca.GetIdx()}
            stack = [nbr.GetIdx()]
            cnt = 0
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                visited.add(idx)
                cnt += 1
                for nn in cat_mol.GetAtomWithIdx(idx).GetNeighbors():
                    if nn.GetIdx() not in visited:
                        stack.append(nn.GetIdx())
            nbr_sizes.append(cnt)
        if len(set(nbr_sizes)) > 1:
            asymmetric = True
            break
    if asymmetric:
        score += 1

    # (ii) Diffuse / delocalised charge on cation ----------------------------
    # Aromatic cations (imidazolium, pyridinium, etc.) or phosphonium/sulfonium
    # have more diffuse charge than simple ammonium.
    if has_substructure(cat_mol, '[n+]') or has_substructure(cat_mol, '[nH+]'):
        score += 1  # aromatic N+ -> delocalised
    elif has_substructure(cat_mol, '[S+]') or has_substructure(cat_mol, '[P+]'):
        score += 1  # S/P centres spread charge better

    # (iii) Flexible side chains on cation -----------------------------------
    n_rotatable_cat = rdMolDescriptors.CalcNumRotatableBonds(cat_mol)
    if n_rotatable_cat >= 2:
        score += 1

    # (iv) Large anion -------------------------------------------------------
    an_mw = Descriptors.MolWt(an_mol)
    if an_mw > 150:
        score += 1

    # (v) Weak H-bonding capacity --------------------------------------------
    # Fewer H-bond donors + acceptors in the combined pair -> weaker lattice.
    total_hbd = rdMolDescriptors.CalcNumHBD(cat_mol) + rdMolDescriptors.CalcNumHBD(an_mol)
    total_hba = rdMolDescriptors.CalcNumHBA(cat_mol) + rdMolDescriptors.CalcNumHBA(an_mol)
    if total_hbd <= 1 and total_hba <= 4:
        score += 1

    # (vi) Low symmetry (ion pair) -------------------------------------------
    # Simple proxy: cation & anion have different heavy-atom counts and
    # cation is not a highly symmetric small ion (e.g. NH4+, NMe4+).
    cat_ha = cat_mol.GetNumHeavyAtoms()
    an_ha = an_mol.GetNumHeavyAtoms()
    # Highly symmetric small cations: tetramethylammonium-like (4 identical substituents)
    is_small_symmetric = (cat_ha <= 5 and not asymmetric)
    if not is_small_symmetric and abs(cat_ha - an_ha) >= 2:
        score += 1

    return score


def tier2_check(cat_smi, an_smi):
    """
    Tier II: Melting-point screening.
    Returns (score: int, flagged: bool).
    Flag if score < 3.
    """
    score = tier2_score_calc(cat_smi, an_smi)
    flagged = score < 3
    return score, flagged


# ---------------------------------------------------------------------------
# Tier III  -  Chemical Reactivity Flags
# ---------------------------------------------------------------------------

def tier3_check(cat_smi, an_smi):
    """
    Tier III: Chemical reactivity.
    Returns (flagged: bool, reason: str).
    """
    cat_mol = mol_from_smiles(cat_smi)
    an_mol = mol_from_smiles(an_smi)
    reasons = []

    if cat_mol is None or an_mol is None:
        return False, ""

    # (i) Strongly nucleophilic anion + electrophilic cation -----------------
    nucleophilic_anion = False
    nuc_name = ""
    # Cyanide [C-]#N or [CN-]
    if has_substructure(an_mol, '[C-]#N') or has_substructure(an_mol, '[#6-]~#[#7]'):
        nucleophilic_anion = True
        nuc_name = "cyanide-type"
    # Azide [N-]=N=N or [N-]=[N+]=[N-]
    if has_substructure(an_mol, '[N-]=[N+]=[N-]') or has_substructure(an_mol, '[N-]=N=N'):
        nucleophilic_anion = True
        nuc_name = "azide"
    # Dicyanamide N#C[N-]C#N
    if has_substructure(an_mol, 'N#C[N-]C#N'):
        nucleophilic_anion = True
        nuc_name = "dicyanamide"
    # Tricyanomethanide N#C[C-](C#N)C#N
    if has_substructure(an_mol, 'N#C[C-](C#N)C#N'):
        nucleophilic_anion = True
        nuc_name = "tricyanomethanide"
    # Imidazolide (deprotonated imidazole) c1c[n-]cn1
    if has_substructure(an_mol, 'c1c[n-]cn1'):
        nucleophilic_anion = True
        nuc_name = "imidazolide"

    electrophilic_cation = False
    elec_name = ""
    # Cation with electrophilic C: ester, acyl, carbonyl in ring, etc.
    if has_substructure(cat_mol, '[CX3](=O)[OX2]'):  # ester in cation
        electrophilic_cation = True
        elec_name = "ester moiety in cation"
    if has_substructure(cat_mol, '[CX3](=O)[NX3]'):  # amide/lactam in cation
        electrophilic_cation = True
        elec_name = "amide/lactam moiety in cation"

    if nucleophilic_anion and electrophilic_cation:
        reasons.append(f"Nucleophilic anion ({nuc_name}) + electrophilic cation ({elec_name})")

    # Also flag strongly nucleophilic anion even without obvious electrophilic
    # centre if the anion is a very strong nucleophile (CN-, N3-)
    strong_nuc_smarts = [
        ('[C-]#N', 'free cyanide'),
        ('[N-]=[N+]=[N-]', 'azide'),
    ]
    for smarts, name in strong_nuc_smarts:
        if has_substructure(an_mol, smarts):
            # Only flag if not already caught above
            if not any("Nucleophilic" in r for r in reasons):
                reasons.append(f"Strongly nucleophilic anion ({name})")

    # (ii) Oxidizing anion + easily oxidized cation --------------------------
    oxidizing_anion = False
    ox_name = ""
    if has_substructure(an_mol, '[N+](=O)([O-])[O-]') or '[NO3-]' in an_smi:
        oxidizing_anion = True
        ox_name = "nitrate"
    # Perchlorate
    if has_substructure(an_mol, '[Cl](=O)(=O)(=O)[O-]'):
        oxidizing_anion = True
        ox_name = "perchlorate"

    easily_oxidized_cation = False
    oxid_cat_name = ""
    # Thioether / sulfonium
    if has_substructure(cat_mol, '[S+]') or has_substructure(cat_mol, '[#16X2]'):
        easily_oxidized_cation = True
        oxid_cat_name = "sulfonium/thioether"
    # Allyl or vinyl groups
    if has_substructure(cat_mol, 'C=C'):
        easily_oxidized_cation = True
        oxid_cat_name = "unsaturated cation"

    if oxidizing_anion and easily_oxidized_cation:
        reasons.append(f"Oxidizing anion ({ox_name}) + oxidizable cation ({oxid_cat_name})")

    # (iii) Reactive functional groups (either ion) --------------------------
    reactive_patterns = [
        ('[CX3](=O)[FX1,ClX1,BrX1,IX1]', 'acyl halide'),
        ('[CX4]1[OX2][CX4]1', 'epoxide'),
        ('[NX2]=[NX2]', 'diazo'),
        ('[OX1]=[OX1]', 'peroxide-like'),
        ('[#6]([FX1])([FX1])([FX1])[SX4]', 'triflyl (potential HF release)'),
    ]
    # Mercury-containing species
    if has_substructure(an_mol, '[Hg]') or has_substructure(cat_mol, '[Hg]'):
        reasons.append("Contains mercury (toxic/reactive heavy metal)")

    for smarts, name in reactive_patterns:
        if has_substructure(cat_mol, smarts) or has_substructure(an_mol, smarts):
            reasons.append(f"Reactive group: {name}")

    flagged = len(reasons) > 0
    return flagged, "; ".join(reasons)


# ---------------------------------------------------------------------------
# Overall classification
# ---------------------------------------------------------------------------

def classify(t1_flag, t2_flag, t3_flag):
    """
    Overall status:
      Fail     - Tier I flagged  (proton transfer -> IL is thermodynamically unstable)
      Fail     - Tier III flagged (reactive -> safety/decomposition risk)
      Caution  - Tier II flagged only (may be solid, but could still work)
      Pass     - no flags
    """
    if t1_flag or t3_flag:
        return "Fail"
    if t2_flag:
        return "Caution"
    return "Pass"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_path = 'D:/wuzeting/output/final_ranked_candidates.csv'
    output_path = 'D:/wuzeting/output/stability_screening.csv'

    # Read candidates
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        candidates = list(reader)

    print(f"Loaded {len(candidates)} candidates from {input_path}\n")

    results = []

    for row in candidates:
        final_rank = int(row['final_rank'])
        cat_smi = row['cation']
        an_smi = row['anion']

        t1_flag, t1_reason = tier1_check(cat_smi, an_smi)
        t2_score, t2_flag = tier2_check(cat_smi, an_smi)
        t3_flag, t3_reason = tier3_check(cat_smi, an_smi)
        status = classify(t1_flag, t2_flag, t3_flag)

        results.append({
            'rank': final_rank,
            'cation': cat_smi,
            'anion': an_smi,
            'tier1_flag': t1_flag,
            'tier1_reason': t1_reason,
            'tier2_score': t2_score,
            'tier2_flag': t2_flag,
            'tier3_flag': t3_flag,
            'tier3_reason': t3_reason,
            'overall_status': status,
        })

    # Sort by rank
    results.sort(key=lambda r: r['rank'])

    # Write output
    fieldnames = ['rank', 'cation', 'anion',
                  'tier1_flag', 'tier1_reason',
                  'tier2_score', 'tier2_flag',
                  'tier3_flag', 'tier3_reason',
                  'overall_status']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}\n")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    n = len(results)
    status_counts = Counter(r['overall_status'] for r in results)
    t1_flagged = sum(1 for r in results if r['tier1_flag'])
    t2_flagged = sum(1 for r in results if r['tier2_flag'])
    t3_flagged = sum(1 for r in results if r['tier3_flag'])

    # Candidates that pass Tier I
    pass_t1 = [r for r in results if not r['tier1_flag']]
    # Of those, flagged by Tier II only (no Tier III)
    caution_t2_only = [r for r in pass_t1 if r['tier2_flag'] and not r['tier3_flag']]
    # Candidates that pass Tier I but fail Tier III
    fail_t3_after_t1 = [r for r in pass_t1 if r['tier3_flag']]

    print("=" * 72)
    print("TABLE 3: Attrition at Each Stability-Screening Tier")
    print("=" * 72)
    print(f"{'Tier':<35} {'Flagged':>10} {'Remaining':>10}")
    print("-" * 72)
    print(f"{'Starting pool':<35} {'':>10} {n:>10}")
    print(f"{'Tier I  (proton transfer)':<35} {t1_flagged:>10} {n - t1_flagged:>10}")
    remaining_after_t1 = n - t1_flagged
    # Tier III applied to those that passed Tier I
    t3_among_pass_t1 = sum(1 for r in pass_t1 if r['tier3_flag'])
    remaining_after_t3 = remaining_after_t1 - t3_among_pass_t1
    print(f"{'Tier III (reactivity)':<35} {t3_among_pass_t1:>10} {remaining_after_t3:>10}")
    # Tier II caution (not removed, but flagged)
    t2_among_remaining = sum(1 for r in results
                             if not r['tier1_flag'] and not r['tier3_flag'] and r['tier2_flag'])
    final_pass = sum(1 for r in results if r['overall_status'] == 'Pass')
    print(f"{'Tier II (melting point, Caution)':<35} {t2_among_remaining:>10} {'':>10}")
    print("-" * 72)
    print(f"{'Final Pass':<35} {'':>10} {final_pass:>10}")
    print(f"{'Final Caution':<35} {'':>10} {status_counts.get('Caution', 0):>10}")
    print(f"{'Final Fail':<35} {'':>10} {status_counts.get('Fail', 0):>10}")
    print("=" * 72)
    print()

    fail_pct = status_counts.get('Fail', 0) / n * 100
    print(f"Attrition: {status_counts.get('Fail', 0)}/{n} = {fail_pct:.1f}% of top-100 FAIL")
    print()

    # Top 10 post-filter candidates (Pass first, then Caution, by rank)
    survivors = [r for r in results if r['overall_status'] in ('Pass', 'Caution')]
    # Sort: Pass before Caution, then by rank
    survivors.sort(key=lambda r: (0 if r['overall_status'] == 'Pass' else 1, r['rank']))
    top10 = survivors[:10]

    print("=" * 72)
    print("TOP 10 POST-FILTER CANDIDATES")
    print("=" * 72)
    print(f"{'Rank':<6} {'Status':<9} {'T2 Score':<9} {'Cation':<35} {'Anion'}")
    print("-" * 100)
    for r in top10:
        print(f"{r['rank']:<6} {r['overall_status']:<9} {r['tier2_score']:<9} "
              f"{r['cation']:<35} {r['anion']}")
    print("=" * 72)
    print()

    # Detail of Tier I failures
    t1_failures = [r for r in results if r['tier1_flag']]
    print(f"\n--- Tier I failures ({len(t1_failures)} candidates) ---")
    for r in t1_failures:
        print(f"  Rank {r['rank']:>3}: {r['cation']} + {r['anion']}  -->  {r['tier1_reason']}")

    # Detail of Tier III failures
    t3_failures = [r for r in results if r['tier3_flag']]
    print(f"\n--- Tier III failures ({len(t3_failures)} candidates) ---")
    for r in t3_failures:
        print(f"  Rank {r['rank']:>3}: {r['cation']} + {r['anion']}  -->  {r['tier3_reason']}")

    # Detail of Tier II cautions (not fail)
    t2_cautions = [r for r in results
                   if r['tier2_flag'] and not r['tier1_flag'] and not r['tier3_flag']]
    print(f"\n--- Tier II cautions ({len(t2_cautions)} candidates) ---")
    for r in t2_cautions:
        print(f"  Rank {r['rank']:>3}: {r['cation']} + {r['anion']}  (score={r['tier2_score']})")


if __name__ == '__main__':
    main()
