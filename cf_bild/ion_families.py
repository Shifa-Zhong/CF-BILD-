"""Descriptive structural-family heuristics used for vocabulary summaries.

Categories are descriptive, mutually exclusive motif bins, not stability
validation. Specific charged-ring motifs precede element-based fallbacks.
"""
from rdkit import Chem


def classify_cation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError('Cannot classify an unparseable ion')
    rings = [[mol.GetAtomWithIdx(i) for i in ring] for ring in mol.GetRingInfo().AtomRings()]
    for atoms in rings:
        nitrogens = [a for a in atoms if a.GetSymbol() == 'N']
        if len(atoms) == 5 and len(nitrogens) == 2 and sum(a.GetSymbol() == 'C' for a in atoms) == 3:
            if all(a.GetIsAromatic() for a in atoms) and any(a.GetFormalCharge() > 0 for a in nitrogens):
                adjacent = mol.GetBondBetweenAtoms(nitrogens[0].GetIdx(), nitrogens[1].GetIdx()) is not None
                return 'pyrazolium' if adjacent else 'imidazolium'
    for atoms in rings:
        if len(atoms) == 6 and sum(a.GetSymbol() == 'N' for a in atoms) == 1 and sum(a.GetSymbol() == 'C' for a in atoms) == 5:
            if all(a.GetIsAromatic() for a in atoms) and any(a.GetFormalCharge() > 0 for a in atoms): return 'pyridinium'
    for size, family in [(5, 'pyrrolidinium'), (6, 'piperidinium')]:
        for atoms in rings:
            if len(atoms) == size and sum(a.GetSymbol() == 'N' for a in atoms) == 1 and sum(a.GetSymbol() == 'C' for a in atoms) == size-1:
                if any(a.GetSymbol() == 'N' and a.GetFormalCharge() > 0 for a in atoms) and not any(a.GetIsAromatic() for a in atoms): return family
    for atoms in rings:
        if len(atoms) == 6 and sorted(a.GetSymbol() for a in atoms) == ['C', 'C', 'C', 'C', 'N', 'O']:
            if any(a.GetSymbol() == 'N' and a.GetFormalCharge() > 0 for a in atoms) and not any(a.GetIsAromatic() for a in atoms): return 'morpholinium'
    for symbol, family in [('P', 'phosphonium'), ('S', 'sulfonium')]:
        if any(a.GetSymbol() == symbol and a.GetFormalCharge() > 0 for a in mol.GetAtoms()): return family
    if any(a.GetSymbol() == 'N' and a.GetFormalCharge() > 0 and a.IsInRing() for a in mol.GetAtoms()): return 'other N-heterocycle'
    if any(a.GetSymbol() == 'N' and a.GetFormalCharge() > 0 and not a.GetIsAromatic() for a in mol.GetAtoms()): return 'ammonium'
    return 'other'


def classify_anion(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError('Cannot classify an unparseable ion')
    if any(a.GetSymbol() == 'F' for a in mol.GetAtoms()): return 'fluorinated'
    amino = mol.HasSubstructMatch(Chem.MolFromSmarts('[N;H1,H2]-[CX4]-[CX3](=[OX1])[O-]'))
    carboxylate = mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]-[CX3](=[OX1])[O-]')) or mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3H1](=[OX1])[O-]'))
    if amino and carboxylate: return 'amino-acid'
    if carboxylate: return 'carboxylate'
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[SX4](=[OX1])(=[OX1])[O-]')): return 'sulfonate/sulfate'
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[P](=[O])[O-]')): return 'P-oxoanion'
    return 'other'
