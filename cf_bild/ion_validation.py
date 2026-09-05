"""Formal-charge checks; never guess protonation or alter source structures.

The current two-block representation is restricted to singly charged 1:1
pairs. Multivalent salts need an explicit stoichiometric representation and
are not chemically invalid merely because they are outside that scope.
"""

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=16384)
def component_info(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None
    return Chem.GetFormalCharge(mol), len(Chem.GetMolFrags(mol))


@dataclass(frozen=True)
class IonPairCheck:
    cation_charge: int | None
    anion_charge: int | None
    valid: bool
    reason: str


def check_ion_pair(cation, anion):
    qc, nc = component_info(cation)
    qa, na = component_info(anion)
    if qc is None or qa is None:
        reason = 'SMILES_parse_failure'
    elif nc != 1 or na != 1:
        reason = 'disconnected_component_requires_source_review'
    elif qc <= 0 or qa >= 0:
        reason = 'neutral_component_or_incorrect_ion_role'
    elif (qc, qa) != (1, -1):
        reason = 'multivalent_salt_requires_explicit_stoichiometry'
    else:
        reason = 'valid_monovalent_1_to_1_pair'
    return IonPairCheck(qc, qa, reason == 'valid_monovalent_1_to_1_pair', reason)


def require_valid_pairs(pairs, context='ion-pair inputs'):
    invalid = []
    for index, (cation, anion) in enumerate(pairs):
        result = check_ion_pair(cation, anion)
        if not result.valid:
            invalid.append((index, result.reason))
    if invalid:
        sample = ', '.join(f'{i}: {reason}' for i, reason in invalid[:5])
        raise ValueError(
            f'{context}: {len(invalid)} pairs fail the explicit +1/-1 '
            f'1:1 contract ({sample}). Preserve source records and resolve '
            'identities or use an author-approved exclusion policy before '
            'fitting, ranking, or assigning Pass.'
        )


def guarded_stability_status(cation, anion, tier1, tier2, tier3):
    if not check_ion_pair(cation, anion).valid:
        return 'Invalid representation'
    if tier1 or tier3:
        return 'Fail'
    return 'Caution' if tier2 else 'Pass'
