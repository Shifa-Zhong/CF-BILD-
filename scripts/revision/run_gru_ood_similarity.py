'''Quantify GRU-to-training structural proximity with Morgan Tanimoto scores.'''

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_bild.fragment_vocab import canonicalize_smiles  # noqa: E402


DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output' / 'revision_2026'
PUBLIC_REFERENCE_PATH = OUTPUT_DIR / 'gru_reference_ions_revision.json'


def fingerprint(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius=2, nBits=2048
    )


def non_test_ions(property_name, column):
    frames = [
        pd.read_csv(
            DATA_DIR / f'{role}_1_group_{property_name}.csv',
            usecols=[column],
        )
        for role in ('train', 'val')
    ]
    return sorted({
        canonicalize_smiles(value.strip())
        for value in pd.concat(frames, ignore_index=True)[column]
        .dropna().astype(str)
        if canonicalize_smiles(value.strip()) is not None
    })


def property_tables_available():
    return all(
        (DATA_DIR / f'{role}_1_group_{property_name}.csv').exists()
        for property_name in ('co2', 'vis', 'tox')
        for role in ('train', 'val')
    )


def load_or_build_references():
    '''Return derived non-test ion sets without exposing source records.'''
    if property_tables_available():
        reference = {}
        for property_name in ('co2', 'vis', 'tox'):
            reference[(property_name, 'cation')] = non_test_ions(
                property_name, 'new_cation'
            )
            reference[(property_name, 'anion')] = non_test_ions(
                property_name, 'new_anion'
            )
        reference[('union', 'cation')] = sorted(set().union(*[
            set(reference[(name, 'cation')])
            for name in ('co2', 'vis', 'tox')
        ]))
        reference[('union', 'anion')] = sorted(set().union(*[
            set(reference[(name, 'anion')])
            for name in ('co2', 'vis', 'tox')
        ]))
        payload = {
            'artifact_version': 'DD-major-revision-2026-09',
            'content_note': (
                'Canonical ion identities derived from non-test records; '
                'no property values or source records.'
            ),
            'references': {
                f'{property_name}_{ion_type}': values
                for (property_name, ion_type), values in reference.items()
            },
        }
        with PUBLIC_REFERENCE_PATH.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return reference
    with PUBLIC_REFERENCE_PATH.open(encoding='utf-8') as handle:
        payload = json.load(handle)
    return {
        tuple(key.rsplit('_', 1)): values
        for key, values in payload['references'].items()
    }


def maximum_similarity(query_smiles, references):
    reference_fingerprints = [
        value for value in map(fingerprint, references) if value is not None
    ]
    scores = np.full(len(query_smiles), np.nan, dtype=float)
    for index, smiles in enumerate(query_smiles):
        query = fingerprint(smiles)
        if query is not None and reference_fingerprints:
            scores[index] = max(DataStructs.BulkTanimotoSimilarity(
                query, reference_fingerprints
            ))
    return scores


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        'n': int(len(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'fraction_ge_0.7': float(np.mean(values >= 0.7)),
        'fraction_ge_0.8': float(np.mean(values >= 0.8)),
        'fraction_exact': float(np.mean(values == 1.0)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_all = pd.read_csv(
        ROOT / 'data' / 'gru' / 'generate_result.csv'
    )
    generated = generated_all.sample(20000, random_state=42).copy()
    generated['cation_canonical'] = generated['cation'].map(canonicalize_smiles)
    generated['anion_canonical'] = generated['anion'].map(canonicalize_smiles)

    reference = load_or_build_references()

    result = {'fingerprint': 'Morgan radius 2, 2048 bits', 'comparisons': {}}
    for property_name in ('co2', 'vis', 'tox', 'union'):
        result['comparisons'][property_name] = {}
        for ion_type, column in (
            ('cation', 'cation_canonical'),
            ('anion', 'anion_canonical'),
        ):
            values = maximum_similarity(
                generated[column],
                reference[(property_name, ion_type)],
            )
            generated[f'{property_name}_{ion_type}_max_tanimoto'] = values
            result['comparisons'][property_name][ion_type] = summarize(values)
            result['comparisons'][property_name][ion_type][
                'n_reference_ions'
            ] = len(reference[(property_name, ion_type)])

    union_pair_min = np.minimum(
        generated['union_cation_max_tanimoto'].to_numpy(),
        generated['union_anion_max_tanimoto'].to_numpy(),
    )
    result['comparisons']['union']['pair_minimum'] = summarize(union_pair_min)
    generated.to_csv(
        OUTPUT_DIR / 'gru_ood_similarity_candidates.csv', index=False
    )
    with (OUTPUT_DIR / 'gru_ood_similarity_summary.json').open(
        'w', encoding='utf-8'
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
