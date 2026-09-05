"""Recompute numerical vocabulary, sparsity, scaffold and association diagnostics.

No figures or documents are generated. Separate cation/anion PCA/t-SNE maps
are intentionally not placed in a shared coordinate system.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cf_bild.fragment_vocab import FragmentVocabulary, canonicalize_smiles, TARGET_COLUMNS
from cf_bild.ion_families import classify_cation, classify_anion


def read_partition(run, prop, role):
    names = [f'{r}_1_group_{prop}.csv' for r in ('train', 'val')] if role == 'non_test' else [f'test_group_{prop}.csv']
    return pd.concat([pd.read_csv(run / 'data' / name) for name in names], ignore_index=True)


def main(run):
    run = Path(run).resolve()
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    vocab = FragmentVocabulary()
    vocab.load(run / 'fragment_vocab.pkl')
    provenance = defaultdict(lambda: {'properties': set(), 'roles': set(), 'files': set()})
    for path in sorted((run / 'data').glob('*.csv')):
        prop = path.stem.split('_')[-1]
        role = path.stem.split('_')[0]
        frame = pd.read_csv(path)
        for kind, column in [('cation', 'new_cation'), ('anion', 'new_anion')]:
            for raw in frame[column].unique():
                entry = provenance[(kind, canonicalize_smiles(raw))]
                entry['properties'].add(prop); entry['roles'].add(role); entry['files'].add(path.name)
    rows, mappings, sparsity = [], [], {}
    for kind, smiles_list, fps, classify in [('cation', list(vocab.cations), vocab.cation_fps, classify_cation),
                                            ('anion', list(vocab.anions), vocab.anion_fps, classify_anion)]:
        matrix = np.stack([fps[s] for s in smiles_list])
        active = np.count_nonzero(matrix, axis=1)
        sparsity[kind] = {'n_features': matrix.shape[1], 'mean_nonzero_positions': float(active.mean()),
                          'mean_active_fraction': float(active.mean()/matrix.shape[1]),
                          'mean_total_environment_count': float(matrix.sum(axis=1).mean()),
                          'feature_value_max': float(matrix.max()), 'binary': bool(np.isin(matrix, [0, 1]).all())}
        reduced = PCA(n_components=min(50, *matrix.shape), svd_solver='full').fit_transform(matrix)
        coordinates = TSNE(n_components=2, random_state=42, perplexity=min(30, len(smiles_list)//4),
                           init='pca', learning_rate='auto', max_iter=1000).fit_transform(reduced)
        for i, smiles in enumerate(smiles_list):
            family = classify(smiles)
            mw = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
            source = provenance[(kind, smiles)]
            rows.append({'smiles': smiles, 'type': kind, 'family': family, 'molecular_weight': mw,
                'source_properties': ';'.join(sorted(source['properties'])),
                'source_split_roles': ';'.join(sorted(source['roles'])), 'source_files': ';'.join(sorted(source['files']))})
            mappings.append({'smiles': smiles, 'type': kind, 'family': family,
                             'tsne_1': float(coordinates[i, 0]), 'tsne_2': float(coordinates[i, 1])})
    table = pd.DataFrame(rows)
    table[['smiles', 'type', 'family', 'molecular_weight']].to_csv(out / 'table_s2_fragment_vocabulary.csv', index=False)
    table.rename(columns={'smiles': 'canonical_smiles', 'type': 'ion_type'}).to_csv(out / 'fragment_vocabulary_provenance.csv', index=False)
    table.rename(columns={'molecular_weight': 'mw'}).to_csv(out / 'vocabulary_ions_with_mw.csv', index=False)
    for kind, name in [('cation', 'vocabulary_cation_families.csv'), ('anion', 'vocabulary_anion_families.csv')]:
        table.loc[table.type.eq(kind)].groupby('family').size().rename('count').reset_index().to_csv(out / name, index=False)
    pd.DataFrame(mappings).to_csv(out / 'vocabulary_tsne_coordinates.csv', index=False)
    aggregates, scaffolds = {}, {}
    for prop in ('co2', 'vis', 'tox'):
        frames = {role: read_partition(run, prop, role) for role in ('non_test', 'test')}
        all_records = pd.concat(list(frames.values()), ignore_index=True)
        aggregates[prop] = all_records.groupby('group')[TARGET_COLUMNS[prop]].mean().rename(prop)
        sets = {}
        for role, frame in frames.items():
            cations = {canonicalize_smiles(s) for s in frame.new_cation}
            murcko = {MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in cations}
            generic = {Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(s))) if s else '' for s in murcko}
            sets[role] = {'cations': cations, 'murcko': murcko, 'generic': generic}
        scaffolds[prop] = {}
        for level in ('cations', 'murcko', 'generic'):
            tr, te = sets['non_test'][level], sets['test'][level]
            scaffolds[prop][level] = {'non_test_unique': len(tr), 'test_unique': len(te), 'shared': len(tr & te),
                'test_only': len(te-tr), 'shared_fraction_of_test': len(tr & te)/max(len(te), 1)}
        scaffolds[prop]['acyclic_empty_murcko_bucket_included'] = True
    correlations = []
    labels = {'co2': 'CO2', 'vis': 'Viscosity', 'tox': 'Toxicity'}
    for p, q in [('co2', 'vis'), ('co2', 'tox'), ('vis', 'tox')]:
        matched = pd.concat([aggregates[p], aggregates[q]], axis=1, join='inner').dropna()
        r, rp = pearsonr(matched[p], matched[q]); s, sp = spearmanr(matched[p], matched[q])
        correlations.append({'property_1': labels[p], 'property_2': labels[q], 'n_overlapping_ILs': len(matched),
                             'pearson_r': r, 'pearson_p': rp, 'spearman_r': s, 'spearman_p': sp})
        matched.to_csv(out / f'correlation_pairs_{p}_{q}.csv')
    pd.DataFrame(correlations).to_csv(out / 'inter_property_correlation.csv', index=False)
    metadata = {'sparsity_raw_counts': sparsity, 'scaffold_overlap': scaffolds,
        'family_classification': 'Ordered RDKit charged-ring/functional-group motif bins; see cf_bild/ion_families.py. These descriptive labels do not enter the GP or stability rules.',
        'family_classifier_sha256': hashlib.sha256((ROOT / 'cf_bild/ion_families.py').read_bytes()).hexdigest(),
        'correlation_protocol': 'Species means over all retained measurements and conditions; non-test and test included for descriptive associations only, never model selection',
        'embedding': {'separate_ion_spaces': True, 'PCA_components_max': 50, 'PCA_solver': 'full', 'TSNE_seed': 42,
            'TSNE_perplexity': 30, 'TSNE_initialization': 'pca', 'TSNE_learning_rate': 'auto', 'TSNE_iterations': 1000,
            'cross_map_distances_meaningful': False},
        'input_curation_sha256': hashlib.sha256((run / 'CURATION_SUMMARY.json').read_bytes()).hexdigest()}
    (out / 'clean_data_diagnostics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'sparsity': sparsity, 'correlations': correlations, 'scaffolds': scaffolds}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    main(parser.parse_args().run_directory)
