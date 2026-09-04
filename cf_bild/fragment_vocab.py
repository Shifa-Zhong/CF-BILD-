"""
Layer 1: Fragment Vocabulary Construction
- Extract unique cations and anions from property datasets
- Compute CF-MF (Collision-Free Morgan Fingerprint) for each fragment
- Build combinatorial space X = C x A
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from collections import OrderedDict
import os
import pickle

from bit_collision_free_MF.fingerprint import CollisionFreeMorganFP


TARGET_COLUMNS = {
    'co2': 'CO2-exp',
    'vis': 'vis',
    'tox': 'Experimental logEC50',
}


def canonicalize_smiles(smi):
    """Canonicalize a SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


class FragmentVocabulary:
    """Fragment vocabulary for ionic liquids using CF-MF fingerprints."""

    def __init__(self, fp_radius=2):
        self.fp_radius = fp_radius
        self.cations = OrderedDict()   # canonical SMILES -> index
        self.anions = OrderedDict()    # canonical SMILES -> index
        self.cation_fps = {}           # canonical SMILES -> fingerprint array
        self.anion_fps = {}
        self.cfmf_cat = None           # Fitted CF-MF generator for cations
        self.cfmf_an = None            # Fitted CF-MF generator for anions
        self.cat_fp_length = None      # Optimal FP length for cations
        self.an_fp_length = None       # Optimal FP length for anions

    def extract_fragments_from_dataframes(self, dfs):
        """Extract unique cations and anions from a list of DataFrames."""
        all_cations = set()
        all_anions = set()

        for df in dfs:
            for _, row in df.iterrows():
                cat = canonicalize_smiles(str(row['new_cation']).strip())
                an = canonicalize_smiles(str(row['new_anion']).strip())
                if cat is not None:
                    all_cations.add(cat)
                if an is not None:
                    all_anions.add(an)

        for i, cat in enumerate(sorted(all_cations)):
            self.cations[cat] = i
        for i, an in enumerate(sorted(all_anions)):
            self.anions[an] = i

        print(f"Fragment vocabulary: {len(self.cations)} cations, {len(self.anions)} anions")
        print(f"Combinatorial space: {len(self.cations) * len(self.anions):,} candidate ILs")

    def compute_fingerprints(self):
        """Compute CF-MF fingerprints for all fragments.

        1. Fit CF-MF on ALL unique cations to determine optimal cation FP length
        2. Fit CF-MF on ALL unique anions to determine optimal anion FP length
        3. Generate fingerprints for each ion
        """
        cat_smiles = list(self.cations.keys())
        an_smiles = list(self.anions.keys())

        # Fit CF-MF for cations (remove_zero_columns to reduce dimensionality)
        print(f"Fitting CF-MF for {len(cat_smiles)} cations (radius={self.fp_radius})...")
        self.cfmf_cat = CollisionFreeMorganFP(radius=self.fp_radius)
        self.cfmf_cat.fit(cat_smiles, remove_zero_columns=True)
        print(f"  Cation CF-MF: collision-free length={self.cfmf_cat.length}, "
              f"zero columns removed={len(self.cfmf_cat._zero_columns)}")

        # Fit CF-MF for anions
        print(f"Fitting CF-MF for {len(an_smiles)} anions (radius={self.fp_radius})...")
        self.cfmf_an = CollisionFreeMorganFP(radius=self.fp_radius)
        self.cfmf_an.fit(an_smiles, remove_zero_columns=True)
        print(f"  Anion CF-MF: collision-free length={self.cfmf_an.length}, "
              f"zero columns removed={len(self.cfmf_an._zero_columns)}")

        # Generate fingerprints (zero columns auto-removed by transform)
        cat_fp_matrix = self.cfmf_cat.transform(cat_smiles)
        an_fp_matrix = self.cfmf_an.transform(an_smiles)

        self.cat_fp_length = cat_fp_matrix.shape[1]  # Effective dim after zero removal
        self.an_fp_length = an_fp_matrix.shape[1]

        for i, cat in enumerate(cat_smiles):
            self.cation_fps[cat] = cat_fp_matrix[i].astype(np.float64)
        for i, an in enumerate(an_smiles):
            self.anion_fps[an] = an_fp_matrix[i].astype(np.float64)

        print(f"CF-MF fingerprints: {len(self.cation_fps)} cations ({self.cat_fp_length}d), "
              f"{len(self.anion_fps)} anions ({self.an_fp_length}d)")
        print(f"Total IL fingerprint dimension: {self.cat_fp_length + self.an_fp_length}")

    def get_il_fingerprint(self, cation_smi, anion_smi):
        """Get cation FP, anion FP, and concatenated FP for an IL."""
        cat = canonicalize_smiles(cation_smi)
        an = canonicalize_smiles(anion_smi)

        cat_fp = self.cation_fps.get(cat)
        an_fp = self.anion_fps.get(an)

        # For unknown ions, compute on the fly using fitted CF-MF
        if cat_fp is None and self.cfmf_cat is not None and cat is not None:
            try:
                cat_fp = self.cfmf_cat.transform([cat])[0].astype(np.float64)
            except Exception:
                cat_fp = None
        if an_fp is None and self.cfmf_an is not None and an is not None:
            try:
                an_fp = self.cfmf_an.transform([an])[0].astype(np.float64)
            except Exception:
                an_fp = None

        if cat_fp is None or an_fp is None:
            return None, None, None

        return cat_fp, an_fp, np.concatenate([cat_fp, an_fp])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'cations': self.cations,
                'anions': self.anions,
                'cation_fps': self.cation_fps,
                'anion_fps': self.anion_fps,
                'fp_radius': self.fp_radius,
                'cat_fp_length': self.cat_fp_length,
                'an_fp_length': self.an_fp_length,
                'cfmf_cat': self.cfmf_cat,
                'cfmf_an': self.cfmf_an,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.cations = data['cations']
        self.anions = data['anions']
        self.cation_fps = data['cation_fps']
        self.anion_fps = data['anion_fps']
        self.fp_radius = data['fp_radius']
        self.cat_fp_length = data['cat_fp_length']
        self.an_fp_length = data['an_fp_length']
        self.cfmf_cat = data['cfmf_cat']
        self.cfmf_an = data['cfmf_an']


def load_property_datasets(data_dir, n_folds=5):
    """Load all property datasets with 5-fold train/val + 1 test structure.

    Returns:
        datasets: dict[prop] -> {
            'folds': [(train_df_1, val_df_1), ..., (train_df_5, val_df_5)],
            'test': test_df
        }
    """
    datasets = {}
    for prop in ['co2', 'vis', 'tox']:
        folds = []
        for fold_i in range(1, n_folds + 1):
            train_path = os.path.join(data_dir, f'train_{fold_i}_group_{prop}.csv')
            val_path = os.path.join(data_dir, f'val_{fold_i}_group_{prop}.csv')
            if os.path.exists(train_path) and os.path.exists(val_path):
                train_df = pd.read_csv(train_path)
                val_df = pd.read_csv(val_path)
                folds.append((train_df, val_df))

        test_path = os.path.join(data_dir, f'test_group_{prop}.csv')
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

        datasets[prop] = {'folds': folds, 'test': test_df}
        print(f"  {prop}: {len(folds)} folds, test={len(test_df) if test_df is not None else 0}")

    return datasets


def df_to_features(df, vocab, property_name):
    """Convert a DataFrame to raw feature arrays (before scaling).

    For CO2 and viscosity, appends raw T and P as extra features.
    Returns:
        X: feature array, shape (N, dim_cat+dim_an [+2 for T,P])
        y: target values, shape (N,)
    """
    target_col = TARGET_COLUMNS[property_name]
    has_env = property_name in ['co2', 'vis']

    X_list, y_list = [], []
    for _, row in df.iterrows():
        cat_fp, an_fp, concat_fp = vocab.get_il_fingerprint(
            str(row['new_cation']).strip(),
            str(row['new_anion']).strip()
        )
        if concat_fp is None:
            continue
        if has_env:
            env = np.array([float(row['T']), float(row['P'])], dtype=np.float64)
            concat_fp = np.concatenate([concat_fp, env])
        X_list.append(concat_fp)
        y_list.append(float(row[target_col]))

    if len(X_list) == 0:
        total_dim = (vocab.cat_fp_length or 0) + (vocab.an_fp_length or 0)
        if has_env:
            total_dim += 2
        return np.empty((0, total_dim)), np.empty(0)

    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def get_non_test_dataframe(datasets, property_name, verify_folds=True):
    '''Return the complete train+validation pool for final model refitting.

    Each predefined fold partitions the same non-test pool. Fold 1 is used as
    the canonical serialization of that pool; the optional check prevents a
    silently inconsistent set of split files from changing the refit data.
    '''
    folds = datasets[property_name]['folds']
    if not folds:
        raise ValueError(f'No cross-validation folds found for {property_name!r}.')

    complete = pd.concat([folds[0][0], folds[0][1]], ignore_index=True)
    if verify_folds:
        target_col = TARGET_COLUMNS[property_name]
        id_cols = ['ind', 'new_cation', 'new_anion', target_col]
        if property_name in ('co2', 'vis'):
            id_cols.extend(['T', 'P'])

        def signatures(frame):
            available = [column for column in id_cols if column in frame.columns]
            return set(map(tuple, frame[available].astype(str).to_numpy()))

        expected = signatures(complete)
        for fold_i, (train_df, val_df) in enumerate(folds[1:], start=2):
            observed = signatures(pd.concat([train_df, val_df], ignore_index=True))
            if observed != expected:
                raise ValueError(
                    f'Fold {fold_i} does not contain the same non-test records '
                    f'as fold 1 for {property_name!r}.'
                )
    return complete


def prepare_cv_splits(datasets, vocab, property_name):
    """Prepare predefined CV splits for GP-BT optimizer.

    Applies StandardScaler fitted on each fold's training data to ensure
    proper scaling for GP. After hyperparameter selection, the final scaler
    and GP conditioning set use the complete train+validation (non-test) pool.

    Returns:
        cv_splits: list of (X_train, y_train, X_val, y_val) tuples (all scaled)
        X_all_train: complete non-test pool (scaled)
        y_all_train: complete non-test targets
        X_test, y_test: held-out test set (scaled with the refit scaler)
        scaler: fitted StandardScaler from the complete non-test pool
    """
    from sklearn.preprocessing import StandardScaler

    folds = datasets[property_name]['folds']
    test_df = datasets[property_name]['test']

    # First pass: get raw features for all folds
    raw_folds = []
    for fold_i, (train_df, val_df) in enumerate(folds):
        X_train, y_train = df_to_features(train_df, vocab, property_name)
        X_val, y_val = df_to_features(val_df, vocab, property_name)
        raw_folds.append((X_train, y_train, X_val, y_val))
        print(f"    Fold {fold_i+1}: train={len(y_train)}, val={len(y_val)}")

    # Scale each fold: fit scaler on fold's train, transform both train and val
    cv_splits = []
    for X_train, y_train, X_val, y_val in raw_folds:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        cv_splits.append((X_train_s, y_train, X_val_s, y_val))

    # Refit on every non-test record after hyperparameter selection.
    non_test_df = get_non_test_dataframe(datasets, property_name)
    X_refit_raw, y_refit = df_to_features(non_test_df, vocab, property_name)

    # Fit the screening/test scaler on the complete non-test pool.
    refit_scaler = StandardScaler()
    X_refit = refit_scaler.fit_transform(X_refit_raw)

    X_test, y_test = None, None
    if test_df is not None:
        X_test_raw, y_test = df_to_features(test_df, vocab, property_name)
        X_test = refit_scaler.transform(X_test_raw)
        print(f"    Test: {len(y_test)}")

    print(f"    All train (refit): {len(y_refit)}")

    return cv_splits, X_refit, y_refit, X_test, y_test, refit_scaler
