"""Regression tests for structure-only curation and whole-species CV repair."""
import sys
import tempfile
import unittest
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/revision'))
from cf_bild.ion_validation import check_ion_pair
from prepare_clean_refit import canonical_group, validation_assignments
from run_clean_refit import optimize_checkpointed
from cf_bild.gp_cvloss import GPCrossValidatedOptimizer


class CurationTests(unittest.TestCase):
    def test_monovalent_pair(self):
        self.assertTrue(check_ion_pair('C[N+](C)(C)C', '[Cl-]').valid)

    def test_neutral_is_not_an_anion(self):
        result = check_ion_pair('C[N+](C)(C)C', 'CC(=O)O')
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, 'neutral_component_or_incorrect_ion_role')

    def test_wrong_sign_is_not_swapped(self):
        self.assertFalse(check_ion_pair('[Cl-]', 'C[N+](C)(C)C').valid)

    def test_multivalent_is_separate_from_wrong_role(self):
        result = check_ion_pair('[Ca+2]', '[Cl-]')
        self.assertEqual(result.reason, 'multivalent_salt_requires_explicit_stoichiometry')

    def test_disconnected_and_unparseable(self):
        self.assertFalse(check_ion_pair('C[N+](C)(C)C.O', '[Cl-]').valid)
        self.assertFalse(check_ion_pair('not_a_smiles', '[Cl-]').valid)

    def test_canonical_group_collapses_equivalent_strings(self):
        self.assertEqual(canonical_group('C[N+](C)(C)CC', '[Cl-]'),
                         canonical_group('CC[N+](C)(C)C', '[Cl-]'))

    def test_majority_fold_and_tie_are_structure_only(self):
        rows = [{'ind': str(i), 'new_cation': 'C[N+](C)(C)C', 'new_anion': '[Cl-]', 'target': str(i)} for i in range(4)]
        frame = pd.DataFrame(rows)
        group = canonical_group(rows[0]['new_cation'], rows[0]['new_anion'])
        frame['group'] = group
        old = [frame.iloc[0:2], frame.iloc[2:4], frame.iloc[0:0], frame.iloc[0:0], frame.iloc[0:0]]
        assignment, _ = validation_assignments(frame, old)
        self.assertEqual(assignment[group], 1)
        frame['target'] = '999999'
        self.assertEqual(validation_assignments(frame, old)[0], assignment)
        old = [frame.iloc[0:1], frame.iloc[1:4], *old[2:]]
        self.assertEqual(validation_assignments(frame, old)[0][group], 2)

    def tiny_optimizer(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(16, 4))
        y = np.sin(x[:, 0]) + .1*x[:, 2]
        cv = [(x[:8], y[:8], x[8:], y[8:]), (x[8:], y[8:], x[:8], y[:8])]
        return GPCrossValidatedOptimizer(x, y, kernels=['RBF'], predefined_cv_splits=cv,
            compositional_kernel_dims=(2, 2), kernel_form='product', device=torch.device('cpu'), random_state=42)

    def test_checkpoint_resume_and_complete_calibration(self):
        with tempfile.TemporaryDirectory(prefix='cf_bild_test_') as tmp:
            directory = Path(tmp)
            model = self.tiny_optimizer()
            optimize_checkpointed(model, directory, {'test': 'synthetic-only'}, 2, 50, 42)
            self.assertEqual(model.calibration_diagnostics_['n_records'], 16)
            self.assertEqual(model.calibration_diagnostics_['n_folds'], 2)
            before = model.predict(np.zeros((2, 4)))
            resumed = self.tiny_optimizer()
            with patch.object(resumed, '_objective', side_effect=AssertionError('Should not add trials')):
                optimize_checkpointed(resumed, directory, {'test': 'synthetic-only'}, 2, 50, 42)
            self.assertEqual(len(resumed.trials), 2)
            after = resumed.predict(np.zeros((2, 4)))
            np.testing.assert_allclose(before[0], after[0], rtol=1e-5)
            np.testing.assert_allclose(before[1], after[1], rtol=1e-5)

    def test_incomplete_calibration_must_fail(self):
        model = self.tiny_optimizer()
        model.best_params = {'kernel_name': 'RBF', 'likelihood_noise_variance': .01}
        with patch.object(model, '_build_kernel', side_effect=ValueError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'every CV fold'):
                model._calibrate_variance()


if __name__ == '__main__':
    unittest.main()
