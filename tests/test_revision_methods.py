'''Public-release numerical and input-pool test runner (no pickle loading).'''

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    'revision_full_tests', HERE / 'test_revision_methods_full.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    tests = [
        MODULE.test_zero_truncated_moments_match_scipy,
        MODULE.test_co2_prediction_is_strictly_nonnegative,
        MODULE.test_analytic_ehvi_matches_vectorized_monte_carlo,
        MODULE.test_fw_aei_is_finite_and_nonnegative,
        MODULE.test_revision_cache_contract,
        MODULE.test_complete_non_test_pool_counts_and_fold_identity,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print('For the separate species-boundary audit, run scripts/revision/verify_public_data.py --strict-splits.')


if __name__ == '__main__':
    main()
