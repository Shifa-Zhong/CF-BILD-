"""Verify raw preservation, structure-only curation and species split integrity."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cf_bild.ion_validation import check_ion_pair


def main(run):
    run = Path(run).resolve()
    report = json.loads((run / 'CURATION_SUMMARY.json').read_text())
    checks = []
    def check(condition, message):
        if not condition: raise AssertionError(message)
        checks.append(message)
    def read(path): return pd.read_csv(path, dtype=str, keep_default_na=False)
    for folder, key in [('raw', 'source_sha256'), ('data', 'cleaned_sha256')]:
        for name, checksum in report[key].items():
            check(hashlib.sha256((run / folder / name).read_bytes()).hexdigest() == checksum,
                  f'{folder}/{name}: exact recorded bytes')
    for prop in report['properties']:
        original = pd.concat([read(run / 'raw' / f'{r}_1_group_{prop}.csv') for r in ('train', 'val')], ignore_index=True).set_index('ind')
        old_test = read(run / 'raw' / f'test_group_{prop}.csv').set_index('ind')
        test = read(run / 'data' / f'test_group_{prop}.csv')
        validation_counts = Counter()
        base_ids = None
        for k in range(1, 6):
            tr = read(run / 'data' / f'train_{k}_group_{prop}.csv')
            va = read(run / 'data' / f'val_{k}_group_{prop}.csv')
            check(not (set(tr.group) & set(va.group)), f'{prop} fold {k}: species-disjoint training/validation')
            check(not ((set(tr.group) | set(va.group)) & set(test.group)), f'{prop} fold {k}: no held-out test species')
            ids = Counter([*tr.ind, *va.ind])
            if base_ids is None: base_ids = ids
            check(ids == base_ids and all(n == 1 for n in ids.values()), f'{prop} fold {k}: identical non-test pool, no duplicated IDs')
            validation_counts.update(va.ind)
            for role, frame in [('training', tr), ('validation', va)]:
                check(all(check_ion_pair(c, a).valid for c, a in zip(frame.new_cation, frame.new_anion)),
                      f'{prop} fold {k} {role}: valid +1/-1 connected components')
                check(all(all(row[c] == original.loc[row['ind'], c] for c in frame.columns if c not in ('ind', 'group'))
                          for row in frame.to_dict('records')),
                      f'{prop} fold {k} {role}: original measurement and structure tokens retained')
        check(validation_counts == base_ids, f'{prop}: every retained record validates exactly once')
        check(all(check_ion_pair(c, a).valid for c, a in zip(test.new_cation, test.new_anion)), f'{prop}: valid held-out structures')
        check(all(all(row[c] == old_test.loc[row['ind'], c] for c in test.columns if c not in ('ind', 'group'))
                  for row in test.to_dict('records')), f'{prop}: test not resampled and measurement tokens unchanged')
        for role, source, kept in [('non_test', original, set(base_ids)), ('test', old_test, set(test.ind))]:
            expected = {idx for idx, row in source.iterrows() if check_ion_pair(row.new_cation, row.new_anion).valid}
            check(expected == kept, f'{prop} {role}: exact application of structure-only exclusion policy')
    output = {'status': 'passed', 'n_checks': len(checks), 'checks': checks,
              'curation_summary_sha256': hashlib.sha256((run / 'CURATION_SUMMARY.json').read_bytes()).hexdigest()}
    (run / 'CLEAN_INPUT_VERIFICATION.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'PASS: {len(checks)} cleaned-input integrity checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    main(parser.parse_args().run_directory)
