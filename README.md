# CF-BILD — Digital Discovery major-revision compendium

This repository contains the reproducible software and author-generated
artifacts for:

> Fragment-Constrained Ionic Liquid Screening with a Compositional Gaussian
> Process Surrogate and an Uncertainty-Penalized Multi-Objective Acquisition
> Function (Digital Discovery, DD-ART-06-2026-000373).

The September 2026 revision supersedes the earlier code-only snapshot.

## What is implemented

1. A deterministic 505-cation x 173-anion vocabulary extracted from the three
   property datasets; PubChem is not part of the CF-BILD vocabulary.
2. The paper-specific GPyTorch compositional-GP implementation with predefined
   species-level splits and product/additive/no-cross kernels.
3. Final refitting on every non-test record after cross-validation selects
   hyperparameters.
4. A zero-truncated Gaussian predictive distribution for non-negative CO2
   capacity.
5. Feasibility-weighted additive expected improvement (FW-AEI), explicitly not
   EHVI, plus analytical q=1 EHVI against a non-empty incumbent Pareto front.
6. Corrected independent two- and three-objective screens, a median-distance
   baseline, GRU similarity audit, and SMARTS stability triage.

## Authoritative layout

    cf_bild/
      fragment_vocab.py       vocabulary, features, predefined splits
      gp_cvloss.py            paper-specific compositional GP
      predictive.py           calibrated physical predictions
      acquisition.py          FW-AEI, q=1 EHVI, hypervolume
    scripts/revision/          authoritative analysis and figure scripts
    data/                      provenance and restricted-data hash manifest
    output/revision_2026/      public prediction caches and result tables
    figures/                   600 dpi PNG and vector PDF figures
    tests/test_revision_methods.py

Legacy scripts with invalid acquisition labels are absent from this release.

## Environment

The validated environment is Python 3.10.8 with versions in
`requirements-revision-lock.txt`. CF-MF is pinned to commit
`698c31559e71f9cb14fd58e56562511ae644fc40`.

Windows PowerShell:

    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-revision-lock.txt

CPU-only PyTorch may be substituted for tests and cached-result analyses.

## Public reproduction path

These commands use only distributed author-generated artifacts and do not
require record-level experimental tables:

    python tests/test_revision_methods.py
    python scripts/revision/run_acquisition_analysis.py
    python scripts/revision/run_gru_ood_similarity.py
    python scripts/revision/run_stability_screening.py
    python scripts/revision/build_revision_figures.py

The acquisition script consumes the full-pool posterior cache, aggregated
training-derived operating definitions, and model-derived incumbent Pareto
coordinates. It never substitutes candidate extrema for the stated training
definitions.

## Full-refit path with authorized local data

The record-level property tables originate from NIST ILThermo and the cited
toxicity literature. The authors do not have permission to redistribute them.
Obtain the records from the original providers under their terms, place the 33
expected split files in `data/`, and verify the local copy:

    python scripts/revision/verify_restricted_data.py --data-dir data
    python scripts/revision/run_revision_models.py
    python scripts/revision/run_median_baseline.py

`data/restricted_data_manifest.json` contains no experimental values; it gives
filenames, schemas, row counts, sizes, and SHA-256 digests. Checkpoints and test
prediction files embedding source-derived targets are also excluded. See
`data/DATA_ACCESS.md` for provenance and the public/restricted boundary.

## Expected headline outputs

| Item | Revised value |
|---|---:|
| Candidate pool | 87,365 |
| Final CO2 test R2 / RMSE | 0.909 / 0.066 |
| Final viscosity test R2 / RMSE (ln space) | 0.855 / 0.614 |
| Final toxicity test R2 / RMSE | 0.755 / 0.445 |
| FW-AEI top-100 hypervolume / mean sigma | 34.041 / 1.334 |
| Analytical q=1 EHVI hypervolume / mean sigma | 38.453 / 1.401 |
| Corrected stability classes (Pass/Caution/Fail) | 36 / 5 / 59 |

Full-precision aggregate values are in `output/revision_2026/`.

## Citation, archive, and license

- Canonical repository: https://github.com/Shifa-Zhong/CF-BILD-
- Release tag: `v0.2.0`
- Archived software version DOI: `[ZENODO_VERSION_DOI]`
- Software concept/latest DOI: `[ZENODO_CONCEPT_DOI]`

No source-derived dataset is redistributed. Replace the DOI placeholders after
Zenodo archival. The software is released under the MIT License; upstream data
remain governed by their providers' terms.
