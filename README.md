# CF-BILD — curated 1:1 ion-pair screening compendium

For Digital Discovery manuscript DD-ART-06-2026-000373, *Fragment-Constrained
Ionic Liquid Screening with a Compositional Gaussian Process Surrogate and
an Uncertainty-Aware Multi-Objective Acquisition Function*.

## Authoritative analysis and scope

The current analysis is in `runs/ion_clean_refit_2026-09-05/`. It supersedes
the pre-cleaning results at historical commit
`2f24b28f69a19b60cfcaf7894f54cca4521df9a1`; old commits and tags remain unchanged.
Use this run and the commands below, not the superseded empty-front EHVI or
development-subset scripts from older releases.

Only parseable, connected +1 cations and −1 anions are retained for the 1:1
model. Neutral/wrong-role encodings and 14 multivalent viscosity species
(195 records, outside verified 1:1 stoichiometry) are excluded, not repaired
by assumption. Exclusion does not establish that an experimental material
cannot exist. Raw files and every record-level decision are retained.

| Property | Non-test records | Test records | Non-test/test species |
| --- | ---: | ---: | ---: |
| CO₂ mole fraction | 10,867 | 1,080 | 158 / 17 |
| ln[viscosity/(Pa s)] | 11,101 | 1,344 | 683 / 80 |
| Source log EC₅₀ (IPC-81) | 300 | 31 | 278 / 31 |

Test membership is inherited except for structure-only exclusions. Within
the non-test pool, each canonical species takes its modal original validation
fold (lowest-fold tie breaking). Every retained record validates exactly
once; all five folds and non-test/test boundaries are species-disjoint.
Raw inputs, retained inputs, exclusions, and fold-membership changes are
separate artifacts in the run directory.

The fixed vocabulary contains 444 cations and 152 anions (67,488 formal
charge-balanced pairs). It includes all retained partition structures but
never uses test targets for training, calibration, or score definitions.
CF-MF uses radius-2 **environment counts**, not binary bits: 1,242 + 828 =
2,070 structural columns. T/P add two inputs. Descriptive ion-family labels
and separate t-SNE coordinates do not enter the GP or stability rules.

## Provenance and rights

The three property datasets are reused from Zhong et al., *Screening
Environmentally Benign Ionic Liquids for CO2 Absorption Using Representation
Uncertainty-Based Machine Learning*, Environmental Science & Technology
Letters 2024, 11, 1193–1199,
[DOI 10.1021/acs.estlett.4c00524](https://doi.org/10.1021/acs.estlett.4c00524).
The [source workbook](https://doi.org/10.1021/acs.estlett.4c00524.s002) has an
ACS Figshare CC BY-NC 4.0 notice. Upstream sources are NIST ILThermo (SRD 147)
and Wang, Song, and Zhou, Processes 2021, 9, 65,
[DOI 10.3390/pr9010065](https://doi.org/10.3390/pr9010065).

Toxicity is IPC-81 cytotoxicity, not a bacterial bioluminescence endpoint.
The source logEC50 values and notation are unchanged; no unsupported unit
or log-base conversion is applied. See `data/TOXICITY_TARGET.md`.
PubChem is the separate GRU training source, not the CF-BILD vocabulary source.
MIT applies to software, not as a replacement for upstream data terms.

## Environment

Recorded versions are in `requirements-revision-lock.txt`,
`environment-revision.yml`, and the run's `COMPUTATIONAL_ENVIRONMENT.json`.
The reported GPU is an NVIDIA GeForce RTX 5090; Python is 3.10.8.
CF-MF is pinned to commit `698c31559e71f9cb14fd58e56562511ae644fc40`.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-revision-lock.txt
```

CPU environments can verify file integrity and cached numerical outputs.
Fresh GP fitting is computationally intensive, and hardware-dependent
floating-point differences are possible even with pinned software.

## Verification (no retuning)

```powershell
python scripts/revision/verify_core_release.py
python scripts/revision/verify_clean_inputs.py --run-directory runs/ion_clean_refit_2026-09-05
python tests/test_core_release.py
python scripts/revision/restore_pubchem_corpus.py --verify-only
python scripts/revision/verify_saved_models.py --run-directory runs/ion_clean_refit_2026-09-05
```

The clean-input verifier performs 186 byte/value/charge/split checks. The last
command loads trusted pickle states after hash checks and reconstructs the
primary GP test predictions. **Pickle can execute code**: run it only on
trusted release artifacts. Hashes establish identity, not trust in the source.

GPyTorch `fast_pred_var` uses a [Lanczos variance approximation](https://docs.gpytorch.ai/en/stable/settings.html#gpytorch.settings.fast_pred_var).
Kernel/scaler parameters alone omit its numerical variance cache. The saved
model verifier restores the final search checkpoint's CPU/CUDA RNG states,
replays the five calibration folds, and conditions on all non-test records.
It does not rerun Hyperopt or use test targets to choose parameters. This
reproduces the archived primary test means and standard deviations at CSV
floating-point precision in the recorded environment. Settings and limitations
are documented in SI Text S10 and `analysis/SAVED_MODEL_RECONSTRUCTION.json`.

## Recompute the archived numerical analyses

Run these in a separate copy if retaining byte-identical archived analysis
files is important: these commands regenerate named output artifacts.

```powershell
python scripts/revision/collect_clean_results.py --run-directory runs/ion_clean_refit_2026-09-05
python scripts/revision/run_acquisition_analysis.py --data-directory runs/ion_clean_refit_2026-09-05/data --output-directory runs/ion_clean_refit_2026-09-05/analysis
python scripts/revision/run_stability_screening.py --output-directory runs/ion_clean_refit_2026-09-05/analysis
python scripts/revision/run_clean_diagnostics.py --run-directory runs/ion_clean_refit_2026-09-05
python scripts/revision/run_clean_prediction_diagnostics.py --run-directory runs/ion_clean_refit_2026-09-05
python scripts/revision/run_gru_ood_similarity.py --data-directory runs/ion_clean_refit_2026-09-05/data --output-directory runs/ion_clean_refit_2026-09-05/analysis --generated-file data/gru/generate_result.csv
python scripts/revision/summarize_clean_comparisons.py --run-directory runs/ion_clean_refit_2026-09-05
```

`cf_bild/acquisition.py` implements FW-AEI and non-empty-front analytical
q = 1 EHVI. The benchmark uses a model-imputed incumbent front and independent
marginals, including lower-truncated CO₂. It is not joint batch qEHVI, and
predicted-set hypervolume is not measured performance. FW-AEI gives a
property-dependent trade-off, not uniformly lower uncertainty. Stability
Pass/Caution/Fail labels are qualitative triage signals.

## Fresh, isolated end-to-end run

The destination must not exist when preparing new inputs. All three properties
must be curated before freezing their shared vocabulary. No old fitted
parameters or candidate caches are used in a new run.

```powershell
python scripts/revision/prepare_clean_refit.py --source-directory runs/ion_clean_refit_2026-09-05/raw --output-directory runs/new_run --exclude-unverified-stoichiometry
python scripts/revision/verify_clean_inputs.py --run-directory runs/new_run
python scripts/revision/run_clean_refit.py --run-directory runs/new_run --kernel-forms product product_no_cross additive additive_no_cross standard --properties tox co2 vis --max-evals 3000 --patience 50 --seed 42
python scripts/revision/collect_clean_results.py --run-directory runs/new_run
python scripts/revision/run_median_baseline.py --data-directory runs/new_run/data --vocabulary-path runs/new_run/fragment_vocab.pkl --selected-directory runs/new_run/results/product --output-directory runs/new_run/analysis
```

Then use the analysis commands above with `runs/new_run`. Each model has a
checkpointed five-fold search, complete fold-residual calibration, and a
full non-test fit. Searches stop after 50 trials without improvement or at
3,000 trials. Interrupted searches resume only when data, code, vocabulary,
and protocol hashes agree. Completed hash-verified fits are not overwritten.
`run_revision_models.py` is retained as a helper-module dependency of the
new runner; its historical main entry point is not the current protocol.

For the GRU, restore the losslessly compressed 453,620,552-byte corpus with
`python scripts/revision/restore_pubchem_corpus.py`; see
`generative_baseline/README.md` for training and sampling. The archived
95,285-pair output defines the fixed 20,000-pair support comparison. Fresh
stochastic training is not claimed to regenerate identical SMILES.

## Additional model and frozen-ranking diagnostics

CF-BILD denotes Compositional Fragment-Based Ionic Liquid Design. The
18 tuned fits comprise the 15 original architecture/property fits and three
new low-parameter single-structural-kernel comparators; three median-distance
baselines are supplied separately. The primary models and screening list
remain frozen. The additional comparator uses one shared structural scale,
the same separate environmental-kernel construction, three/five continuous
parameters, and unchanged CV/search/calibration/full-pool fitting procedures.
It is not exactly parameter-count matched to the compositional model.
Its held-out R2 values are 0.910, 0.885 and 0.806, so compositional structure
is not claimed necessary for accurate prediction on these data.

The additions are stored separately under:

- `runs/ion_clean_refit_2026-09-05/extensions/low_parameter_2026-09-06/`
- `runs/ion_clean_refit_2026-09-05/extensions/ranking_diagnostics_2026-09-06/`

Manuscript Table 1 combines the original six rows in
`analysis/model_comparison_clean.csv` with the three property metric records
`extensions/low_parameter_2026-09-06/metrics_<property>.json` for the shared-scale
row. The original CSV is preserved; the extension records supply the new row.

Reproduction commands (use a separate copy when regenerating audit/timing files):

```powershell
python scripts/revision/run_shared_kernel_baseline.py --run-directory runs/ion_clean_refit_2026-09-05
python scripts/revision/run_shared_kernel_baseline.py --run-directory runs/ion_clean_refit_2026-09-05 --verify-only
python scripts/revision/run_ranking_diagnostics.py --run-directory runs/ion_clean_refit_2026-09-05 --timing-repeats 3
```

The baseline command resumes only hash-compatible searches and skips verified
completed fits. The verifier loads trusted pickle states after integrity checks;
never load untrusted pickle files. It reconstructs all three saved comparator
test predictions without retuning. Use the same commands with `runs/new_run`
after the fresh-run procedure above for isolated end-to-end recomputation.

The ranking diagnostics document positive-weight/scale sensitivity, marginal
score shares, modeled threshold probabilities, residual upper-support tails,
and scoring times with one warm-up plus three order-rotated repetitions.
Halving/doubling one weight preserves 98–100 top-100 members; non-test IQR
scaling preserves 97, and CO2 percentage units preserve 90. This is local set
stability, not scale invariance. Mean joint probabilities are 1.570% (FW-AEI),
0.319% (additive EI) and 1.256% (EHVI), all low in absolute terms. These are
model-defined preference probabilities, not validated process success rates.
Arithmetic EI shares do not measure chemical importance. FW-AEI is a heuristic
product of expectations, not a conditional improvement identity for constraints
on the same uncertain properties. Timings compare these implementations and
exclude model fitting/prediction; they are not universal speed guarantees.

## Release boundary and archiving

This repository contains core scientific code, data, model/search artifacts,
numerical figure/table inputs, environments, and tests. Manuscript/response
generation, Word typesetting, plotting scripts, and publication graphics are
maintained separately and are intentionally not uploaded.

Canonical repository: https://github.com/Shifa-Zhong/CF-BILD-.
Zenodo software-version/concept and dataset DOIs are deferred until the
final version is frozen and will be provided by acceptance. No DOI is invented
or implied by this provisional archival status.
