# CF-BILD — public data and analysis compendium

For *Fragment-Constrained Ionic Liquid Screening with a Compositional Gaussian
Process Surrogate and an Uncertainty-Aware Multi-Objective Acquisition Function*
(Digital Discovery manuscript DD-ART-06-2026-000373).

## Data release and scientific status

The authors authorized public data release on 5 September 2026. This checkout
now includes the exact 33 property input CSVs (five training/validation folds
and one test for each property), test-prediction tables, saved GP parameter/
scaler states, vocabulary, screening caches, and the PubChem GRU corpus.

This deposit preserves the supplied SMILES, target values, partitions, and
numerical outputs. It does **not** certify all encoded combinations as
charge-balanced salts. Formal-charge/stoichiometry review remains open, and
existing screening classifications are provisional pending that review.

The split audit also identifies one viscosity group crossing training and
validation in folds 3 and 5 (19 records). No non-test/test encoded-group overlap
was found. This limitation is disclosed rather than silently changing the
partitions or claiming strictly species-disjoint CV in every fold.

## Data provenance

The three property datasets are reused from:

Shifa Zhong et al., *Screening Environmentally Benign Ionic Liquids for CO2
Absorption Using Representation Uncertainty-Based Machine Learning*,
Environmental Science & Technology Letters **2024**, 11, 1193–1199.
https://doi.org/10.1021/acs.estlett.4c00524

The study's [experimental-data workbook](https://doi.org/10.1021/acs.estlett.4c00524.s002)
is archived by ACS Figshare with a CC BY-NC 4.0 notice. These are processed
property inputs; the software's MIT license does not replace that data notice.

That study collected CO2 and viscosity records from NIST ILThermo and obtained
IPC-81 cytotoxicity data from Wang, Song, and Zhou, Processes **2021**, 9, 65,
https://doi.org/10.3390/pr9010065. Toxicity is **not** a Vibrio fischeri
bioluminescence endpoint. The supplied logEC50 numbers are unchanged. The
concentration unit and logarithm base require source confirmation; do not
exponentiate or relabel them as ln values without that evidence.

PubChem is the source of the separate GRU corpus, not of the CF-BILD ion
vocabulary and not a dataset attributed to the EST Letters study.

See [data/DATA_ACCESS.md](data/DATA_ACCESS.md) and
[data/PUBLIC_DATA_RELEASE.json](data/PUBLIC_DATA_RELEASE.json).

## Environment

The reported calculations used Python 3.10.8 and the versions in
requirements-revision-lock.txt. CF-MF is pinned to commit
698c31559e71f9cb14fd58e56562511ae644fc40.

    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-revision-lock.txt

CPU PyTorch is sufficient for integrity tests and cached-result plotting.

## Verify the exact public inputs

    python scripts/revision/verify_public_data.py
    python scripts/revision/restore_pubchem_corpus.py --verify-only
    python tests/test_revision_methods.py

The first command checks hashes, pool identity across folds, and group overlap.
It explicitly reports the known viscosity CV warning. Use --strict-splits to
make any training/validation overlap a failure. Passing byte-integrity tests
does not certify chemical identity or remove the CV warning.

## Reproduce figures and downstream analyses

    python scripts/revision/build_revision_figures.py
    python scripts/revision/run_acquisition_analysis.py
    python scripts/revision/run_gru_ood_similarity.py
    python scripts/revision/run_stability_screening.py

Figure 3 can now be rebuilt from the distributed test-prediction/experimental
tables; no private inputs are needed for the figure command. Figure 4b shows
each property's uncertainty as a ratio to the corresponding EHVI mean,
without averaging quantities on different target scales.

The downstream commands reproduce the archived encoded-input analysis.
They are not a new chemical validation. Existing numerical headline values
remain in output/revision_2026/ and are provisional under the issues above.

## Refit and GRU training inputs

    python scripts/revision/run_revision_models.py
    python scripts/revision/run_median_baseline.py
    python scripts/revision/restore_pubchem_corpus.py

The first two commands are computationally intensive and overwrite generated
analysis outputs in the checkout; use a separate clone for new runs. They
were **not rerun** as part of the public-data/editorial update. Fixed selected
hyperparameters remain in config/selected_hyperparameters.json.

Saved model_*.pkl files contain parameter/scaler states and test arrays; they
are not standalone GPs without conditioning data. The exact conditioning
records and vocabulary are supplied. Pickle files can execute code: load only
trusted artifacts after checking their published hashes. Integrity tests use
CSV, JSON and NPZ and do not execute pickle.

The restored PubChem corpus is byte-identical to the archived 453,620,552-byte
source. See generative_baseline/README.md for the training interface. The fixed
95,285-pair generated sample defines the reported 20,000-pair similarity audit;
fresh stochastic training is not claimed to reproduce identical strings.

## Citation, archive, and license

Canonical repository: https://github.com/Shifa-Zhong/CF-BILD-

The v0.2.2 tag remains immutable historical material. Cite the commit of this
public-data release when using these newly deposited inputs. Persistent
archival software-version, concept, and dataset DOIs are pending and will be
supplied by acceptance.

Software is MIT-licensed. Cite the direct and upstream dataset sources above;
software licensing does not replace applicable data-provider terms.
