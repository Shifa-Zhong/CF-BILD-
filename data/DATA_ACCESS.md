# Property inputs and derived curation

The authoritative run is `runs/ion_clean_refit_2026-09-05/`:

- `raw/`: byte-preserved 33 pre-split source CSV files.
- `data/`: 33 retained-data CSVs with target-independent species grouping.
- `record_curation_audit.csv`: every source record, decision and reason.
- `excluded_records.csv`: the explicitly excluded records.
- `fold_membership_changes.csv`: nine distinct viscosity records whose
  validation fold changes, represented as 18 train/validation membership rows.
- `CURATION_SUMMARY.json`: exact raw/derived hashes and partition counts.
- `CLEAN_INPUT_VERIFICATION.json`: 186 independent integrity and scope checks.

Columns retain original source IDs (`ind`), constituent SMILES (`new_cation`,
`new_anion`), measurement conditions when present, and source target values.
`group` identifies the canonical cation/anion pair. No target values are
imputed or altered during the structure curation. The test set is not resampled.

The data were reused from Zhong et al., Environmental Science & Technology
Letters 2024, 11, 1193–1199,
[10.1021/acs.estlett.4c00524](https://doi.org/10.1021/acs.estlett.4c00524).
The [published source workbook](https://doi.org/10.1021/acs.estlett.4c00524.s002)
has a CC BY-NC 4.0 notice on ACS Figshare. Upstream property sources are NIST
ILThermo and Wang et al., Processes 2021, 9, 65,
[10.3390/pr9010065](https://doi.org/10.3390/pr9010065).

The authors authorized data publication and the stated derived curation.
MIT software licensing does not supersede upstream data terms. No new archive
DOI has been issued; final-version Zenodo archiving is deferred until freezing.
The separate PubChem corpus is under `data/gru/`, not attributed to the property
compilation and not used to construct the CF-BILD vocabulary.
