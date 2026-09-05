# Public data access and provenance

The exact model inputs are now publicly deposited with author authorization.
The former no-redistribution description applies only to the historical v0.2.2
snapshot, not this data release.

## Property data

Direct source: Zhong et al., Environmental Science & Technology Letters 2024,
11, 1193–1199, https://doi.org/10.1021/acs.estlett.4c00524.

The source study's experimental-data workbook is archived as
[ez4c00524_si_002.xlsx](https://doi.org/10.1021/acs.estlett.4c00524.s002).
Its ACS Figshare record (article 26976310) specifies **CC BY-NC 4.0**.
This repository distributes the project's processed inputs, not an assertion
that every encoded row is a verbatim copy of the source workbook.

Upstream: NIST ILThermo (SRD 147), https://ilthermo.boulder.nist.gov/, for CO2
and viscosity; Wang et al., Processes 2021, 9, 65,
https://doi.org/10.3390/pr9010065, for IPC-81 cytotoxicity.

All 33 train/validation/test CSVs are supplied unchanged. Fold-1 train plus
validation defines the complete non-test pool; the other four fold pairs
contain the same records. The files retain local record indices and encoded
species groups. Original raw-database extraction/curation is not recreated:
these exact deposited inputs are the reproducible starting point.

| Property | Non-test records | Test records | Target |
|---|---:|---:|---|
| CO2 | 12503 | 1306 | dissolved CO2 mole fraction |
| Viscosity | 12374 | 1465 | ln(viscosity / (Pa s)) |
| Toxicity | 302 | 32 | supplied IPC-81 logEC50 |

Toxicity retains the source dataset's logEC50 notation and numerical scale.
The main article, official Table S1, and author data README do not explicitly
state a logarithm base or concentration unit. No unit is inferred and no
additional logarithm or concentration conversion is applied. See
[TOXICITY_TARGET.md](TOXICITY_TARGET.md) for the checked sources.

## Exact-input and split checks

Run python scripts/revision/verify_public_data.py. PUBLIC_DATA_RELEASE.json
contains SHA-256 hashes for data and source-bearing model/test artifacts.

Known limitation: viscosity group CC[n+]1ccccc1.CCOS(=O)(=O)[O-] crosses the
train/validation boundary in folds 3 and 5 (19 records). There is no observed
non-test/test encoded-group overlap. The audit and release preserve this
limitation rather than changing historical partitions. Charge and
stoichiometry checks are separate from these encoded-group checks.

The three output/revision_2026/models/model_*.pkl files contain model parameter
and scaler states, with test arrays. Reconstructing a GP posterior also
requires the provided conditioning records and vocabulary. Do not load pickle
from untrusted sources; the verification program does not unpickle anything.

## PubChem GRU corpus

The separate corpus is stored in data/gru/pubchem_corpus_portable as 109 gzip parts.
python scripts/revision/restore_pubchem_corpus.py restores the exact source
to data/Pubchem dataset.txt. --verify-only checks restoration without writing.
The manifest specifies part order, compressed hashes, restored size and hash.

PubChem corpus and GRU source/sample are separate from the three property
datasets reused from Zhong et al. The fixed sample in data/gru/generate_result.csv
is the reproducible input to the reported support comparison.

Software is MIT-licensed. The property-data source workbook carries the
separate CC BY-NC 4.0 notice above; retain attribution and applicable upstream
terms for reused data. That notice is not a license for the separate PubChem
corpus. Public access does not imply chemical validation.
