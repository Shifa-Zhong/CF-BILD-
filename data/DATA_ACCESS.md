# Data provenance, access, and redistribution

The CO2-equilibrium and viscosity records used in this study were curated from
NIST ILThermo (SRD 147): <https://ilthermo.boulder.nist.gov/>. The toxicity
records were curated from the *Vibrio fischeri* subset reported by Zhao et al.,
*Journal of Hazardous Materials* 278 (2014) 320-329,
<https://doi.org/10.1016/j.jhazmat.2014.06.018>.

The authors do **not** have permission to redistribute the source-derived
record-level tables. The public GitHub/Zenodo software release therefore
intentionally excludes all train, validation, and test CSV files. Researchers
must obtain the underlying records from the original providers under their
applicable terms.

`restricted_data_manifest.json` publishes no experimental values. It records
the expected filenames, column schemas, row counts, byte sizes, and SHA-256
digests so an authorized local copy can be checked exactly:

    python scripts/revision/verify_restricted_data.py --data-dir data

The complete species-grouped split handling, final-refit, and prediction code
is public. Model refitting and the median-distance baseline require a verified
local copy of the restricted tables. Fixed hyperparameters and aggregated
metrics are public; checkpoints and test-prediction files containing
source-derived target values are excluded.

The public archive includes author-generated artifacts that do not contain
record-level experimental values: the 505-cation x 173-anion vocabulary and
CF-MF representation, full-pool posterior prediction cache, acquisition and
stability outputs, aggregated model metrics, figures, tests, and environment
files.

PubChem contributed only to the separate GRU training corpus, not to the
CF-BILD vocabulary. The approximately 454 MB corpus is also excluded because
the authors do not have permission to redistribute it. The fixed generated
sample and its similarity output are included, so retraining the GRU is not
required to reproduce the reported comparison.
