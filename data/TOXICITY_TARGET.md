# IPC-81 toxicity target convention

Source: Wang, Z.; Song, Z.; Zhou, T. Machine Learning for Ionic Liquid
Toxicity Prediction. Processes 2021, 9(1), 65.
https://doi.org/10.3390/pr9010065

The following primary materials were checked:

- Main article, Section 2, Experimental Data: IPC-81 cytotoxicity is reported
  as the logarithm of the half-maximal effective concentration (logEC50).
- [Official supplementary workbook](https://mdpi-res.com/d_attachment/processes/processes-09-00065/article_deploy/processes-09-00065-s001.zip),
  processes-1052496-supplementary.xlsx: Table S1 contains 355 records and
  labels the target column Experimental logEC50. No explicit concentration
  unit or logarithm base was found in the workbook.
- [Author repository data documentation](https://github.com/zwang1995/IL-Toxicity/blob/main/data/README.md):
  the logEC50 field is identified as the IPC-81 toxicity value, without an
  explicit base or concentration unit.

CF-BILD retains the source notation and the supplied logarithmic numbers.
It does not infer ln, log10, molar, mass-based, or other concentration units,
and does not back-transform this endpoint to an absolute concentration.
Larger values indicate lower cytotoxicity on this source-reported scale.
This documentation does not claim that EC50 physically has no units.

The current property inputs are reused from Zhong et al., EST Letters 2024,
11, 1193-1199 (https://doi.org/10.1021/acs.estlett.4c00524), with Wang et al.
as the toxicity compilation source. No target, split, model, or ranking was
changed in this source-convention and typesetting update.
