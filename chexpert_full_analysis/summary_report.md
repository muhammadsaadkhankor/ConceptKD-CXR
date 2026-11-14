# CheXpert five-disease analysis

## Scope

Analyzed **223414** training records. `SAMPLE_SIZE = None`; `None` denotes the full training CSV.
No image pixels were loaded and source dataset files were not modified. Only label value `1` is positive; `0`, `-1`, and missing values remain distinct.
All tables are consolidated in `chexpert_analysis_tables.xlsx`.

## Dataset statistics

| Metric | Value |
| --- | --- |
| Image records analyzed | 223414 |
| Unique patients from image paths | 64540 |
| Rows with unparseable/missing patient ID | 0 |
| SAMPLE_SIZE configuration | All records |

## Label distribution

| Disease | Positive (1) | Negative (0) | Uncertain (-1) | Missing (NaN) |
| --- | --- | --- | --- | --- |
| Atelectasis | 33376 | 1328 | 33739 | 154971 |
| Cardiomegaly | 27000 | 11116 | 8087 | 177211 |
| Consolidation | 14783 | 28097 | 27742 | 152792 |
| Edema | 52246 | 20726 | 12984 | 137458 |
| Pleural Effusion | 86187 | 35396 | 11628 | 90203 |

## Co-occurrence

| Positive disease count | Image records |
| --- | --- |
| 0 | 84657 |
| 1 | 80260 |
| 2 | 43911 |
| 3 | 12883 |
| 4 | 1654 |
| 5 | 49 |

Images with at least two positive labels: **58497**. Largest positive pair: **Edema + Pleural Effusion** (26521 records).

## Class imbalance (known binary labels only)

| Disease | Known binary labels | Positive-to-negative ratio | Positive percentage among known binary labels |
| --- | --- | --- | --- |
| Atelectasis | 34704 | 25.1325 | 96.17 |
| Cardiomegaly | 38116 | 2.4289 | 70.84 |
| Consolidation | 42880 | 0.5261 | 34.48 |
| Edema | 72972 | 2.5208 | 71.6 |
| Pleural Effusion | 121583 | 2.4349 | 70.89 |

## Frontal/lateral views

| View | Image records | Percentage |
| --- | --- | --- |
| Frontal | 191027 | 85.5 |
| Lateral | 32387 | 14.5 |

## Image-file check

| Check | Status | Value | Details |
| --- | --- | --- | --- |
| Image-file existence | Completed | 223414 | Image paths checked; image pixels were not loaded. |
| Missing image files | Completed | 0 | Reported separately from missing disease labels. |

## Workflow status

| Status | Detail |
| --- | --- |
| Completed | Metadata-only analysis completed. |

Radiological concepts in the workbook are educational descriptors, separate from CheXpert labels, and marked for literature verification.