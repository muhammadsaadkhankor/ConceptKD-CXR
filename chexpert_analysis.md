# CheXpert Dataset Analysis Report

- Dataset: `ashery/chexpert`
- Dataset directory: `D:\uottawa\phd\5255 ML\project\dataset`
- Number of labeled classes: **14**
- Total unique images: **223,648**

## Number of Images per Class

| Class | Negative (0) count (% of total) | Positive (1) count (% of total) | Uncertain (-1) count (% of total) | Missing count (% of total) |
|---|---:|---:|---:|---:|
| No Finding | 196 (0.09%) | 22,419 (10.02%) | 0 (0.00%) | 201,033 (89.89%) |
| Enlarged Cardiomediastinum | 21,763 (9.73%) | 10,907 (4.88%) | 12,403 (5.55%) | 178,575 (79.85%) |
| Cardiomegaly | 11,282 (5.04%) | 27,068 (12.10%) | 8,087 (3.62%) | 177,211 (79.24%) |
| Lung Lesion | 1,503 (0.67%) | 9,187 (4.11%) | 1,488 (0.67%) | 211,470 (94.55%) |
| Lung Opacity | 6,707 (3.00%) | 105,707 (47.26%) | 5,598 (2.50%) | 105,636 (47.23%) |
| Edema | 20,915 (9.35%) | 52,291 (23.38%) | 12,984 (5.81%) | 137,458 (61.46%) |
| Consolidation | 28,298 (12.65%) | 14,816 (6.62%) | 27,742 (12.40%) | 152,792 (68.32%) |
| Pneumonia | 3,025 (1.35%) | 6,047 (2.70%) | 18,770 (8.39%) | 195,806 (87.55%) |
| Atelectasis | 1,482 (0.66%) | 33,456 (14.96%) | 33,739 (15.09%) | 154,971 (69.29%) |
| Pneumothorax | 56,567 (25.29%) | 19,456 (8.70%) | 3,145 (1.41%) | 144,480 (64.60%) |
| Pleural Effusion | 35,563 (15.90%) | 86,254 (38.57%) | 11,628 (5.20%) | 90,203 (40.33%) |
| Pleural Other | 549 (0.25%) | 3,524 (1.58%) | 2,653 (1.19%) | 216,922 (96.99%) |
| Fracture | 2,746 (1.23%) | 9,040 (4.04%) | 642 (0.29%) | 211,220 (94.44%) |
| Support Devices | 6,264 (2.80%) | 116,108 (51.92%) | 1,079 (0.48%) | 100,197 (44.80%) |

## Class Imbalance

Ranked by the coefficient of variation across the four label-status percentages; a higher score indicates a more imbalanced distribution.

| Rank | Finding | Negative (0) % | Positive (1) % | Uncertain (-1) % | Missing % | Imbalance score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Pleural Other | 0.25% | 1.58% | 1.19% | 96.99% | 1.663 |
| 2 | Lung Lesion | 0.67% | 4.11% | 0.67% | 94.55% | 1.607 |
| 3 | Fracture | 1.23% | 4.04% | 0.29% | 94.44% | 1.605 |
| 4 | No Finding | 0.09% | 10.02% | 0.00% | 89.89% | 1.507 |
| 5 | Pneumonia | 1.35% | 2.70% | 8.39% | 87.55% | 1.448 |
| 6 | Enlarged Cardiomediastinum | 9.73% | 4.88% | 5.55% | 79.85% | 1.269 |
| 7 | Cardiomegaly | 5.04% | 12.10% | 3.62% | 79.24% | 1.259 |
| 8 | Atelectasis | 0.66% | 14.96% | 15.09% | 69.29% | 1.049 |
| 9 | Consolidation | 12.65% | 6.62% | 12.40% | 68.32% | 1.005 |
| 10 | Pneumothorax | 25.29% | 8.70% | 1.41% | 64.60% | 0.978 |
| 11 | Support Devices | 2.80% | 51.92% | 0.48% | 44.80% | 0.940 |
| 12 | Lung Opacity | 3.00% | 47.26% | 2.50% | 47.23% | 0.890 |
| 13 | Edema | 9.35% | 23.38% | 5.81% | 61.46% | 0.882 |
| 14 | Pleural Effusion | 15.90% | 38.57% | 5.20% | 40.33% | 0.598 |

## Clinical Findings Notes

- Labels are extracted from radiology reports and are not independently verified patient diagnoses.
- An image may have multiple findings, so positive image sets can overlap.
- `1` means positive, `0` means negative, and `-1` means uncertain; missing labels are reported separately.
- This report describes dataset labels only; it does not establish clinical prevalence, causation, or diagnostic performance.

## t-SNE Visualization

- Figure: [chexpert_tsne.png](./chexpert_tsne.png)
- Images used: **223,648**
- Method: Uses all labeled images; grayscale images are resized to 32×32 (1,024 pixel features per image), then reduced with incremental PCA and t-SNE; each panel shows one finding's label status.
