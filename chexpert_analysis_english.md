# CheXpert Dataset Analysis Report

- Dataset: `ashery/chexpert`
- Dataset directory: `C:\Users\Gloria\.cache\kagglehub\datasets\ashery\chexpert\versions\1`
- Number of label classes: **14**
- Total number of images in the label CSV files (after removing duplicate image paths): **223,648**

## Number of Images per Class

| Class | Positive (1) | Uncertain (-1) | Negative (0) | Missing Labels | Positive Percentage of All Images |
|---|---:|---:|---:|---:|---:|
| No Finding | 22,419 | 0 | 196 | 201,033 | 10.02% |
| Enlarged Cardiomediastinum | 10,907 | 12,403 | 21,763 | 178,575 | 4.88% |
| Cardiomegaly | 27,068 | 8,087 | 11,282 | 177,211 | 12.10% |
| Lung Lesion | 9,187 | 1,488 | 1,503 | 211,470 | 4.11% |
| Lung Opacity | 105,707 | 5,598 | 6,707 | 105,636 | 47.26% |
| Edema | 52,291 | 12,984 | 20,915 | 137,458 | 23.38% |
| Consolidation | 14,816 | 27,742 | 28,298 | 152,792 | 6.62% |
| Pneumonia | 6,047 | 18,770 | 3,025 | 195,806 | 2.70% |
| Atelectasis | 33,456 | 33,739 | 1,482 | 154,971 | 14.96% |
| Pneumothorax | 19,456 | 3,145 | 56,567 | 144,480 | 8.70% |
| Pleural Effusion | 86,254 | 11,628 | 35,563 | 90,203 | 38.57% |
| Pleural Other | 3,524 | 2,653 | 549 | 216,922 | 1.58% |
| Fracture | 9,040 | 642 | 2,746 | 211,220 | 4.04% |
| Support Devices | 116,108 | 1,079 | 6,264 | 100,197 | 51.92% |

## Class Imbalance

- Median number of positive images across classes: **20,937.5**.
- Ratio of the highest to the lowest nonzero positive image count: **32.9:1**.
- Classes with positive image counts below 10% of the highest class count are considered to have relatively few samples:
  - **Pleural Other**: 3,524 positive images
  - **Pneumonia**: 6,047 positive images
  - **Fracture**: 9,040 positive images
  - **Lung Lesion**: 9,187 positive images
  - **Enlarged Cardiomediastinum**: 10,907 positive images

## Clinical Findings (Descriptive Analysis)

- Dataset labels were extracted from radiology reports rather than independently verified patient diagnoses. A positive count means the finding has a label of `1`.
- One image can have multiple positive findings, so class counts overlap and should not be added together to calculate the total number of images.
- In CheXpert, `-1` indicates an uncertain label. Uncertain and missing labels are counted separately, and neither is treated as positive.
- Clinical findings with the highest positive image counts:
  - **Lung Opacity**: 105,707 positive images (47.26% of all images)
  - **Pleural Effusion**: 86,254 positive images (38.57% of all images)
  - **Edema**: 52,291 positive images (23.38% of all images)
  - **Atelectasis**: 33,456 positive images (14.96% of all images)
  - **Cardiomegaly**: 27,068 positive images (12.10% of all images)

Most frequent pairs of co-occurring positive clinical findings:
- **Lung Opacity + Pleural Effusion**: 52,258 images
- **Lung Opacity + Edema**: 27,500 images
- **Edema + Pleural Effusion**: 26,537 images
- **Atelectasis + Pleural Effusion**: 16,339 images
- **Lung Opacity + Atelectasis**: 14,226 images

This report describes dataset labels only. The results do not represent clinical disease prevalence, causal relationships, or diagnostic performance.
