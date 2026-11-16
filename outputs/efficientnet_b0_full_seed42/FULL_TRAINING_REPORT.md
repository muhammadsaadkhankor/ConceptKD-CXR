EfficientNet-B0 full-data training completed and passed the requested execution and strict-reload checks on the NVIDIA GeForce GTX 1080 Ti. Both configured epochs completed. Only EfficientNet-B0 was trained in this step; the single Word comparison report was left for the next step.

Working directory: `D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR`. Python environment: `..\.venv\Scripts\python.exe`. The existing trainer and shared model builder already supported this run; no training-code changes were needed.

Model: `torchvision.models.efficientnet_b0`, selected with `--model efficientnet_b0`. Initialization: `EfficientNet_B0_Weights.IMAGENET1K_V1`, with a fresh `Linear(1280, 5)` replacing the original 1000-output classifier. All model parameters were optimized. No fine-tuned checkpoint was used for initialization. The official cached ImageNet file `efficientnet_b0_rwightman-7f5810bc.pth` was copied into this run's cache and hash-verified. URL: `https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth`. SHA-256: `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`. Full provenance is in `initialization.json`.

Settings: CUDA, 2 epochs, batch size 8, 0 DataLoader workers, seed 42, AdamW learning rate 0.0001 and weight decay 0.0001, no scheduler, and progress every 500 batches. PyTorch remained 2.12.1+cu126 and torchvision remained 0.27.1+cu126.

Exact PowerShell command, also saved in `command.txt`:

```powershell
Set-Location -LiteralPath 'D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR'
$env:TORCH_HOME = Join-Path (Get-Location) 'outputs\efficientnet_b0_full_seed42\torch_cache'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\efficientnet_b0_full_seed42\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u train_student.py --model efficientnet_b0 --full-data --data-dir 'D:\Grad School\AppliedML\CheXpert\data' --output-dir 'outputs\efficientnet_b0_full_seed42' --device cuda --epochs 2 --batch-size 8 --num-workers 0 --seed 42 --lr 0.0001 --log-every 500 2>&1 | Tee-Object -FilePath 'outputs\efficientnet_b0_full_seed42\training.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Original sources: `D:\Grad School\AppliedML\CheXpert\data\train.csv` and `D:\Grad School\AppliedML\CheXpert\data\valid.csv`. Full-data mode bypassed pilot sampling, selected frontal views, and retained all target-negative examples. It applied no positive-only or other-finding exclusion row filter.

| Split | Images expected and processed per epoch | Patients | Observed labels per epoch | Definite negative target labels | Batches | Final batch |
|---|---:|---:|---:|---:|---:|---:|
| Training | 191,027 | 64,534 | 262,378 | 69,716 | 23,879 | 3 |
| Validation | 202 | 200 | 1,010 | 731 | 26 | 2 |

Patient overlap was zero. All 67,226 training images without any definite target positive were retained. The original CSV hashes were unchanged after training. `train_used.csv` and `valid_used.csv` capture the effective frontal rows and derived patient IDs; `dataset_summary.json` records source paths, expected counts, and hashes.

Preprocessing matched both MobileNet full runs: convert to RGB, directly resize to 224 x 224, convert to tensor, and normalize using ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Label/output order: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

Only exact 0 and 1 labels contribute to loss and accuracy. Uncertain -1 and blank/NaN labels are masked out, while definite negative 0 labels are scored normally. Loss is binary cross entropy with logits averaged over observed labels; epoch loss aggregates batches weighted by observed-label counts. Accuracy is correct observed binary labels divided by total observed labels, after sigmoid threshold >= 0.5, pooled across images and findings. Training metrics aggregate batches as weights are updated; validation uses evaluation mode after each epoch. These policies and the run hyperparameters were checked against both MobileNet full-run configurations.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Epoch seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 0.2899909558224625 | 0.8865682336171478 | 0.9437245579049138 | 0.6376237623762376 | 1741.7420668999985 |
| 2 | 0.25492949960621564 | 0.9020535258291472 | 0.8523224533194362 | 0.6742574257425743 | 1867.8583979999967 |

Both epochs matched all expected image, observed-label, batch, and final-partial-batch counts. Correct-label counts were 232,616/262,378 training and 644/1,010 validation in epoch 1; 236,679/262,378 training and 681/1,010 validation in epoch 2. All batches passed finite-logit and finite-loss checks, and CUDA placement was checked during execution. The model parameter device was cuda:0.

Best checkpoint by validation loss: epoch 2. Best and final metrics are identical: training loss 0.25492949960621564, training accuracy 0.9020535258291472, validation loss 0.8523224533194362, and validation accuracy 0.6742574257425743.

Trainer-reported elapsed time, including its setup: 3686.403763399998 seconds (1h 1m 26s). Training-loop time, including epoch-end saves: 3612.1744028999965 seconds (1h 0m 12s). These measurements begin inside the trainer and exclude earlier repository inspection, cache staging, and Python module import startup. Completion timestamp: 2026-11-16T22:43:09.343377+00:00. Exact timing and best/final metrics are in `run_summary.json`.

A read-only cache pass read all 191,027 training files (10,110,910,855 bytes) in 88.56210860000283 seconds using 16 I/O threads, without image copies or modifications. This warmed the OS file cache concurrently with trainer startup. The training DataLoader still had 0 workers; model settings, RNG state, input selection, and optimization were unchanged. Details are in `cache_warmup.json`.

`student_best.pth` and `student_last.pth` contain epoch-2 model state dictionaries and architecture, label order, exact ImageNet initialization variant and URL, preprocessing, source paths/counts, versions, seed, and run settings. `student_weights.pth` contains the standalone final state dictionary. These are model checkpoints, without optimizer state for exact training resume. Checkpoints, metrics, graphs, and summaries were saved after each completed epoch, with temporary-file-and-rename writes for model files.

Reload command:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\efficientnet_b0_full_seed42\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u scripts\verify_smoke_checkpoint.py --run-dir 'outputs\efficientnet_b0_full_seed42' --validation-images 8
```

Verification ran in a separate Python process. Both checkpoints were loaded with weights_only=True and strict=True into separate fresh EfficientNet-B0 instances created by the shared builder with pretrained=False and a five-output classifier. Each produced shape (8, 5), with all 40 logits finite on the eight-image original validation batch. Both produced logits ranging from -5.323691368103027 to 4.1058855056762695. Standalone final weights matched every tensor in the last checkpoint. Full results are in `reload_test.json`.

Artifacts in this folder:

- `student_best.pth`, `student_last.pth`, `student_weights.pth`: model checkpoints and standalone final weights.
- `history.csv`: full-precision metrics, image/batch/observed/correct-label counts, final-batch sizes, and timing.
- `learning_curves.png`: visually checked training/validation loss and accuracy graphs.
- `run_config.json`, `dataset_summary.json`, `initialization.json`, `run_summary.json`: configuration, counts, provenance, and final summary.
- `reload_test.json`: strict fresh-model reload, finite outputs, and tensor equality results.
- `train_used.csv`, `valid_used.csv`: effective full frontal split snapshots.
- `command.txt`, `training.log`, `progress.json`, `FULL_TRAINING_REPORT.md`: execution records and this report.
- `cache_warmup.json`, `cache_warmup_progress.json`: read-only cache-pass records.
- `torch_cache`, `matplotlib_cache`: official initialization and font caches.

Checks passed for both epochs' exact coverage and partial batches, finite losses, accuracy ratios, matching MobileNet full-run settings and policies, unchanged source CSVs and pretrained file, strict reload into fresh models, tensor-for-tensor standalone-weight equality, and `git diff --check`.

Stopped after EfficientNet-B0 full-run verification. The Word comparison report was not created in this step.
