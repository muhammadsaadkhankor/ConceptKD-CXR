MobileNetV4-Small full-data training completed and passed the requested execution and reload checks on the NVIDIA GeForce GTX 1080 Ti. Both configured epochs completed. Validation loss remained high under the requested protocol; the actual values are reported below.

Working directory: `D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR`. Python environment: `..\.venv\Scripts\python.exe`. Only MobileNetV4-Small was trained in this step. The existing trainer and shared model builder already supported this run; no training-code changes were needed.

Model identifier: `mobilenetv4_conv_small.e2400_r224_in1k` from timm 1.0.31, selected with `--model mobilenetv4_small`. Initialization used the original ImageNet-1k weights and a freshly initialized five-output classifier via `timm.create_model(identifier, pretrained=True, num_classes=5)`. No fine-tuned model checkpoint was used for initialization. The official pretrained cache was copied from the earlier run's initialization cache, verified by hash, and used offline. Revision: `331fb803779522b685cf942e15f914fb6741c1eb`. Pretrained model.safetensors SHA-256: `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4`. Full provenance is in `initialization.json`.

Settings: CUDA, 2 epochs, batch size 8, 0 DataLoader workers, seed 42, AdamW learning rate 0.0001 and weight decay 0.0001, no scheduler, and progress every 500 batches. PyTorch remained 2.12.1+cu126 and torchvision remained 0.27.1+cu126. All model parameters were optimized.

Exact PowerShell command, also saved in `command.txt`:

```powershell
Set-Location -LiteralPath 'D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR'
$env:TORCH_HOME = Join-Path (Get-Location) 'outputs\mobilenetv4_small_full_seed42\torch_cache'
$env:HF_HOME = Join-Path (Get-Location) 'outputs\mobilenetv4_small_full_seed42\hf_cache'
$env:HF_HUB_OFFLINE = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\mobilenetv4_small_full_seed42\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u train_student.py --model mobilenetv4_small --full-data --data-dir 'D:\Grad School\AppliedML\CheXpert\data' --output-dir 'outputs\mobilenetv4_small_full_seed42' --device cuda --epochs 2 --batch-size 8 --num-workers 0 --seed 42 --lr 0.0001 --log-every 500 2>&1 | Tee-Object -FilePath 'outputs\mobilenetv4_small_full_seed42\training.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

The original input CSVs were `D:\Grad School\AppliedML\CheXpert\data\train.csv` and `D:\Grad School\AppliedML\CheXpert\data\valid.csv`. Full-data mode bypassed pilot sampling and retained all frontal rows. There was no target-positive-only requirement or other-finding exclusion. Target-negative images and labels were retained.

| Split | Images expected and processed per epoch | Patients | Observed labels per epoch | Definite negative target labels | Batches | Final batch |
|---|---:|---:|---:|---:|---:|---:|
| Training | 191,027 | 64,534 | 262,378 | 69,716 | 23,879 | 3 |
| Validation | 202 | 200 | 1,010 | 731 | 26 | 2 |

Patient overlap was zero. All 67,226 training images without a definite target positive remained eligible. The original CSV contents were unchanged, verified by SHA-256 after training. `train_used.csv` and `valid_used.csv` capture the effective frontal rows and derived patient IDs. Expected counts and original CSV hashes are in `dataset_summary.json`.

Preprocessing was identical to the MobileNetV3 full run: RGB conversion, direct resize to 224 x 224, tensor conversion, and ImageNet normalization with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Label/output order: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

Only exact 0 and 1 labels contribute to loss and accuracy. Uncertain -1 and blank/NaN labels are masked. Negative 0 labels are observed and scored normally. Loss is binary cross entropy with logits over observed labels, with epoch loss aggregating batches weighted by observed-label counts. Accuracy is correct observed binary labels divided by observed labels after sigmoid threshold >= 0.5, pooled across images and findings. Training metrics aggregate batches as weights are updated; validation uses evaluation mode after the epoch. These preprocessing, label, optimizer, and run-setting fields were checked against the MobileNetV3 full-run configuration.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Epoch seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 0.40396039145172713 | 0.8344373384963678 | 9.888751314418151 | 0.48316831683168315 | 1114.056317399998 |
| 2 | 0.35906812019743795 | 0.8511917919947557 | 7.480097633777278 | 0.5623762376237624 | 1317.3325026999955 |

Both epochs matched all expected image, batch, final-partial-batch, and observed-label counts. Correct-label counts were 218,938/262,378 training and 488/1,010 validation in epoch 1; 223,334/262,378 training and 568/1,010 validation in epoch 2. Every batch passed finite-logit and finite-loss checks, and model logits were checked for CUDA placement. The model parameter device was cuda:0.

Best checkpoint by validation loss: epoch 2. Best and final metrics are identical: train loss 0.35906812019743795, train accuracy 0.8511917919947557, validation loss 7.480097633777278, validation accuracy 0.5623762376237624. Execution and reload checks passed; these checks do not imply strong validation performance.

Trainer-reported total elapsed time, including its setup: 2477.1707957000035 seconds (41m 17s). Training-loop time, including epoch-end saves: 2434.0325047000006 seconds (40m 34s). Measurements begin inside the trainer and exclude earlier repository inspection, cache staging, and Python import startup. Completion timestamp: 2026-11-16T21:20:13.096403+00:00. Timing and best/final metrics are recorded in `run_summary.json`.

A concurrent read-only cache pass read all 191,027 training JPEGs (10,110,910,855 bytes) in 50.43411609999748 seconds, using 16 I/O threads. It made no image copies or modifications. This was an OS file-cache warmup; the training DataLoader still had 0 workers and the model, RNG state, sample order, and optimization settings were unchanged. Details are in `cache_warmup.json`.

`student_best.pth` and `student_last.pth` are epoch-2 model checkpoints containing state dictionary, architecture, exact model identifier, label order, initialization, backend/version, preprocessing, source paths and counts, and run settings. `student_weights.pth` is the standalone final state dictionary. These checkpoints omit optimizer state for exact training resume. Checkpoints, metrics, graphs, and summaries were persisted after each completed epoch; model writes use temporary files followed by rename.

Reload verification ran in a separate Python process using the shared builder:

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'outputs\mobilenetv4_small_full_seed42\hf_cache'
$env:HF_HUB_OFFLINE = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\mobilenetv4_small_full_seed42\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u scripts\verify_smoke_checkpoint.py --run-dir 'outputs\mobilenetv4_small_full_seed42' --validation-images 8
```

Both checkpoints were loaded with weights_only=True and strict=True into separate fresh MobileNetV4-Small instances created with pretrained=False and num_classes=5. Each produced shape (8, 5), with all 40 logits finite on the eight-image original validation batch. Logits ranged from -2.0506250858306885 to 3.7940285205841064. Standalone final weights matched every tensor in the last checkpoint. Full results are in `reload_test.json`.

Artifacts in this folder:

- `student_best.pth`, `student_last.pth`, `student_weights.pth`: model checkpoints and standalone final weights.
- `history.csv`: full-precision epoch metrics, coverage/correct-label counts, final-batch sizes, and timing.
- `learning_curves.png`: train/validation loss and accuracy graphs, visually checked; the accuracy legend was positioned clear of the curves when rendered from the saved metrics.
- `run_config.json`, `dataset_summary.json`, `initialization.json`, `run_summary.json`: configuration, counts, provenance, and final summary.
- `reload_test.json`: strict reload, finite-output, and weight-equality verification.
- `train_used.csv`, `valid_used.csv`: effective frontal split snapshots.
- `command.txt`, `training.log`, `progress.json`, `FULL_TRAINING_REPORT.md`: execution records and this report.
- `cache_warmup.json`, `cache_warmup_progress.json`: read-only cache pass records.
- `hf_cache`, `matplotlib_cache`: official initialization and font caches.

Post-run checks confirmed exact coverage in both epochs, finite losses, correct accuracy ratios, matching MobileNetV3 full-run settings/policies, unchanged original CSVs and pretrained weight file, strict reload into fresh models, tensor-for-tensor raw-weight equality, and `git diff --check`.

Stopped after MobileNetV4-Small full-run verification. No EfficientNet-B0 full training was run.
