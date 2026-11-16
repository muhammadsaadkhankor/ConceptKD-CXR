MobileNetV3-Small full-data run completed and was verified on the GTX 1080 Ti. Both configured epochs completed without interruption. Only MobileNetV3-Small was trained in this step.

`README.md` was empty (0 bytes), and `train_student.py` contained no separate documented full-run schedule. This run therefore used the trainer's existing defaults explicitly: 2 epochs, batch size 8, 0 DataLoader workers, seed 42, AdamW with learning rate 0.0001 and weight decay 0.0001, and no learning-rate scheduler. The default MobileNetV3-Small ImageNet initialization was `MobileNet_V3_Small_Weights.IMAGENET1K_V1`; a fresh five-output classifier was initialized. No fine-tuned smoke-test checkpoint was loaded. The cached official pretrained initialization file was reused, with its URL and SHA-256 recorded in `initialization.json`.

The original input files were `D:\Grad School\AppliedML\CheXpert\data\train.csv` and `D:\Grad School\AppliedML\CheXpert\data\valid.csv`. Their SHA-256 hashes were unchanged after training.

The team's trainer filters rows to frontal views and masks labels; it has no target-positive-only or other-finding exclusion row filter. The new `--full-data` option uses every frontal row from both original CSVs and bypasses `choose_pilot`. It preserves multiple studies/images from a patient within a split. Patient sets across splits do not overlap.

| Split | Frontal images expected and processed per epoch | Patients | Observed labels per epoch | Definite negative target labels | Batches at size 8 | Final batch |
|---|---:|---:|---:|---:|---:|---:|
| Train | 191,027 | 64,534 | 262,378 | 69,716 | 23,879 | 3 |
| Validation | 202 | 200 | 1,010 | 731 | 26 | 2 |

All 67,226 training images without any definite target positive were retained, including target-negative examples. Of the full training set, 39,095 images have no observed target labels and are still processed, with their labels masked. There are 4 training images with all five targets explicitly negative and 74 such validation images. No positive-only pilot restriction was applied. Detailed preflight counts are in `dataset_summary.json`.

Exact PowerShell command, also saved in `command.txt`:

```powershell
Set-Location -LiteralPath 'D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR'
$env:TORCH_HOME = Join-Path (Get-Location) 'outputs\mobilenetv3_small_full_seed42\torch_cache'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\mobilenetv3_small_full_seed42\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u train_student.py --model mobilenet_v3_small --full-data --data-dir 'D:\Grad School\AppliedML\CheXpert\data' --output-dir 'outputs\mobilenetv3_small_full_seed42' --device cuda --epochs 2 --batch-size 8 --num-workers 0 --seed 42 --lr 0.0001 --log-every 500 2>&1 | Tee-Object -FilePath 'outputs\mobilenetv3_small_full_seed42\training.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

The original CSVs contain lateral rows, so the existing exact-subset CSV overrides were unsuitable for loading them directly. `--full-data` reads the originals with the existing frontal-view filter and never invokes the pilot sampler. `train_used.csv` and `valid_used.csv` record the effective frontal rows, including the derived patient_id column.

Preprocessing was the same shared pipeline as the smoke runs: convert to RGB, resize directly to 224 x 224, convert to tensor, normalize by ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Label/output order: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

Only exact 0 and 1 labels are observed. Uncertain -1 and blank/NaN labels contribute to neither loss nor accuracy. Negative 0 labels are retained and scored normally. Loss is binary cross entropy with logits averaged over observed labels; epoch loss aggregates batches weighted by their observed-label counts. Accuracy is correct observed binary labels divided by observed labels after sigmoid threshold >= 0.5, pooled across findings and images. Training metrics aggregate batches as weights are updated; validation is evaluated after the epoch. Entirely unobserved batches, if encountered, skip the optimizer step but still count toward processed images.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Epoch seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 0.2933036745248074 | 0.8829856161720876 | 0.8745850194798838 | 0.6386138613861386 | 4883.0402558999995 |
| 2 | 0.2637861341375801 | 0.897133143784921 | 0.8715247860049257 | 0.6554455445544555 | 1159.1220388000002 |

Epoch 1 actual coverage matched all expected counts, including final batches of 3 training and 2 validation images. Epoch 2 matched them again. Correct-label counts were 231,676 / 262,378 train and 645 / 1,010 validation in epoch 1; 235,388 / 262,378 train and 662 / 1,010 validation in epoch 2. All batches passed finite-logit and finite-loss checks, and CUDA outputs were checked during execution. The model's parameter device was cuda:0.

Best epoch by validation loss: 2. Best and final metrics are therefore identical: train loss 0.2637861341375801, train accuracy 0.897133143784921, validation loss 0.8715247860049257, validation accuracy 0.6554455445544555.

Trainer-reported total elapsed time, including CSV loading, path checks, model construction, and training: 6230.8507433999985 seconds (1h 43m 51s). Training-loop time, including epoch-end artifact saves: 6043.201409299996 seconds (1h 40m 43s). These measurements begin inside the trainer and exclude the earlier repository inspection/code edits and Python module import startup. The run finished at 2026-11-16T20:29:42.619088+00:00. Timing details are in `run_summary.json`.

Uncached file reads dominated the first epoch. A concurrent read-only cache warmup read all 191,027 frontal training JPEGs (10,110,910,855 bytes) in 510.21474820000003 seconds using 16 I/O threads, with no image copies or modifications. This populated the OS file cache and improved throughput. It did not change the training DataLoader's 0 workers, batch size, RNG state, sampler order, inputs, or optimization protocol. The cache pass ran concurrently, so its elapsed time is already included in the training wall time, not added separately. Details are in `cache_warmup.json`. The original training process was never restarted.

`student_best.pth` and `student_last.pth` are epoch-2 checkpoints containing the complete state dictionary, architecture, label order, initialization identifier, preprocessing, source paths/counts, model/backend/version, and run settings. `student_weights.pth` is the standalone final state dictionary. Checkpoints are model checkpoints, without optimizer state for exact training resume. Checkpoint writes use a temporary file plus rename, and last weights, metrics, graph, and summary are saved after every completed epoch.

Reload command:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\mobilenetv3_small_full_seed42\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u scripts\verify_smoke_checkpoint.py --run-dir 'outputs\mobilenetv3_small_full_seed42' --validation-images 8
```

Verification ran in a separate Python process. Best and last checkpoints were each loaded with weights_only=True and strict=True into separate fresh MobileNetV3-Small instances built with pretrained=False. Each ran inference on 8 original validation frontal images on CUDA and produced shape (8, 5), with all 40 logits finite. Logits ranged from -4.56248140335083 to 4.379424571990967. The standalone final weights matched the last checkpoint tensor-for-tensor. Details are in `reload_test.json`.

Artifacts in this folder:

- `student_best.pth`, `student_last.pth`, `student_weights.pth`: best/last model checkpoints and standalone final weights.
- `history.csv`: full-precision epoch metrics, observed/correct labels, image and batch counts, final-batch counts, and timing.
- `learning_curves.png`: visually checked train/validation loss and accuracy graphs.
- `run_config.json`, `dataset_summary.json`, `initialization.json`, `run_summary.json`: settings, expected counts, provenance, and final summary/timing.
- `reload_test.json`: strict reload and finite-logit results.
- `train_used.csv`, `valid_used.csv`: effective full frontal split snapshots.
- `command.txt`, `training.log`, `progress.json`, `FULL_TRAINING_REPORT.md`: execution records and this report.
- `cache_warmup.json`, `cache_warmup_progress.json`, `read_benchmark_paths.json`: read-only I/O diagnostics and cache warmup record.
- `torch_cache`, `matplotlib_cache`: official initialization and font caches.

Code changes: added explicit `--full-data` sampler bypass while preserving existing pilot and exact-subset modes; added batch progress and timing; checked final-batch and observed-label coverage; enforced CUDA output placement and finite logits/losses; persisted best/last weights, metrics, graphs, and summaries after each epoch; added validation-batch selection to the shared checkpoint verifier. Existing model construction, preprocessing, masks, optimizer defaults, and default model were retained. Focused full-data/partial-batch tests, actual full-epoch coverage checks, original-source hash checks, strict reload checks, and `git diff --check` passed.

Stopped after verifying the completed MobileNetV3-Small full-data run. No MobileNetV4-Small or EfficientNet full training was run.
