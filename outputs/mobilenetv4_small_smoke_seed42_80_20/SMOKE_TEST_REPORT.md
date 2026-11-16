MobileNetV4-Small completed its two-epoch CUDA smoke test on the NVIDIA GeForce GTX 1080 Ti. Working directory: `D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR`. This run used only MobileNetV4-Small; no MobileNetV3-Small or EfficientNet training was run.

The exact timm identifier is `mobilenetv4_conv_small.e2400_r224_in1k`, exposed by the trainer as `--model mobilenetv4_small`. Initialization used the official timm ImageNet-1k pretrained weights, with the original 1000-class classifier discarded and a newly initialized five-output classifier. All parameters were optimized with AdamW, learning rate 0.0001 and weight decay 0.0001. Seed was 42. No random-initialization fallback was used.

Official source: https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k

The downloaded pretrained weight revision was `331fb803779522b685cf942e15f914fb6741c1eb`; the SHA-256 of `model.safetensors` was `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4`. Full provenance and versions are recorded in `initialization.json`.

timm 1.0.31 was installed in the existing Python 3.11 environment. Installation constraints preserved torch 2.12.1+cu126 and torchvision 0.27.1+cu126. The repository requirements were unchanged. `dependency_constraints.txt` records the protected builds.

Exact PowerShell command, also saved in `command.txt`:

```powershell
Set-Location -LiteralPath 'D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR'
$env:TORCH_HOME = Join-Path (Get-Location) 'outputs\mobilenetv4_small_smoke_seed42_80_20\torch_cache'
$env:HF_HOME = Join-Path (Get-Location) 'outputs\mobilenetv4_small_smoke_seed42_80_20\hf_cache'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\mobilenetv4_small_smoke_seed42_80_20\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u train_student.py --model mobilenetv4_small --data-dir 'D:\Grad School\AppliedML\CheXpert\data' --train-csv 'subsets\target_only_seed42\train_80.csv' --valid-csv 'subsets\target_only_seed42\validation_20.csv' --output-dir 'outputs\mobilenetv4_small_smoke_seed42_80_20' --device cuda --epochs 2 --batch-size 8 --num-workers 0 --seed 42 --lr 0.0001 2>&1 | Tee-Object -FilePath 'outputs\mobilenetv4_small_smoke_seed42_80_20\training.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Train images / batches / observed labels | Validation images / batches / observed labels |
|---|---:|---:|---:|---:|---|---|
| 1 | 2.0011401674640714 | 0.5572139303482587 | 1.6236029741715412 | 0.5714285714285714 | 80 / 10 / 201 | 20 / 3 / 49 |
| 2 | 1.0019700215823615 | 0.7064676616915423 | 1.7604632888521468 | 0.4897959183673469 | 80 / 10 / 201 | 20 / 3 / 49 |

Each epoch processed all 80 training images and all 20 validation images. Batch size was 8, zero workers were used, and the final validation batch contained four images. There were 80 distinct training patients and 20 distinct validation patients, with no overlap. Both CSVs were used directly without sampling. The saved CSV snapshots match the inputs byte-for-byte, and the input hashes match the verified subset manifest. The original full validation CSV remains reserved and was not used.

Preprocessing is unchanged from the MobileNetV3-Small run: convert to RGB, resize directly to 224 x 224 using the existing torchvision transform, convert to tensor, and normalize with ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. The timm model's alternative inference resize/crop recipe was not substituted.

Label order is Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion. Only exact 0 and 1 labels contribute to loss and accuracy. Uncertain -1 and blank/NaN labels are masked out. Loss is mean binary cross entropy with logits over observed labels, with epoch loss weighted by each batch's observed-label count. Accuracy is correct observed labels divided by total observed labels after sigmoid threshold >= 0.5, rather than per-image accuracy. Correct counts were 112/201 training and 28/49 validation in epoch 1; 142/201 training and 24/49 validation in epoch 2.

`student_best.pth` is epoch 1, selected by lowest validation loss. `student_last.pth` is epoch 2. Both include model architecture, exact model identifier, backend and versions, label order, initialization, preprocessing, seed, input CSV paths, counts, and run settings. `student_weights.pth` is the standalone final state dictionary and matches the last checkpoint tensor-for-tensor. The checkpoints are reloadable model checkpoints; optimizer state for resuming training is not included.

Fresh-model verification ran in a separate process using `scripts/verify_smoke_checkpoint.py`. Each checkpoint was strictly loaded into its own fresh timm MobileNetV4-Small instance created with pretrained=False and num_classes=5, without downloading weights. Both ran CUDA inference on all 100 images and produced shape (100, 5), with all 500 logits finite per checkpoint. The best checkpoint's logits ranged from -17.371328353881836 to 10.787470817565918; the last checkpoint's ranged from -14.074885368347168 to 16.23900032043457. Details are in `reload_test.json`.

Artifacts in this folder:

- `student_best.pth`, `student_last.pth`: strictly reloadable model checkpoints.
- `student_weights.pth`: standalone final weights.
- `history.csv`: full-precision metrics, image/batch counts, and observed/correct-label counts.
- `learning_curves.png`: visually checked train/validation loss and accuracy graphs.
- `run_config.json`, `initialization.json`: configuration and exact initialization provenance.
- `reload_test.json`: fresh-model inference verification.
- `train_used.csv`, `valid_used.csv`: exact input subset CSV snapshots.
- `training.log`, `command.txt`, `SMOKE_TEST_REPORT.md`, `dependency_constraints.txt`: execution and provenance records.
- `hf_cache`, `matplotlib_cache`: local model and font caches.

Code changes: `train_student.py` now exposes `--model` with the original MobileNetV3-Small default plus an explicit MobileNetV4-Small option. Shared factory and metadata functions keep training and reload construction aligned; timm is imported only by the MobileNetV4 branch. Per-epoch observed-label counts are checked against dataset masks and printed. `scripts/verify_smoke_checkpoint.py` now reconstructs the architecture specified by the checkpoint and supports existing MobileNetV3 checkpoints as well as MobileNetV4. Existing preprocessing, masking, optimizer settings, CSV bypass behavior, and the MobileNetV3 default were retained.

Checks passed for the default model option, exact MobileNetV4 factory arguments, timm registry identifier, preserved CUDA package versions, zero patient overlap, exact image and observed-label coverage in both epochs, unchanged subset hashes, byte-identical CSV snapshots, equality of preprocessing/policies/hyperparameters with the V3 configuration, strict checkpoint reload, raw-weight equality, finite inference outputs, and `git diff --check`.

The Hugging Face cache emitted a Windows symlink-support warning; it used regular cached files successfully. No changes to Windows settings were required.

Stopped after MobileNetV4-Small verification.
