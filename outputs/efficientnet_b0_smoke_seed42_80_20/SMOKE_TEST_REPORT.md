EfficientNet-B0 completed its two-epoch CUDA smoke test on the NVIDIA GeForce GTX 1080 Ti using the existing Python 3.11 environment. Working directory: `D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR`. Only EfficientNet-B0 was trained in this step.

Model: `torchvision.models.efficientnet_b0`. Exact initialization: `EfficientNet_B0_Weights.IMAGENET1K_V1`, downloaded from `https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth`. The original 1000-output classifier was replaced with a newly initialized `Linear(1280, 5)`. All parameters were trained with AdamW, learning rate 0.0001, weight decay 0.0001, and seed 42. No random-initialization fallback was used. The pretrained file SHA-256 was `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`; provenance is saved in `initialization.json`. PyTorch remained 2.12.1+cu126 and torchvision remained 0.27.1+cu126.

Exact PowerShell command, also saved in `command.txt`:

```powershell
Set-Location -LiteralPath 'D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR'
$env:TORCH_HOME = Join-Path (Get-Location) 'outputs\efficientnet_b0_smoke_seed42_80_20\torch_cache'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\efficientnet_b0_smoke_seed42_80_20\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u train_student.py --model efficientnet_b0 --data-dir 'D:\Grad School\AppliedML\CheXpert\data' --train-csv 'subsets\target_only_seed42\train_80.csv' --valid-csv 'subsets\target_only_seed42\validation_20.csv' --output-dir 'outputs\efficientnet_b0_smoke_seed42_80_20' --device cuda --epochs 2 --batch-size 8 --num-workers 0 --seed 42 --lr 0.0001 2>&1 | Tee-Object -FilePath 'outputs\efficientnet_b0_smoke_seed42_80_20\training.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Train images / batches / observed labels | Validation images / batches / observed labels |
|---|---:|---:|---:|---:|---|---|
| 1 | 0.6980999874238351 | 0.472636815920398 | 0.6666056915205352 | 0.6122448979591837 | 80 / 10 / 201 | 20 / 3 / 49 |
| 2 | 0.6360795758849945 | 0.7562189054726368 | 0.6486758760043553 | 0.673469387755102 | 80 / 10 / 201 | 20 / 3 / 49 |

Batch size was 8 with zero workers. Every epoch processed all 80 training images and 20 validation images, with a final four-image validation batch. Observed-label counts were exactly 201 training and 49 validation in both epochs. Correct-label counts were 95/201 training and 30/49 validation in epoch 1; 152/201 training and 33/49 validation in epoch 2. These counts were checked against the saved accuracy values and the trainer's dataset masks.

The input CSVs were used directly without sampling. There were 80 distinct training patients and 20 distinct validation patients, with zero overlap. The original subset hashes remain unchanged, and `train_used.csv` / `valid_used.csv` match the provided CSVs byte-for-byte. Both subsets come from the original training CSV; the original full validation CSV remains reserved.

Preprocessing matches both previous runs: convert to RGB, resize directly to 224 x 224, convert to tensor, normalize with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Label order is Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion. Only exact 0 and 1 labels contribute to loss and accuracy. Uncertain -1 and blank/NaN labels are masked out. Loss is binary cross entropy with logits over observed labels; epoch loss weights batches by their observed-label counts. Accuracy pools correct observed binary labels across images and findings, using sigmoid threshold >= 0.5. Training metrics aggregate batches during weight updates; validation uses evaluation mode after each epoch. The saved configuration was checked against both prior runs for preprocessing, label policy, accuracy calculation, split paths, seed, learning rate, batch size, workers, and package builds.

`student_best.pth` is epoch 2, selected by the lowest validation loss. `student_last.pth` is also epoch 2. Both contain the complete state dictionary plus architecture, model identifier, backend, exact weight variant and source URL, label order, preprocessing, seed, split paths/counts, and run settings. `student_weights.pth` contains the standalone final state dictionary. These are reloadable model checkpoints; optimizer state for resuming training is not included.

Reload verification ran in a separate process through `scripts/verify_smoke_checkpoint.py`. Each checkpoint was loaded with weights_only=True and strict=True into its own fresh EfficientNet-B0 instance built by the same shared model builder with pretrained=False. Both fresh instances ran CUDA inference on all 100 subset images, producing shape (100, 5), with all 500 logits finite for each checkpoint. Logits ranged from -0.3703537881374359 to 0.826260507106781 for both checkpoints. The standalone weights matched every tensor in the last checkpoint. Full results are in `reload_test.json`.

Artifacts in this folder:

- `student_best.pth`, `student_last.pth`: strictly reloadable checkpoints.
- `student_weights.pth`: standalone final weights.
- `history.csv`: full-precision epoch metrics, coverage counts, and observed/correct-label counts.
- `learning_curves.png`: visually checked train/validation loss and accuracy graphs.
- `run_config.json`, `initialization.json`: configuration and exact initialization provenance.
- `reload_test.json`: strict fresh-model inference and standalone-weight verification.
- `train_used.csv`, `valid_used.csv`: exact subset snapshots.
- `training.log`, `command.txt`, `SMOKE_TEST_REPORT.md`: execution log, exact command, and this report.
- `torch_cache`, `matplotlib_cache`: pretrained initialization and font caches.

Code changes: added `efficientnet_b0` to the existing `--model` choices; added the torchvision EfficientNet-B0 branch to the shared builder with explicit IMAGENET1K_V1 initialization and a five-output classifier; recorded exact initialization metadata; added EfficientNet-B0 to the reload architecture lookup. MobileNetV3-Small remains the default, and preprocessing, label masking, accuracy, optimizer settings, and CSV override behavior remain shared and unchanged. Focused checks passed for the exact factory arguments, classifier dimensions, pretrained=False reload behavior, default model, CSV sampler bypass, subset counts, and patient separation. `git diff --check` passed.

Stopped after EfficientNet-B0 verification. Full-dataset training was not run.
