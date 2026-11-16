MobileNetV3-Small CUDA smoke test completed successfully on the NVIDIA GeForce GTX 1080 Ti, using PyTorch 2.12.1+cu126 (CUDA runtime 12.6), Python 3.11, seed 42, batch size 8, zero data-loading workers, and two epochs. The official torchvision ImageNet pretrained weights were used, with the classifier replaced by a five-output linear layer. All parameters were trained with AdamW (learning rate 0.0001, weight decay 0.0001).

Working directory: `D:\Grad School\AppliedML\CheXpert\ConceptKD-CXR`.

The exact PowerShell command, including cache environment variables and log capture, is saved in `command.txt`:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) 'outputs\mobilenetv3_small_smoke_seed42_80_20\torch_cache'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs\mobilenetv3_small_smoke_seed42_80_20\matplotlib_cache'
& '..\.venv\Scripts\python.exe' -u train_student.py --data-dir 'D:\Grad School\AppliedML\CheXpert\data' --train-csv 'subsets\target_only_seed42\train_80.csv' --valid-csv 'subsets\target_only_seed42\validation_20.csv' --output-dir 'outputs\mobilenetv3_small_smoke_seed42_80_20' --device cuda --epochs 2 --batch-size 8 --num-workers 0 --seed 42 --lr 0.0001 2>&1 | Tee-Object -FilePath 'outputs\mobilenetv3_small_smoke_seed42_80_20\training.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

The saved CSVs were used directly in their saved row order before DataLoader training shuffle. There was no call to `choose_pilot`. Training used 80 images from 80 patients; validation used 20 images from 20 other patients, with zero patient overlap. Both subsets originate from the original training CSV. The original validation CSV was not used.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Train images / batches | Validation images / batches |
|---|---:|---:|---:|---:|---|---|
| 1 | 0.6709233424556789 | 0.582089552238806 | 0.6136322581038183 | 0.7551020408163265 | 80 / 10 | 20 / 3 |
| 2 | 0.5874797955377778 | 0.7313432835820896 | 0.5660457562427131 | 0.7142857142857143 | 80 / 10 | 20 / 3 |

No images were dropped. Each validation epoch includes a final batch of four images. The trainer counts images and raises an error if a loader fails to process its entire dataset. There are 201 observed training labels and 49 observed validation labels per epoch. Correct-label counts were 117/201 and 37/49 in epoch 1; 147/201 and 35/49 in epoch 2.

Only exact 0 and 1 labels are observed. Uncertain -1 and blank/NaN labels receive mask zero and contribute to neither loss nor accuracy; their placeholder target zero is not treated as a definite negative. The loss is binary cross entropy with logits, averaged over observed labels. Epoch loss is weighted by each batch's observed-label count. Accuracy is the fraction of observed binary labels predicted correctly after sigmoid with threshold >= 0.5, pooled across findings and images. It is not per-image accuracy. Training metrics aggregate the batches while weights are being updated; validation metrics use the model in evaluation mode at the end of the epoch.

Preprocessing is grayscale-to-RGB conversion, resize to 224 x 224, conversion to tensor, and ImageNet normalization (mean 0.485/0.456/0.406, standard deviation 0.229/0.224/0.225).

Label/output order: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

`student_best.pth` was selected by lowest validation loss, which occurred in epoch 2. Both `student_best.pth` and `student_last.pth` contain the complete model state dictionary plus architecture, label order, epoch, preprocessing, source CSV paths, split counts, seed, environment, and run settings. They can be loaded with `torch.load(..., weights_only=True)` and reconstructed using `torchvision.models.mobilenet_v3_small(weights=None)` with a five-output classifier. `student_weights.pth` contains the standalone final state dictionary; it exactly matches the final checkpoint's state dictionary. These are model checkpoints, not optimizer-resume checkpoints.

Reload verification ran in a separate Python process using `scripts/verify_smoke_checkpoint.py`. Each checkpoint was strictly loaded into its own fresh MobileNetV3-Small instance without downloading pretrained weights. Both models ran inference on all 100 subset images on CUDA, in evaluation/inference mode. Each produced an aggregate output shape of (100, 5), with all 500 logits finite. Both checkpoints were from epoch 2; their logits ranged from -0.7260240316390991 to 1.2889432907104492. The raw weights also matched the final checkpoint tensor-for-tensor. Details are in `reload_test.json`.

Artifacts in this folder:

- `student_best.pth`, `student_last.pth`: reloadable model checkpoints.
- `student_weights.pth`: standalone final model weights.
- `history.csv`: full-precision per-epoch metrics, image/batch counts, and observed/correct-label counts.
- `learning_curves.png`: visually checked train/validation loss and accuracy graphs.
- `run_config.json`: architecture, label order, preprocessing, run settings, and label/accuracy policies.
- `reload_test.json`: fresh-model inference verification results.
- `train_used.csv`, `valid_used.csv`: exact byte copies of the provided subset CSVs.
- `training.log`, `command.txt`, `SMOKE_TEST_REPORT.md`: execution log, command, and this report.
- `torch_cache`: cached official pretrained initialization weights; `matplotlib_cache`: local plot font cache.

Code changes: paired optional `--train-csv`/`--valid-csv` flags, exact frontal CSV loading and sampler bypass, optional `--device cuda` enforcement, actual epoch coverage/finite-output checks, richer checkpoint metadata and raw weights, and full metric persistence. With neither CSV override, the original default CSV locations, 100-image sampler, seed, optimizer, model, batch size, epoch count, and automatic device selection remain in effect. A lone CSV override is rejected. Focused checks passed for paired/default arguments, exact CSV row order and sampler bypass, masked-label gradients, partial-batch counting, and dropped-image detection. `git diff --check` passed.

The environment was missing pandas and matplotlib; the repository-pinned pandas 3.0.3 and matplotlib 3.11.0 were installed without changing requirements. No other models or full-dataset training were run.
