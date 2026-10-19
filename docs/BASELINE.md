# ResNet18 baseline (Phase 3)

**Status:** code written and tested on CPU with synthetic random-noise images
only. No real training has been run, so there are no results yet. Loading the
ImageNet weights was not tested in the build environment (the download host
is blocked there); it is exercised by the smoke run on Colab.

## What the baseline does

| Step | Choice | Why |
|---|---|---|
| Backbone | torchvision ResNet18, ImageNet weights | small (11.7 M parameters), fast on a free Colab GPU, standard reference model |
| Head | dropout 0.2 + linear layer 512 → 7 | the original 1000-class ImageNet layer is replaced for our 7 classes |
| Stage 1 | train the head only, backbone frozen (lr 1e-3, up to 10 epochs) | pretrained features are reused; only ~3.6 k weights learn, so it is fast and hard to overfit |
| Stage 2 | also fine-tune `layer4` (lr 1e-4, up to 5 epochs), starting from the best stage-1 weights | lets the high-level features adapt to dermoscopy; a smaller learning rate avoids destroying them |
| Frozen BatchNorm | kept in eval mode | frozen layers keep their ImageNet statistics instead of drifting with our batches |
| Optimiser | AdamW + cosine learning-rate decay within each stage | common, robust defaults |
| Model selection | best epoch by **validation macro-F1**, early stopping (patience 3) | macro-F1 weighs rare classes equally; the test set is never used |
| Imbalance | class-weighted cross-entropy, `balanced` weights from the training subset | rare classes (df, vasc) would otherwise be almost ignored; compare against `none` |
| Input | shorter side 256 → centre crop 224; ImageNet mean/std | matches the pretrained weights; fixed constants, so nothing is fitted on validation/test data |
| Augmentation (train only) | random resized crop (80–100 %), horizontal + vertical flips, ±20° rotation, mild colour jitter | dermoscopy images have no fixed orientation; mild colour change mimics different devices |
| Mixed precision | on GPU only | faster and lighter; ignored on CPU |
| Reproducibility | seeds for Python/NumPy/PyTorch, seeded data loaders, deterministic cuDNN | GPU runs can still differ slightly, so Phase 5 repeats runs with several seeds |

## Files

| File | Role |
|---|---|
| `configs/baseline.yaml` | every setting above |
| `src/models/resnet.py` | model, freezing stages, parameter count, model size |
| `src/preprocessing/transforms.py` | train and eval transforms |
| `src/data/dataset.py` | reads `splits.csv`, resolves image paths up front, returns tensors |
| `src/training/imbalance.py` | class weights |
| `src/training/trainer.py` | epochs, staged fine-tuning, early stopping, best-model tracking |
| `src/training/checkpoint.py` | save/load weights with class names, preprocessing, config, versions |
| `src/evaluation/metrics.py` | all metrics + confusion-matrix plot (shared with the hybrid models) |
| `scripts/train_baseline.py` | training command (`--check-only`, `--smoke`) |
| `scripts/evaluate.py` | evaluation on `val` (during development) or `test` (once, at the end) |
| `notebooks/colab_train_baseline.ipynb` | the same steps on Colab |

## How to run

On Colab, use `notebooks/colab_train_baseline.ipynb`. The command-line equivalent:

```bash
# 1. Preflight: config, split and every image path. Trains nothing.
python -m scripts.train_baseline --check-only --splits <splits.csv> --image-dir <images> --output-dir <artifacts>

# 2. Smoke run: 2 batches per epoch, 1 epoch per stage. Numbers are NOT results.
python -m scripts.train_baseline --smoke --splits <splits.csv> --image-dir <images> --output-dir <artifacts>

# 3. Full training (only after 1 and 2 look right)
python -m scripts.train_baseline --splits <splits.csv> --image-dir <images> --output-dir <artifacts>

# 4. Validation metrics of the selected checkpoint
python -m scripts.evaluate --checkpoint <artifacts>/<run>/best.pt --split val --splits <splits.csv> --image-dir <images>
```

Useful options: `--subset train_25pct` (limited-data runs), `--class-weighting none`,
`--seed 7`. Each combination gets its own run folder, and finished runs are
never overwritten without `--overwrite`.

## What gets saved

`best.pt` holds the weights plus class order, preprocessing, model and
training settings, seed, the selected stage and epoch, validation metrics, a
SHA-256 of `splits.csv` and software versions. `history.csv` has one row per
epoch with time per epoch. `train_summary.json` records data sizes, class
weights, total training time, parameter counts and model size.
`scripts.evaluate` adds accuracy, balanced accuracy, macro precision, recall and F1,
weighted F1, per-class metrics, confusion matrix, one-vs-rest ROC-AUC (only
when every class is present), log loss, inference time per image and
per-image predictions.

## Why accuracy alone is not enough here

About two thirds of HAM10000 images are `nv`. A model that always predicts
`nv` would score roughly 67 % accuracy while missing every melanoma. Macro-F1
and balanced accuracy give each class equal weight, and per-class recall
shows exactly which lesion types are missed, which matters far more in a
medical setting than the overall hit rate.

## Probabilities are not calibrated confidence

The softmax outputs are model scores. Class weighting in particular shifts
them towards rare classes. They must not be presented as calibrated
probabilities unless calibration is measured (optional Phase 5 analysis).
