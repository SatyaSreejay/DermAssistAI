# DermAssist AI: Phase 1 plan

Status: proposal, awaiting approval. Nothing below has been run or measured yet.

## 1. What was inspected

- **Project folder**: was empty. No existing files were
  touched. The skeleton below was created fresh.
- **Git**: no GitHub repository is attached to this project yet.
- **This cloud container** (where I work):
  - Python 3.13 default, with 3.11 and 3.12 also installed (we will target **3.11**)
  - 4 CPU cores, 15 GB RAM, **no GPU** (`nvidia-smi` not present)
  - PyTorch is not installed yet
- **What that means**: the container can run the data pipeline, tests,
  feature extraction and SVM/RF training comfortably. Training only the ResNet18
  head on CPU is feasible but slow-ish; fine-tuning backbone layers and the
  repeated limited-data runs are much faster on a GPU (your machine, Colab or
  Kaggle). All code will auto-detect CUDA and fall back to CPU.

## 2. Directory structure (created)

```
DermAssist-AI/
  README.md              (Phase 2+)
  requirements.txt       (Phase 2)
  .gitignore             (Phase 2: data/, artifacts/, *.pt, *.npz, .env ...)
  PHASE1_PLAN.md         this file
  configs/               YAML configs: paths, splits, model hyperparameters
  data/
    README.md            where to put HAM10000 (written)
    raw/                 dataset as downloaded, read-only
    processed/           split CSVs, cached CNN features
  notebooks/             exploration only; logic lives in src/
  src/
    data/                metadata loading, validation, grouped splits, Dataset class
    preprocessing/       transforms, normalisation stats (fit on train only)
    models/              ResNet18 classifier, feature extractor, SVM/RF wrappers
    training/            training loops, seeds, checkpoint saving
    evaluation/          metrics, confusion matrix, ROC-AUC, timing, plots
    explainability/      Grad-CAM
    inference/           shared predict() used by both API and Streamlit
  scripts/               CLI entry points (validate_data, make_splits, train_*, evaluate)
  app/                   streamlit_app.py, api.py
  tests/                 pytest
  artifacts/             checkpoints + metadata (git-ignored, .gitkeep only)
  reports/               metrics CSV/JSON and figures
```

Small adaptations from your sketch, with reasons: `data/raw` vs `data/processed`
keeps the original download untouched; `src/inference/` holds one predict
function so the API and Streamlit app cannot drift apart.

## 3. Implementation plan by phase

**Phase 2: data pipeline**
- `scripts/validate_data.py`: find images, match to metadata, report missing
  files/labels, unreadable images, exact duplicates (file hash), images per
  `lesion_id`, class counts. Writes a JSON report and a class-distribution plot.
- Leakage-aware split with `StratifiedGroupKFold` grouped by `lesion_id`
  (e.g. ~70/15/15 train/val/test). Report how close each split's class mix is to
  the full data, since grouping and stratifying cannot both be perfect.
- Splits saved once to `data/processed/splits.csv` and reused by every model.
- Limited-data subsets (10/25/50%) drawn from **train only**, grouped by lesion,
  stratified, fixed seeds. Validation and test never change.
- Tests run on a tiny synthetic dataset, so they never need the real images.

**Phase 3: ResNet18 baseline**
- ImageNet-pretrained ResNet18, new 7-class head, frozen backbone first,
  optional fine-tune of `layer4` afterwards.
- Class imbalance (nv is roughly two thirds of HAM10000): compare class-weighted
  loss against plain loss on validation; the choice is justified by macro-F1, not
  accuracy.
- Checkpoint saved with class mapping, transforms, config and seed.

**Phase 4: hybrids**
- Extract 512-d pooled features from the same frozen ResNet18 once and cache
  them per split. `StandardScaler` fitted on train features only.
- SVM (RBF, `class_weight="balanced"` option) and Random Forest, small grids
  tuned on the validation split or grouped CV on train.

**Phase 5: research evaluation**
- All metrics you listed, per model, per data fraction, over 3 seeds where time
  allows (mean ± std). Confusion matrices, per-class error analysis,
  Grad-CAM gallery with limitations written up, optional reliability diagram/ECE.
- Test set evaluated once per final model, after all choices are made on validation.

**Phase 6: app**
- FastAPI `/health` and `/predict` (file type/size validation, clear errors),
  Streamlit upload, model choice, probabilities, Grad-CAM for the CNN,
  disclaimer, graceful handling of missing checkpoints. Tests + Windows guide.

## 4. Dataset, storage and dependency needs

- **Dataset**: HAM10000, about 10,015 JPEGs (600x450) plus `HAM10000_metadata.csv`.
  Roughly 2 to 3 GB unzipped (the Kaggle zip is larger because it bundles extra
  copies and CSV versions; exact size to be confirmed when you add it).
- **Generated data**: cached features are small (10k x 512 floats is about 20 MB);
  ResNet18 checkpoints are about 45 MB each.
- **Python packages** (Phase 2 will pin versions in `requirements.txt`):
  torch, torchvision, scikit-learn, numpy, pandas, pillow, matplotlib, seaborn,
  pyyaml, tqdm, grad-cam (`pytorch-grad-cam`) or a small own implementation,
  fastapi, uvicorn, python-multipart, streamlit, requests, pytest.
- **Pretrained weights**: torchvision downloads ResNet18 ImageNet weights
  (about 45 MB) on first use.
- **Git**: data, artifacts and checkpoints excluded via `.gitignore`.
