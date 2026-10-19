# DermAssist AI

Explainable seven-class skin-lesion classification on dermoscopic images
(HAM10000), comparing an end-to-end CNN with hybrid CNN + SVM and
CNN + Random Forest classifiers, with a focus on limited training data.

> **Educational and research use only.** This project is not a medical device
> and does not provide diagnoses. Its predictions must never be used to decide
> whether a skin lesion is safe or needs a doctor.

## Research question

*Can hybrid CNN–SVM and CNN–Random Forest classifiers provide competitive
classification performance compared with an end-to-end CNN, particularly when
labelled training data is limited?*

## Project status

| Phase | Content | Status |
|---|---|---|
| 1 | Plan and structure | done |
| 2 | Data validation and lesion-level splitting | implemented and reviewed; tested on synthetic data only |
| 3 | ResNet18 baseline | implemented; tested on CPU with synthetic data; no real training yet |
| 4 | CNN + SVM, CNN + Random Forest | not started |
| 5 | Limited-data experiments, evaluation, Grad-CAM | not started |
| 6 | FastAPI service and Streamlit app | not started |

No models have been trained and no results exist yet. Any numbers added to
this README later will come from files in `reports/`.

## Repository layout

```
configs/        YAML configuration (paths, split settings)
data/           local data only, git-ignored (see data/README.md)
docs/           COLAB_SETUP.md (start here for Colab), DATA_PIPELINE.md, REAL_DATA_VALIDATION.md, BASELINE.md
notebooks/      Colab notebooks (quickstart, baseline training); logic lives in src/
src/            reusable Python modules (data, models, training, ...)
scripts/        command-line entry points
app/            Streamlit app and FastAPI service (Phase 6)
tests/          pytest tests, synthetic data only
artifacts/      model checkpoints, git-ignored
reports/        generated metrics and figures, git-ignored
```

## Setup

Python 3.11 is the target version.

```bash
git clone https://github.com/SatyaSreejay/DermAssistAI.git
cd DermAssistAI
```

For Colab, follow `docs/COLAB_SETUP.md`.

### Windows (PowerShell)

```powershell
# 1. Install Python 3.11 from python.org (tick "Add python.exe to PATH").
# 2. In the project folder:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-core.txt     # enough for Phase 2
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

`requirements.txt` adds PyTorch (needed from Phase 3) and the app libraries.
On Windows without an NVIDIA GPU, `pip install -r requirements.txt` installs
the CPU build of PyTorch, which is enough for tests and inference; train on Colab.

## Phase 2: run and verify

All commands run from the project root.

**1. Run the tests** (uses generated random-noise images, no real data needed):

```bash
python -m pytest
```

Expected: all tests pass (66 at the time of writing; the 17 Phase 3 tests are
skipped automatically if PyTorch is not installed).

**2. Try the whole pipeline on synthetic data** (optional, writes to `data/synthetic`):

```bash
python -m scripts.make_synthetic_dataset --out data/synthetic
python -m scripts.validate_data --metadata data/synthetic/SYNTHETIC_metadata.csv --image-dir data/synthetic/images --processed-dir data/synthetic/processed --reports-dir data/synthetic/reports
python -m scripts.make_splits --processed-dir data/synthetic/processed --reports-dir data/synthetic/reports
```

**3. Run on your real metadata without images:**

```bash
# put HAM10000_metadata.csv in data/raw/ (or pass --metadata <path>)
python -m scripts.validate_data --metadata-only
python -m scripts.make_splits
```

Check in the output: no `ERROR` lines, the class counts, how many lesions
have more than one image, and in `reports/data/split_summary.csv` that every
class appears in train, val and test with similar percentages.

**4. Run with images (Colab or locally):**

```bash
python -m scripts.validate_data --image-dir <folder with the images> --overwrite
```

See `notebooks/colab_quickstart.ipynb` for Colab with Google Drive,
`docs/DATA_PIPELINE.md` for what every check and file means, and
`docs/REAL_DATA_VALIDATION.md` for the checklist to follow on the real data.

## Phase 3: ResNet18 baseline

Run `notebooks/colab_train_baseline.ipynb` on a Colab GPU runtime. It runs, in
order: the tests, a preflight that checks every image path
(`--check-only`), a smoke run (`--smoke`, whose numbers are not results), the
full training (switched off until you set `RUN_FULL_TRAINING = True`), and a
validation-set evaluation. Design decisions and the command-line equivalents
are in `docs/BASELINE.md`. The test split is only evaluated in Phase 5.

## Dataset and attribution

HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. "The HAM10000 dataset, a
large collection of multi-source dermatoscopic images of common pigmented
skin lesions." *Scientific Data* 5, 180161 (2018).
Source: Harvard Dataverse, doi:10.7910/DVN/DBW86T.

The dataset is released for non-commercial use under a Creative Commons
license (exact terms to be confirmed; see `data/README.md`). Images are not
included in this repository.

## Limitations (so far)

- HAM10000 has no patient identifier, so images of different lesions from
  the same patient cannot be kept together; only lesion-level grouping is possible.
- Split fractions and class shares are approximate because lesions are kept whole.
- The dataset is dominated by one class (`nv`), mostly from fair-skinned
  populations; results will not generalise to all skin types or devices.
