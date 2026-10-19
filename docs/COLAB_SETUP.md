# Setting up DermAssist AI on Google Colab

This guide takes you from nothing to a validated dataset, a fixed split and a
tested training setup on Colab. It does **not** start full training: that
stays switched off in the notebook until you decide to run it.

You need: a Google account, the two notebooks (`notebooks/colab_quickstart.ipynb`
and `notebooks/colab_train_baseline.ipynb`, attached in the project thread and
in the repository), and `HAM10000_metadata.csv`.

## Step 1: get the code

The notebooks clone the code from
[github.com/SatyaSreejay/DermAssistAI](https://github.com/SatyaSreejay/DermAssistAI)
automatically (`REPO_URL` in section 3). This works with no setup as long as
the repository is **public**, which also suits a portfolio project.

If you keep the repository **private**, use the zip instead (this avoids
putting a GitHub token into Colab, where it could end up in printed output):

1. Download `DermAssist-AI-code.zip` from the thread.
2. Upload it, still zipped, to the top level of **My Drive**.
3. In section 3 of each notebook, set `REPO_URL = ""`.

## Step 2: put HAM10000 on Google Drive

1. In My Drive, create a folder named `HAM10000`.
2. Upload `HAM10000_metadata.csv` into it.
3. Get the images into the same folder. Pick one:

**Option A: download straight from Kaggle into Drive (nothing goes through your laptop).**
You need a Kaggle account. On Kaggle, open the
[dataset page](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
and accept its terms if asked, then go to *Settings > API > Create New Token*. This downloads `kaggle.json`,
which contains your username and key. In Colab, open the key icon
(*Secrets*) on the left and add two secrets, `KAGGLE_USERNAME` and
`KAGGLE_KEY`, with those values. Never paste them into a notebook cell. Then
run this once in a Colab cell (after mounting Drive):

```python
import os
from google.colab import userdata, drive
drive.mount("/content/drive")
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
!pip install -q kaggle
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p /content/drive/MyDrive/HAM10000
```

This saves `skin-cancer-mnist-ham10000.zip` (several GB) into your Drive
folder. It uses Drive storage (the free plan has 15 GB) and takes a while.
This route has not been tested from the build environment.

**Option B: upload the zip yourself** if you already have the archive
somewhere. Put the zip file (not the unzipped folders) into `HAM10000`.

A single zip is strongly preferred over thousands of loose images: Colab
reads one big file from Drive quickly, but thousands of small files slowly.

## Step 3: open the quickstart notebook

1. Go to [colab.research.google.com](https://colab.research.google.com),
   choose *File > Upload notebook*, and pick `colab_quickstart.ipynb`.
2. *Runtime > Change runtime type*: CPU is enough for this notebook.
3. In section 5, set the zip name, for example:

   ```python
   DRIVE_DATA = Path("/content/drive/MyDrive/HAM10000")
   ZIP_ON_DRIVE = DRIVE_DATA / "skin-cancer-mnist-ham10000.zip"
   ```

   Leave `REPO_URL` in section 3 as it is (or set it to `""` if you use the code zip).
4. *Runtime > Run all*. When asked, allow access to Google Drive.

What happens, in order: hardware check, Drive mount, code unpacked into
`/content/DermAssist-AI`, packages installed, the test suite (all tests
should pass), the image zip unpacked to Colab's fast local disk, data
validation, and the lesion-level split. Any failing step stops the
notebook with an error message.

**Check the output:** no `ERROR` lines in validation; read every `WARNING`.
With the Kaggle archive a warning about "image ids matching several files" is
expected, because that archive contains every image twice; it is harmless.
Then go through `docs/REAL_DATA_VALIDATION.md` and compare the class counts
with the published ones.

Results are saved to Drive in `My Drive/DermAssist-AI/` (`processed/splits.csv`
and `reports/`), so they survive when the Colab session ends. The notebook
refuses to overwrite an existing split, which keeps every later experiment on
the same split.

## Step 4: test the training setup (no full training)

1. Upload `colab_train_baseline.ipynb` the same way.
2. *Runtime > Change runtime type > T4 GPU* (or any GPU offered).
3. Set `DRIVE_DATA` and `ZIP_ON_DRIVE` in section 5 exactly as before.
4. *Runtime > Run all*.

This runs the tests, a **preflight** that checks every training and
validation image can be found, and a **smoke run** (2 batches per epoch, which
also downloads the ImageNet ResNet18 weights). The smoke run's numbers are
not results. Full training is skipped because `RUN_FULL_TRAINING = False`.

When the preflight and smoke run look fine and you want to train, set
`RUN_FULL_TRAINING = True` in section 9 and run that cell and the ones after it.

## Each new Colab session

Colab forgets everything except your Drive when a session ends. Just open
the notebook again and *Run all*: the code and image zip are unpacked again,
and the split and checkpoints are read from Drive.

## Updating the code

With GitHub, re-running section 3 pulls the latest version. With the zip,
when the code changes, download the new `DermAssist-AI-code.zip`, replace the
old one on Drive (delete the old file first so Drive does not rename the new
one to `DermAssist-AI-code (1).zip`), and re-run the notebook.

## Troubleshooting

| Problem | Fix |
|---|---|
| `git clone` fails | the repository is private or the URL is wrong; make it public, or use the code zip |
| `REPO_URL is empty and no code zip was found ...` | the code zip is not at `My Drive/DermAssist-AI-code.zip`; check the name and location |
| `HAM10000_metadata.csv not found` | put the CSV in the `HAM10000` folder, or make sure it is inside the image zip |
| `splits.csv not found` (training notebook) | run the quickstart notebook first |
| `... already exist and would be replaced` | an earlier run's output is protected; keep it, or add `"--overwrite"` only if replacing it is intended |
| very slow validation or training | use `ZIP_ON_DRIVE` instead of loose images on Drive |
| `CUDA available: False` in the training notebook | switch the runtime type to a GPU |
| session disconnects during training | free Colab sessions have time limits; checkpoints are saved to Drive after every improvement |
