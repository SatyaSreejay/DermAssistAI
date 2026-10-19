# Validating the real HAM10000 data

**Status: not done yet.** The pipeline has only been run on synthetic data.
Follow these steps once you can reach the real metadata and images, and keep
the generated reports as evidence. Do not report the real data as validated
until every check below has passed.

## Step 1: metadata only (laptop, no images needed)

```bash
python -m scripts.validate_data --metadata path/to/HAM10000_metadata.csv --metadata-only
```

Check the console output and `reports/data/validation_report.json`:

| Check | What you want to see |
|---|---|
| `ERROR` lines | none |
| `n_rows` and `n_unique_images` | equal (one row per image) |
| `unknown_labels`, `missing_values` | empty / all zero |
| `conflicting_image_labels`, `conflicting_image_lesions` | empty |
| `lesions_with_conflicting_labels` | ideally empty; if not, list them in your report |
| `n_unique_lesions` | clearly smaller than `n_unique_images` (many lesions have several photos) |
| `class_counts_images` | compare with the published counts below |

**Published class counts to compare against** (from Tschandl et al., 2018,
Table 1; these come from the paper, not from our run): akiec 327, bcc 514,
bkl 1,099, df 115, mel 1,113, nv 6,705, vasc 142; total 10,015. If your
counts differ, your copy is a different version or is incomplete. Find out
which before going further.

Then make the split (metadata only is enough for this):

```bash
python -m scripts.make_splits
```

| Check | What you want to see |
|---|---|
| `split_summary.csv` | every class has images in train, val and test; percentages per class similar across splits |
| `image_fraction` in `split_report.json` | roughly 0.70 / 0.15 / 0.15 |
| `WARNING` lines | read each one; `df` and `vasc` have few lesions, so a warning about few lesions in val/test is possible and must be mentioned when reporting per-class results |
| leakage | built in: the script raises an error if any lesion appears in two splits |

## Step 2: with images (Colab)

Open `notebooks/colab_quickstart.ipynb`, set `DRIVE_DATA` (and `ZIP_ON_DRIVE`
if you use the zip) and run all cells. The notebook stops on the first
failing command.

| Check | What you want to see |
|---|---|
| `missing_images` | empty; otherwise `image_dirs` points at the wrong folder or the copy is incomplete |
| `unreadable_images` | empty |
| `orphan_images` | empty, or explained (e.g. extra files in the folder) |
| `ambiguous_images` | expected with the full Kaggle archive (images are stored twice); harmless if the copies are identical |
| `image_sizes` | HAM10000 images are expected to be 600x450; note any others |
| `duplicate_file_groups` | report the number; groups spanning lesions are merged automatically |

If you already made the split on your laptop in step 1, copy `splits.csv`
and its reports to `DRIVE_OUT` instead of making a new one, so every run uses
the same split. If image validation drops rows that the laptop split
contains, re-make the split with `--overwrite` **before** any training and
use only that version from then on.

## Step 3: record what you found

Write the following in your project report (numbers taken from the JSON
files, not typed from memory): rows, images, lesions, class counts,
duplicates found, rows dropped and why, split sizes, any rare-class
warnings, the seed, and the `metadata_clean_sha256` from `split_report.json`.
Also record the dataset license as shown on the Dataverse "Terms" tab.
