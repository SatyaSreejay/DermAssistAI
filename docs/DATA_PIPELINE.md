# Data pipeline (Phase 2)

This page explains how the pipeline turns the HAM10000 metadata and images
into a fixed, leakage-free train/validation/test split, and how to run and
verify it.

## Overview

```
HAM10000_metadata.csv ──► load + normalise ──► validate ──► metadata_clean.csv ──► split ──► splits.csv
image folders ─────────► image index ──────────┘   │                                │
                                          validation_report.json        split_report.json
                                          class_distribution.png        split_summary.csv
                                                                        split_proportions.png
```

| Step | Code | Command |
|---|---|---|
| Load and normalise metadata | `src/data/metadata.py` | (used by the scripts) |
| Map `image_id` to files | `src/data/image_index.py` | (used by the scripts) |
| Validate | `src/data/validation.py` | `python -m scripts.validate_data` |
| Split by lesion | `src/data/splitting.py` | `python -m scripts.make_splits` |
| Synthetic test data | `src/data/synthetic.py` | `python -m scripts.make_synthetic_dataset` |

## 1. Mapping `image_id` values to image files

The metadata column `image_id` (for example `ISIC_0027419`) is the image's
file name without its extension. Different HAM10000 downloads put the files
in different folders, so the pipeline does not assume a layout:

1. Every folder listed in `paths.image_dirs` (or given with `--image-dir`) is
   searched **recursively** for `.jpg`, `.jpeg` and `.png` files (any letter case).
2. Each file is indexed by its stem: `.../part_2/ISIC_0027419.jpg` becomes `ISIC_0027419`.
3. Each metadata `image_id` is looked up in that index. Ids written with an
   extension in the CSV (`ISIC_0027419.jpg`) are normalised first.

Things to know:

- **Ambiguous ids.** The Kaggle archive contains the images twice (upper- and
  lower-case `part_1`/`part_2` folders). If an id matches several files, the
  first path in sorted order is used and the report lists it under
  `ambiguous_images`. Pointing `image_dirs` at just one copy avoids this.
- Hidden files such as macOS `._ISIC_...` resource forks are ignored.
- `splits.csv` stores **ids only, never paths**, so the same split works on
  your laptop, on Colab with Google Drive, or anywhere else. Paths are
  resolved again each time data is loaded.

## 2. What validation checks

| Check | Why it matters | Severity |
|---|---|---|
| Required columns `lesion_id`, `image_id`, `dx` exist | the pipeline cannot run without them | fatal |
| Missing values in those columns | an image with no label or lesion cannot be used safely | error, row dropped |
| Labels outside the 7 known classes | typos or a different dataset version | error, row dropped |
| Same `image_id` with different labels | contradictory ground truth | error, rows dropped |
| Same `image_id` repeated with the same label | would double-count an image | warning, extra rows dropped |
| Same `image_id` under different `lesion_id`s | the image could land in two groups and leak | error, rows dropped |
| A lesion whose images have different labels | splitting uses the lesion's most common label | warning |
| Image file missing | cannot train on it | error, row dropped |
| Image file cannot be decoded (corrupt or truncated) | would crash training | error, row dropped |
| Image file with no metadata row (orphan) | ignored, but worth knowing | warning |
| Byte-identical files under different ids | possible hidden duplicates; if they belong to different lesions they could leak | warning, grouped together |

It also reports images per class, lesions per class, how many lesions have
1, 2, 3... images, and image sizes.

Only `metadata_clean.csv` (the rows that passed) is used for splitting. The
original CSV is never modified.

**Metadata-only mode.** `--metadata-only` skips all file checks, so you can
run validation and splitting with just `HAM10000_metadata.csv`. Re-run the
full validation where the images are (for example on Colab) before training.

## 3. Lesion-level splitting (preventing leakage)

**The problem.** HAM10000 has about 10,000 images but fewer distinct lesions:
many lesions were photographed more than once. If image A of a lesion is in
training and image B of the same lesion is in the test set, the model is
tested on something it has effectively already seen, and the test score is
too optimistic. This is called *data leakage*.

**The fix.** Every image gets a *group id*, which is its `lesion_id`. All
images in a group go to the same split. If validation found byte-identical
files filed under different lesion ids, those lesions are merged into one
group as well.

**Stratification.** The classes are very imbalanced. Stratifying keeps each
class's share similar in train, validation and test, so rare classes are
present everywhere. We use scikit-learn's `StratifiedGroupKFold`, which does
both: it never splits a group, and it balances classes as well as it can.

**How the split is made.**
1. Test set: split the groups into `round(1 / test_fraction)` stratified
   folds (7 folds for 0.15) and keep the fold closest to 15 % of images.
2. Validation set: repeat on the remaining data with the matching relative
   fraction.
3. Everything else is training data.

**Limitations (honest version).**
- Fractions are approximate (for 0.15 you get roughly 14 to 15 %), because
  whole lesions move together and folds come in sizes of 1/k.
- Class shares in each split are close to, but not exactly, the overall
  shares. Very rare classes (such as `df`) have few lesions, so their shares
  vary most. `split_summary.csv` and `split_proportions.png` show the result,
  and `make_splits` prints a `WARNING` (also stored in `split_report.json`)
  for every class with fewer than 5 lesions in a split, because per-class
  metrics on so few lesions are unreliable.
- A lesion whose images carry different labels is kept whole and stratified
  by its most common label. Validation lists such lesions as a warning.
- Grouping uses `lesion_id` plus exact file duplicates only. Two different
  lesions from the same patient, or near-identical but not byte-identical
  photos with different lesion ids, cannot be detected from this metadata
  (HAM10000 has no patient id). This is a known limitation of the dataset.

**Training subsets for the limited-data experiment.** Inside the training
set, lesions of each class are shuffled once with the seed; the 10 % subset
takes the first 10 % of each class's lesions (rounded, halves up, and never
fewer than one lesion per class), the 25 % subset the first 25 %, and so on. So the subsets are nested (10 % ⊂ 25 % ⊂ 50 % ⊂ 100 %), keep
whole lesions together, keep every class present, and validation and test
stay exactly the same for every experiment. These are stored as the columns
`train_10pct`, `train_25pct`, `train_50pct`, `train_100pct` in `splits.csv`.

**Reproducibility.** The same metadata, seed and scikit-learn version give
the same split. `split_report.json` records the seed, library versions and a
SHA-256 of `metadata_clean.csv`. The scripts refuse to overwrite an existing
split unless you pass `--overwrite`, so the fixed split cannot be replaced by
accident.

## 4. Output files

| File | Contents |
|---|---|
| `reports/data/validation_report.json` | every check above, with the affected ids |
| `reports/data/class_distribution.png` | images per class |
| `data/processed/metadata_clean.csv` | metadata rows that passed validation |
| `data/processed/splits.csv` | `image_id, lesion_id, dx, group_id, split, train_10pct ... train_100pct` |
| `reports/data/split_summary.csv` | images and % per class in each split |
| `reports/data/split_report.json` | seed, sizes, lesion counts, subset sizes, provenance |
| `reports/data/split_proportions.png` | class share per split |

All of these are git-ignored and can be regenerated.
