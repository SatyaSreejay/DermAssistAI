# Data folder

Image data lives here locally and is **never committed to Git**.

## Where to put HAM10000

Put the unzipped dataset anywhere under `data/raw/`. The exact layout does not
matter, because the pipeline (Phase 2) will read paths from a config file
rather than assuming a structure. A typical layout after unzipping the Kaggle
or Dataverse download looks like this:

```
data/raw/
  HAM10000_metadata.csv          # one row per image
  HAM10000_images_part_1/        # ~5,000 .jpg files
  HAM10000_images_part_2/        # ~5,000 .jpg files
```

If your copy looks different (one combined image folder, extra CSV files such
as `hmnist_*.csv`, duplicate lowercase folders), that is fine. Leave it as it
is and the validation script will report what it finds.

## What the metadata should contain

`HAM10000_metadata.csv` is expected to have these columns:

| column | meaning |
|---|---|
| `lesion_id` | identifies the physical lesion; several images can share one |
| `image_id` | file name of the image without `.jpg` |
| `dx` | diagnosis label (one of 7 classes below) |
| `dx_type` | how the diagnosis was confirmed (histopathology, follow-up, ...) |
| `age`, `sex`, `localization` | patient and body-site metadata |

The 7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

`lesion_id` matters: images of the same lesion must all go into the same
split (train, validation or test), otherwise the model is tested on near-copies
of training images and the scores look better than they really are.

## Subfolders

- `raw/` the dataset exactly as downloaded. Treated as read-only.
- `processed/` generated files such as split CSVs and cached CNN features.

## Source, license and attribution

- Official: Harvard Dataverse, doi:10.7910/DVN/DBW86T
- Mirror: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Paper to cite: Tschandl, P., Rosendahl, C. & Kittler, H. "The HAM10000
  dataset, a large collection of multi-source dermatoscopic images of common
  pigmented skin lesions." *Scientific Data* 5, 180161 (2018).

**License: still to be confirmed.** The Dataverse page could not be opened
from the build environment, so the exact terms have not been verified here.
As far as we know the dataset is released under a Creative Commons
**non-commercial** license (CC BY-NC 4.0 on Dataverse; the Kaggle mirror
shows CC BY-NC-SA 4.0). Before training, open the Dataverse page, check the
"Terms" tab, and record the exact license and version here. In practice this
means: academic, non-commercial use only, cite the paper above, and do not
redistribute the images (another reason they are not in Git).

## Using Google Drive on Colab

Keep the images in Drive (for example `MyDrive/HAM10000/`) and pass the
folder on the command line; nothing needs to be copied into this repo:

```
python -m scripts.validate_data \
  --metadata /content/drive/MyDrive/HAM10000/HAM10000_metadata.csv \
  --image-dir /content/drive/MyDrive/HAM10000
```

Reading thousands of small files directly from Drive is slow. If validation
or training feels slow, copy the zip to the Colab disk (`/content`) and unzip
it there; see `notebooks/colab_quickstart.ipynb`.

See `docs/DATA_PIPELINE.md` for how image ids map to files, what validation
checks, and how lesion-level splitting works.
