# Data check (step 1)

Checked on 2026-10-27 with `scripts/check_data.py`, `scripts/near_duplicates.py`
and a visual check of every flagged PlantDoc and Dhan-Shomadhan pair. Raw
numbers are in [`data/meta/data_check.json`](../data/meta/data_check.json); its
near-duplicate counts are unverified pHash candidates, and this page gives the
numbers after the visual check.

## Downloads and integrity

| Dataset | Source and version | Integrity check |
|---|---|---|
| PlantVillage | Hugging Face `mohanty/PlantVillage`, `data.zip` (2,184,723,441 bytes) | SHA-256 `fba30c6a…41745e84` matches the Hugging Face LFS record. Only `raw/color/` is extracted (54,305 images, 850 MB). |
| PlantDoc | GitHub `pratikkayal/PlantDoc-Dataset`, commit `5467f60` (2 May 2021) | Exported from git objects (`scripts/export_plantdoc.py`): 2,578 images. A plain checkout on macOS gives only 2,572, because 6 pairs of file names differ only in letter case (e.g. `CAR1.jpg` / `car1.jpg`); the second file of each pair is saved with a `__2` suffix. |
| Dhan-Shomadhan | Mendeley Data, DOI 10.17632/znsxdctwtt.1 | All 1,106 files match the SHA-256 published by the Mendeley API. |

Every image in all three datasets decodes without error.

## Counts

**PlantVillage (color):** 38 classes, 54,305 images; official split 43,596 train / 10,709 test.

**PlantDoc (classification):** 28 classes, 2,342 train / 236 test. The test set
has 27 classes (no `Tomato two spotted spider mites leaf`) and only 4–12 images
per class, so a single test image is worth about 0.4 percentage points of
accuracy, and per-class numbers are very noisy. All 28 classes map onto
PlantVillage classes (`configs/plantdoc_to_plantvillage.csv`). Smallest image
side: 69 px.

**Dhan-Shomadhan:** 5 classes, 1,106 images, mostly 1952×4160 phone photos.

| Class | White background | Field background |
|---|---:|---:|
| Brown spot | 90 | 49 |
| Leaf scald | 143 | 74 |
| Rice blast | 198 | 74 |
| Rice tungro | 119 | 76 |
| Sheath blight | 219 | 64 |
| **Total** | **769** | **337** |

The Mendeley folder names contain typos ("Leaf Scaled", "Browon Spot", "Rice
Turgro", "Shath Blight"); they are mapped to clean names in the download script.

## Leakage findings

### PlantVillage: the official split is only partly leaf-grouped

- 13,194 of 54,305 images (24%) have **no entry in the official leaf map**. For
  eight classes there is no leaf information at all (all four corn classes,
  Grape healthy, Squash powdery mildew, Tomato mosaic virus, Tomato target spot).
  For those images the official split is effectively random per image.
- 227 groups appear on both sides of the official split (533 images, 243 of
  them in test). Most are files named `X.JPG` and `X copy.jpg`.
- 21 groups of byte-identical files; 5 of them cross train/test.
- Perceptual hashing does **not** work as a duplicate detector on PlantVillage:
  photos of different leaves on the same grey background in the same pose get
  near-identical hashes. At Hamming distance ≤ 2 (of 64) the sampled pairs are
  a mix of true copies and different leaves; at 3–6 most are different leaves,
  and many have different labels. Chaining at ≤ 6 merges 3,486 unrelated images
  into one group. So near-duplicate grouping for PlantVillage needs a stricter
  pixel-level comparison; this is the first task of step 2.

**Consequence:** the official PlantVillage test score is a slightly optimistic
in-domain number. Step 2 builds its own splits from the leaf map plus verified
duplicate groups, and reports the official split only for comparison with
published work.

### PlantDoc: test copies of train images, some with different labels

15 of the 236 test images (6.4%) are the same picture as a train image (14
have an identical perceptual hash and look identical, 11 of them byte-identical
files; 1 is the same drawing with parts edited out). **9 of these 15 have a different label in train than in test**, for
example the same photo labelled "Corn Gray leaf spot" in one split and "Corn
leaf blight" in the other, or Potato early blight vs late blight. The verified
list is in `configs/plantdoc_test_train_duplicates.csv`.

*Update (step 4):* the thumbnail-correlation + ORB check from step 2, run on
all PlantDoc images, found every one of these pairs plus one more, a mirrored
copy that pHash cannot match (potato early blight, same label). That makes
**16 of 236 test images** (6.8%). The same check found 35 linked pairs that are
one photo filed under two classes, 25 of them inside PlantDoc train (mostly
potato/tomato early vs late blight); see [results_gap.md](results_gap.md).

**Consequence:** PlantDoc results are reported on the full test set (for
comparison with other work) and on the 220 test images that are not copies of
train images. Label noise in PlantDoc limits how high any model can score;
this is stated next to every PlantDoc number.

Other PlantDoc quality notes from the visual check: images are collected from
the web, some carry stock-photo watermarks, and at least one train image in
"Apple leaf" is a photo of an apple fruit.

### PlantDoc vs PlantVillage: no shared images

0 byte-identical files. 16 PlantDoc images had a PlantVillage image within
Hamming distance 6; all 16 were checked by eye and are different leaves. So
the lab → field test is not inflated by PlantVillage images inside PlantDoc.

### Dhan-Shomadhan: burst shots, no white/field overlap found

0 byte-identical files. Of the 46 pairs within Hamming distance 6, most are
different leaves that look alike (one leaf on a white sheet), but a few field
pairs are clearly consecutive shots of the same plant. No white-background image
matched a field image. The file names carry no leaf or plant ID, so rice splits
(step 5) will group near-identical shots with a stricter similarity measure
and keep each group in one fold.

## What this means for the plan

- No dataset blocks the plan. The three questions stay as written.
- PlantDoc test is small and noisy, so lab → field results will be given with
  bootstrap confidence intervals, on both the full and the de-duplicated test set.
- PlantVillage splits and rice folds need a verified duplicate grouping before
  any training. That is done at the start of step 2.
