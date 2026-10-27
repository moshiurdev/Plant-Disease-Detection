# PlantVillage splits (step 2)

Goal: no physical leaf may appear in both training and test images. The
official leaf map covers only 76% of PlantVillage (see [data.md](data.md)), so
duplicates are also searched for in the pixels.

## Finding photos of the same leaf

1. **Candidates: thumbnail correlation** (`scripts/dedup.py`). Every image is
   reduced to a 32×32 grayscale thumbnail, mean-centred and normalised; all
   54,305² pairs are compared with chunked matrix products, also against a
   horizontally mirrored copy. 21,705 pairs have correlation ≥ 0.90.
2. **Verification: keypoint matching** (`scripts/verify_pairs.py`). For each
   candidate pair, ORB keypoints (1,000 per 256×256 image) are matched with a
   ratio test, and RANSAC fits one similarity transform. Photos of the same
   leaf share vein and lesion details, so many matches agree on one transform
   ("inliers"); different leaves give few.

Why two stages: by eye, correlation alone is not enough. Pairs above 0.99 were
all copies, but between 0.95 and 0.99 there was a mix of re-shots of the same
leaf and different leaves of the same shape (soybean especially). pHash was
worse (see data.md).

Calibration by looking at random pairs from each band:

| ORB inliers | Pairs | Different labels | What the pairs were, by eye |
|---|---:|---:|---|
| ≤ 5 | 20,540 | 4,908 | mostly different leaves |
| 6–10 | 406 | 83 | mixed: some re-shots of one leaf, some different leaves of the same crop |
| 11–20 | 233 | 15 | mostly the same leaf; the different-label pairs are different leaves |
| 21–30 | 161 | 1 | same leaf |
| > 30 | 365 | 1 | same leaf or exact copy |

**Linking rule** (`scripts/make_splits.py`): two images are joined if they
have the same label and ≥ 6 inliers, or ≥ 20 inliers, or thumbnail correlation
≥ 0.985. The rule leans towards joining too much inside a class: a wrong join
only makes a group bigger, while a missed join leaks a leaf into test.

## Groups and split

Groups = official leaf IDs (including the loader's fallback IDs, which join
`X.JPG` and `X copy.jpg`) ∪ the 1,068 linked pairs. Result: 19,596 groups,
mean 2.8 images; the largest is 156 Orange (citrus greening) images that
chain 39 leaf IDs together, an over-merge inside one class that was accepted.

Groups are split 70/15/15, stratified by label (StratifiedGroupKFold, 20 folds,
seed 0): **38,001 train / 8,146 val / 8,158 test**, all 38 classes in each.
The split is committed as `data/splits/plantvillage_grouped.csv`.

Two linked pairs have different labels (the same tomato leaf filed under two
classes); they are listed in `data/splits/plantvillage_label_conflicts.csv`.

Measured with the same groups, the **official split has 312 groups on both
sides, covering 529 official test images**. That is about 5% of the official
test set showing a leaf the model has already seen.

## Limits

- Candidates need thumbnail correlation ≥ 0.90, so two photos of the same leaf
  taken from clearly different angles or after rotation by other than a
  mirror flip are not found. For the 8 classes with no leaf map at all, such
  re-shots may still cross splits.
- Thresholds were set by eye on samples, not on a labelled set of pairs.
