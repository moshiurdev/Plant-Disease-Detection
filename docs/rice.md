# Rice transfer: does a PlantVillage start help on Bangladeshi rice? (step 5)

Dhan-Shomadhan has 1,106 phone photos of rice leaves from Bangladesh in five
diseases (brown spot, leaf scald, rice blast, rice tungro, sheath blight).
769 are on a white sheet and 337 in the field. PlantVillage has no rice at
all. The question: when only a few rice photos are labelled, is the step-2
PlantVillage model (EfficientNet-B0) a better starting point than plain
ImageNet weights?

Numbers come from `scripts/run_step5.py` and `scripts/summarise_step5.py`
and are saved in [results/step5_rice.json](../results/step5_rice.json).

## Data preparation

**Cache.** The originals are 8-megapixel photos. `scripts/cache_rice.py`
turns each upright, rotates it to portrait and saves it 224 px wide (about
224 × 477). Training uses a 256 × 128 (tall) input, since the leaves are long
and thin.

**Repeated shots.** Two photos of the same leaf or plant must not end up in
both train and test. `scripts/rice_pairs.py` matches ORB keypoints between
every two photos with the same background:

- **Field photos:** pairs with different diseases never reach more than 16
  inliers, while repeated shots of one plant reach hundreds. A pair is linked
  at ≥ 20 inliers, or ≥ 15 if both photos have the same label. This links 23
  pairs into 16 groups (35 photos, largest group 4). No group mixes labels.
- **White-background photos:** keypoint matching does not work here. Rice
  leaves have long parallel veins, which give many false matches. Pairs with
  *different* diseases reach more inliers than same-disease pairs (169 vs 8
  pairs at ≥ 50 inliers), even with keypoints restricted to leaf pixels. The
  top pairs, checked by eye, are different leaves. Thumbnail correlation finds
  no near copies either: no pair reaches 0.98, and same- and different-label
  pairs score alike. A look at consecutively numbered photos also showed
  different leaves. So white photos are not linked. A leaf photographed twice
  in different positions cannot be ruled out.

**Splits** (`scripts/make_rice_splits.py`): five repeats, each a different
grouped split (StratifiedGroupKFold by label × background, seed = repeat):
about 64% pool (708 images), 16% val (177) and 20% test (221: 154 white,
67 field). For each repeat, k training images per class are drawn from the
pool (at most one per group), with the repeat number as seed. Every result
below is mean ± std over the 5 repeats, so it covers both split and training
noise.

## Training

`scripts/finetune_rice.py`: 300 steps of AdamW (batch 32, weight decay 0.05,
warmup then cosine, label smoothing 0.1, crop/flip/colour augmentation). Val
macro-F1 is checked every 25 steps and the best checkpoint is kept, then
scored once on the repeat's test images. For the PlantVillage start, the
38-class head is replaced by a new 5-class head. Train and val use both
backgrounds.

**Learning rate**, chosen on val (k = 10, repeat 0):

| Start | 1e-4 | 3e-4 | 1e-3 |
|---|---:|---:|---:|
| ImageNet | 0.483 | 0.490 | **0.509** |
| PlantVillage | 0.536 | 0.559 | **0.589** |

1e-3 is used for both. As in step 4 it is the largest value tried.

## Results

| k / class | Train images | Start | Val macro-F1 | Test acc | Test macro-F1 | Test acc, white | Test acc, field |
|---:|---:|---|---|---|---|---|---|
| 10 | 50 | ImageNet | 0.527 ± 0.027 | 0.519 ± 0.030 | 0.505 ± 0.030 | 0.565 ± 0.054 | 0.416 ± 0.044 |
| 10 | 50 | PlantVillage | 0.576 ± 0.029 | 0.553 ± 0.021 | 0.536 ± 0.024 | 0.619 ± 0.019 | 0.401 ± 0.035 |
| 25 | 125 | ImageNet | 0.622 ± 0.044 | 0.625 ± 0.063 | 0.606 ± 0.065 | 0.695 ± 0.056 | 0.466 ± 0.081 |
| 25 | 125 | PlantVillage | 0.641 ± 0.050 | 0.635 ± 0.026 | 0.620 ± 0.030 | 0.712 ± 0.031 | 0.460 ± 0.053 |
| 50 | 250 | ImageNet | 0.672 ± 0.024 | 0.696 ± 0.047 | 0.678 ± 0.047 | 0.770 ± 0.044 | 0.526 ± 0.058 |
| 50 | 250 | PlantVillage | 0.704 ± 0.025 | 0.705 ± 0.027 | 0.691 ± 0.027 | 0.775 ± 0.027 | 0.543 ± 0.066 |
| all | ~708 | ImageNet | 0.776 ± 0.026 | 0.780 ± 0.021 | 0.761 ± 0.028 | 0.826 ± 0.026 | 0.674 ± 0.064 |
| all | ~708 | PlantVillage | 0.767 ± 0.023 | 0.775 ± 0.025 | 0.755 ± 0.028 | 0.823 ± 0.035 | 0.665 ± 0.048 |

**PlantVillage minus ImageNet, test accuracy, per repeat** (same split and
same training images, so the comparison is paired):

| k | Repeats 0–4 | Mean | Repeats where PV is better |
|---:|---|---:|---:|
| 10 | +0.054, +0.050, −0.036, +0.045, +0.054 | +0.034 | 4 / 5 |
| 25 | −0.054, −0.018, +0.045, +0.081, −0.005 | +0.010 | 2 / 5 |
| 50 | −0.023, −0.018, +0.018, +0.086, −0.018 | +0.009 | 2 / 5 |
| all | −0.023, +0.054, −0.005, −0.032, −0.018 | −0.005 | 1 / 5 |

## What this shows

- **The PlantVillage start helps only with very few labels, and only a
  little.** At 10 photos per class it is ahead on val in all 5 repeats
  (+0.01 to +0.08 macro-F1) and on test in 4 of 5 (mean +0.034 accuracy). The
  gain comes from white-background test photos (+0.054). On field photos it
  is no better (0.401 vs 0.416).
- **From 25 photos per class on, there is no clear difference.** The mean
  gaps (+0.01, +0.01, −0.005) are smaller than the spread between repeats,
  and the sign changes from repeat to repeat.
- This is a mostly negative answer to the research question: lab photos of
  14 other crops do not teach much that transfers to rice beyond what
  ImageNet already gives. Unlike PlantDoc in step 4, rice is a new crop with
  new diseases, and PlantVillage's single photo setting is also unlike
  Dhan-Shomadhan's.
- **More labels matter far more than the start point.** Accuracy rises from
  about 0.52–0.55 at 10 per class to about 0.78 with all ~140 per class, and
  has not levelled off.
- **Field photos are much harder than white-background photos** at every k
  (with all labels: about 0.67 vs 0.82), although both are in training. Field
  photos are only 30% of the pool. Step 6 looks at the backgrounds
  separately.

**Per class** (all labels, test predictions pooled over the 5 repeats):

| Class | ImageNet, white | ImageNet, field | PlantVillage, white | PlantVillage, field |
|---|---:|---:|---:|---:|
| Brown spot | 0.73 | 0.49 | 0.77 | 0.49 |
| Leaf scald | 0.66 | 0.45 | 0.67 | 0.42 |
| Rice blast | 0.87 | 0.65 | 0.87 | 0.60 |
| Rice tungro | 0.79 | 0.83 | 0.72 | 0.83 |
| Sheath blight | 0.95 | 0.91 | 0.96 | 0.95 |

The most common error for both start points, on both backgrounds, is leaf
scald predicted as rice blast. Brown spot, leaf scald and rice blast all show
as spots or streaks on a thin leaf. Sheath blight is almost always right.
By eye, each class's white photos share a framing: sheath blight photos are
closer and more diagonal, and leaf scald leaves are thin and dry. So part of
the score may come from how each class was photographed, not from the
disease. This was not tested.

## Limits

- Small data: 221 test images per repeat, 67 of them field photos. The ± values
  are the spread over 5 repeats, not confidence intervals.
- One architecture and input size, three learning rates (the best was the
  largest), 300 steps. The input size (256 × 128) was chosen for memory on
  an 8 GB laptop before any results were seen. It was not tuned.
- Possible repeated shots of the same white-background leaf could not be
  detected (see above). If any exist, all accuracies here may be slightly
  high, for both start points alike.
- The dataset has no information about farm, date or camera, so splits are
  random over photos (grouped by repeated shots), not by location or time.
