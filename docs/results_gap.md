# Closing the lab → field gap (step 4)

Step 3 showed that the PlantVillage model (EfficientNet-B0, chosen in step 2)
gets 0.309 accuracy on PlantDoc test. Step 4 tries two fixes:

1. **Stronger appearance augmentation** while training on PlantVillage, so the
   model cannot rely on the single PlantVillage look (lighting, colour,
   framing).
2. **Few-shot fine-tuning** on k labelled PlantDoc images per class
   (k = 5, 10, 25 or all), starting from ImageNet, from the step-2 PlantVillage
   model, or from the strong-augmentation PlantVillage model.

Every choice (learning rate, checkpoint, which start point is best) is made on
a **PlantDoc validation set carved out of PlantDoc train**. PlantDoc test is
scored once per finished run and not used for any choice.

Numbers come from `scripts/run_step4.py`, `scripts/summarise_step4.py` and
`scripts/analyse_field.py` and are saved in
[results/step4_gap.json](../results/step4_gap.json) and
[results/step4_field_views.json](../results/step4_field_views.json).

## PlantDoc validation split

PlantDoc has only an official train/test split, so a validation set was made
from train (`scripts/make_plantdoc_splits.py`,
[data/splits/plantdoc_grouped.csv](../data/splits/plantdoc_grouped.csv)):

- All PlantDoc images were grouped with the same near-duplicate check as
  PlantVillage (thumbnail correlation, then keypoint matching; same linking
  rule). 82 pairs were linked.
- **35 linked pairs carry different labels**: the same photo filed under two
  classes, mostly potato/tomato early vs late blight. Those 47 train images
  were left out, because no label can be trusted for them.
- 17 train images that are copies of a test image were left out of train and
  val, so validation cannot reward memorising test photos. This check also
  found one more copy (a mirrored potato early blight photo) that step 3's
  list had missed; the test-duplicate list now has 16 test images, and the
  step-3 "clean" number was updated to 220 images.
- The remaining groups were split 80/20, stratified by class
  (StratifiedGroupKFold, seed 0): **1,822 train / 456 val**.

One class, tomato spider mites, has just 1 train image and 1 val image after
this (and none in test), so with k = 5 a run sees 136 images, not 140.

## 1. Stronger augmentation alone: no gain

The step-2 recipe was rerun with a much stronger augmentation
(`pdd.data.strong_train_transform`: random crops down to 25% of the image,
TrivialAugmentWide, strong colour jitter, grayscale, blur, random erasing).
Everything else stayed the same. Its best epoch was chosen on **PlantDoc val**,
which favours it over the step-2 model, chosen on PlantVillage val.

| Model (no PlantDoc training) | PlantDoc val macro-F1 | PlantDoc test acc [95% CI] | Test macro-F1 | Test acc, 220 clean |
|---|---:|---|---:|---:|
| Step-2 model (default augmentation) | **0.193** | **0.309 [0.250, 0.364]** | **0.300** | **0.314** |
| Strong augmentation | 0.173 | 0.284 [0.225, 0.339] | 0.271 | 0.291 |

The strong-augmentation model is worse on validation at every epoch
(0.144–0.173 against 0.193), and also on test. It still reaches 0.983
PlantVillage val macro-F1, so it did learn the lab task. **Changing how
PlantVillage photos look in training did not make the model work on field
photos.** Only one strength of augmentation was tried, so this does not rule
out every augmentation, but it agrees with step 3: the gap is not mainly
about colour or background.

## 2. Few-shot fine-tuning

`scripts/finetune.py`: k images per class from the PlantDoc train split (one
per duplicate group, chosen by the seed), 400 steps of AdamW (batch 32,
weight decay 0.05, warmup then cosine, label smoothing 0.1, step-2
augmentation, 160 px). Every 50 steps the model is scored on PlantDoc val, and
the step with the best val macro-F1 is kept. The 38 PlantVillage outputs are
kept, so scoring is the same as in step 3. A given seed draws the same
k-image subset for every start point, so start points are compared on the
same images.

**Learning rate**, chosen on val at k = 10, seed 0:

| Start | 1e-4 | 3e-4 | 1e-3 |
|---|---:|---:|---:|
| ImageNet | 0.464 | 0.471 | **0.486** |
| PlantVillage | 0.431 | 0.467 | **0.509** |

1e-3 wins for both, and is used for every run (the strong-augmentation start
uses the PlantVillage choice). It is the largest value tried, so a still
larger rate might be a little better. This was not checked.

**Results**: mean ± std over 3 seeds. The seed changes both the k-image
subset and the training order.

| Start | k / class | Train images | PlantDoc val macro-F1 | PlantDoc test acc | Test macro-F1 | Test acc, 220 clean |
|---|---:|---:|---|---|---|---|
| none (step 3) | 0 | 0 | 0.193 | 0.309 | 0.300 | 0.314 |
| ImageNet | 5 | 136 | 0.387 ± 0.010 | 0.380 ± 0.010 | 0.361 ± 0.026 | 0.382 ± 0.005 |
| PlantVillage | 5 | 136 | 0.411 ± 0.011 | 0.472 ± 0.014 | 0.460 ± 0.006 | 0.483 ± 0.019 |
| PV, strong aug | 5 | 136 | 0.417 ± 0.018 | 0.449 ± 0.037 | 0.435 ± 0.034 | 0.452 ± 0.043 |
| ImageNet | 10 | 271 | 0.498 ± 0.020 | 0.442 ± 0.031 | 0.423 ± 0.040 | 0.447 ± 0.029 |
| PlantVillage | 10 | 271 | 0.516 ± 0.019 | 0.510 ± 0.024 | 0.506 ± 0.018 | 0.520 ± 0.021 |
| PV, strong aug | 10 | 271 | 0.518 ± 0.019 | 0.489 ± 0.022 | 0.478 ± 0.024 | 0.500 ± 0.028 |
| ImageNet | 25 | 676 | 0.588 ± 0.010 | 0.532 ± 0.021 | 0.528 ± 0.023 | 0.548 ± 0.010 |
| PlantVillage | 25 | 676 | 0.622 ± 0.013 | 0.581 ± 0.007 | 0.577 ± 0.010 | 0.589 ± 0.007 |
| PV, strong aug | 25 | 676 | 0.609 ± 0.020 | 0.565 ± 0.018 | 0.559 ± 0.024 | 0.582 ± 0.024 |
| ImageNet | all | 1,822 | 0.643 ± 0.020 | 0.624 ± 0.027 | 0.621 ± 0.021 | 0.638 ± 0.017 |
| **PlantVillage** | **all** | **1,822** | **0.651 ± 0.003** | **0.654 ± 0.022** | **0.646 ± 0.023** | **0.662 ± 0.021** |
| PV, strong aug | all | 1,822 | 0.636 ± 0.019 | 0.668 ± 0.011 | 0.660 ± 0.014 | 0.677 ± 0.008 |

**What this shows**

- **A few field photos help far more than anything done to the lab data.**
  Five labelled PlantDoc photos per class take the PlantVillage model from
  0.309 to 0.472 test accuracy. All 1,822 train photos take it to 0.654.
- **Starting from PlantVillage beats starting from ImageNet** on test at every
  k, and for each of the 3 seeds (same images, paired):
  +0.08 to +0.12 accuracy at k = 5, +0.02 to +0.12 at k = 10,
  +0.03 to +0.08 at k = 25 and +0.01 to +0.04 with all images. The benefit
  shrinks as more field data is added. On validation the PlantVillage start is
  also ahead at k = 5, 10 and 25 for every seed. With all images the val
  difference is within seed noise (−0.013, +0.031, +0.007).
- **The strong-augmentation start gives no clear benefit.** On val it is level
  with the plain PlantVillage start at k = 5 and 10 and slightly behind at
  k = 25 and all. On test it is lower at k = 5–25 and higher with all images
  (0.668 vs 0.654). That last difference was not used for any choice and is
  about the size of the seed spread.
- **Selected on val:** the PlantVillage start with all PlantDoc train images
  has the best mean val macro-F1 (0.651). Its best seed on val (seed 1, val
  0.655) is the field model used later in the demo. That run gets
  **0.636 [0.572, 0.699] test accuracy** and 0.624 [0.560, 0.668] macro-F1.
  It is not the best run on test (seed 0 gets 0.678); test was not used to
  choose.
- The selected step was often the last one (400) for larger k, so longer
  training might help a little. Only 400 steps were tried.

## What changes after fine-tuning (selected model)

Same views as step 3, for the step-2 model and the selected fine-tuned model
(PlantVillage start, all images, seed 1):

| View | Step-2 model | Fine-tuned |
|---|---:|---:|
| 38-way accuracy | 0.309 | 0.636 |
| Right crop | 0.538 | 0.822 |
| Healthy vs diseased | 0.856 | 0.945 |
| Predictions of classes PlantDoc lacks | 49 / 236 | 0 / 236 |
| Accuracy when confidence > 0.9 | 0.39 (23 images) | 0.89 (71 images) |

- **Catch-all classes are gone.** Strawberry leaf scorch is no longer
  predicted. The most predicted class (tomato leaf mold) now gets 15 of 236
  photos, close to the class sizes.
- **Remaining errors are mostly between related diseases of the same crop**:
  corn Northern leaf blight vs gray leaf spot (8 errors in both directions),
  potato early vs late blight (6), and tomato bacterial spot vs Septoria leaf
  spot (4). PlantDoc's label conflicts are concentrated in early vs late
  blight, so part of that error may be label noise.
- Tomato bacterial spot (9 images) is still never right.
- **Background:** after fine-tuning, plain-background photos score 0.724
  [0.632, 0.816] and real scenes 0.618 [0.529, 0.691]. Before, they scored
  about the same. The intervals overlap, so this is a hint, not a finding:
  once the "PlantVillage look" problem is reduced, clutter may start to matter.
- Confidence is now more useful: above 0.9, the model is right 89% of the
  time (step 3: 39%). The probabilities are still not calibrated.

## Cost: the fine-tuned model forgets PlantVillage

The fine-tuned model was also scored on PlantVillage test (8,158 lab photos):

| Model | PV test acc, 28 classes PlantDoc has (5,753 images) | PV test acc, 10 other classes (2,405 images) | PV test acc, all |
|---|---:|---:|---:|
| Step-2 model | 0.996 | 0.999 | 0.997 |
| Fine-tuned (PV start, all, seed 1) | 0.763 | 0.028 | 0.546 |

Fine-tuning on PlantDoc alone teaches the model that the 10 classes PlantDoc
lacks never occur, and it also loses a quarter of its lab accuracy on the
shared classes. So the fine-tuned model is a **28-class field model**, not a
better general model. Mixing PlantVillage images into fine-tuning would
probably limit this, but it was not tried. The demo offers the two models
separately and says which is which.

## Limits

- PlantDoc is small (456 val, 236 test images, 4–12 test images per class),
  web-collected and has label noise (35 conflicting duplicate pairs found).
  Test accuracy intervals are about ±0.06.
- Three seeds per condition. The std values show the spread over three runs,
  not a confidence interval.
- One architecture (EfficientNet-B0, 160 px), one strength of strong
  augmentation, three learning rates and one training length were tried.
- **The gap is narrowed, not closed**: 0.997 on lab photos against about 0.65
  on field photos, even with every PlantDoc train image.
- Background removal was not tried. Step 3 found that plain-background photos
  were not easier for the lab model, so it was not expected to help much.
