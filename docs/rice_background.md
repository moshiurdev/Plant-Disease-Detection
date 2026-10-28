# Rice background shift: white sheet ↔ field (step 6)

Dhan-Shomadhan has the same five diseases photographed on a white sheet (769
photos) and in the field (337). This is a small, controlled version of the
lab → field question from steps 3–4: same crop, same diseases, same
dataset, only the setting changes. Does a model trained on one setting work
on the other?

Numbers come from `scripts/run_step6.py` and `scripts/summarise_step6.py` and
are saved in [results/step6_background.json](../results/step6_background.json).

## Setup

Same 5 repeated grouped splits, training recipe and learning rate as step 5
([rice.md](rice.md)), ImageNet start (step 5 found no real difference
between start points with all labels). Each condition trains on one
background's pool images and picks its checkpoint on **val images of the
same background**; the other background is never seen before test.

| Condition | Train images | Checkpoint chosen on |
|---|---:|---|
| White, all | ~492 | white val (~123) |
| White, 40 per class | 200 | white val |
| Field, all | ~216 | field val (~54) |
| Both, all (step 5 runs) | ~708 | val of both (~177) |

"White, 40 per class" has about as many images as the field pool, so the
white vs field comparison is not just about how much data there is.

## Results

Test accuracy (macro-F1 in brackets), mean ± std over 5 repeats:

| Trained on | White test (154 / repeat) | Field test (67 / repeat) |
|---|---|---|
| White, all | **0.788 ± 0.038** (0.761) | 0.261 ± 0.047 (0.161) |
| White, 40 per class | 0.712 ± 0.020 (0.691) | 0.258 ± 0.045 (0.177) |
| Field, all | 0.232 ± 0.073 (0.139) | **0.593 ± 0.062** (0.572) |
| Both, all | 0.826 ± 0.026 (0.803) | 0.674 ± 0.064 (0.660) |

For reference, always guessing the largest class would give 0.28–0.29 on
white test and 0.22–0.24 on field test.

## What this shows

- **A model trained on one background barely works on the other.** White →
  field gives 0.26, just above guessing the largest class. Field → white
  gives 0.23, *below* that guess. Macro-F1 (0.14–0.18) shows that most
  classes are almost never recognised.
- **The models fall back on one or two classes.** Over the 5 repeats the
  white-trained model calls 297 of 337 field test photos rice tungro or sheath
  blight; brown spot, leaf scald and rice blast are nearly never right on
  field photos (0–8%). The field-trained model calls 600 of 770 white photos
  sheath blight or brown spot.
- **Not a data-size effect.** With 200 white images (about the field pool
  size) white → field is unchanged (0.258 vs 0.261).
- **Training on both backgrounds helps both.** Compared with training on a
  single background, field accuracy rises from 0.593 to 0.674 (higher in all 5
  repeats, +0.02 to +0.13), and white accuracy from 0.788 to 0.826 (higher or
  equal in all 5 repeats). The white photos are useful for field photos when
  they are in training together, but useless alone.

**Reading.** The same disease looks different enough on a white sheet and in
a field scene that a model learns cues tied to the setting (background,
framing, light) rather than to the disease. Step 5 noted that each class's
white photos share a framing; this result fits with that, though it does not
prove which cue the model uses. It is the same pattern as PlantVillage →
PlantDoc in step 3, inside a single dataset of a single crop: **photos of the
target setting are needed in training**, and a mix of settings is better
than either alone.

## Limits

- Field val has only ~54 images per repeat, so checkpoint choice for the
  field-only model is noisy.
- One architecture, one recipe, ImageNet start only.
- 67 field and 154 white test images per repeat; ± values are the spread over
  5 repeats.
- Background removal or colour normalisation was not tried. Step 3's PlantDoc
  check suggests the background itself is not the whole story, but that was
  a different dataset.
