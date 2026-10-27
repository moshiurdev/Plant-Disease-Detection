# PlantVillage baselines (step 2)

Three ImageNet-pretrained backbones were fine-tuned on the leaf-grouped
PlantVillage split ([splits.md](splits.md)): 38,001 train, 8,146 validation
and 8,158 test images in 38 classes.

## Setup

Every model used the same recipe (`scripts/run_step2.sh`):

- 160 × 160 input, 6 epochs, batch 32;
- AdamW with lr 5e-4 and weight decay 0.05;
- half an epoch of warm-up, then cosine decay;
- label smoothing 0.1;
- augmentation: random resized crop (scale 0.5–1), horizontal and vertical flips, colour jitter;
- seed 0, Apple M1 GPU (MPS).

The checkpoint kept for each model is the epoch with the best validation
macro-F1. The model chosen for later steps is the one with the best
validation macro-F1. The test split was used once per model, after training
had finished (`scripts/evaluate.py --set pv_test`). The 95% intervals come from
a 1,000-sample bootstrap over test images.

## Results

| Model | Params | Train speed | Val macro-F1 (errors) | Test accuracy [95% CI] | Test macro-F1 [95% CI] | Test errors |
|---|---:|---:|---:|---|---|---:|
| ResNet-18 | 11.2 M | 150 img/s | 0.9946 (28) | 0.9940 [0.9923, 0.9956] | 0.9913 [0.9884, 0.9940] | 49 / 8,158 |
| MobileNetV3-Large | 4.3 M | 103 img/s | 0.9951 (25) | 0.9950 [0.9934, 0.9964] | 0.9932 [0.9908, 0.9953] | 41 / 8,158 |
| **EfficientNet-B0** (selected) | 4.1 M | 51 img/s | **0.9958 (24)** | **0.9972 [0.9960, 0.9983]** | **0.9956 [0.9936, 0.9975]** | **23 / 8,158** |

Training time for 6 epochs was about 25 min (ResNet-18), 37 min (MobileNetV3)
and 77 min (EfficientNet-B0). Per-epoch numbers are in `runs/pv_<arch>/history.json`.
That folder is not committed; the table above is copied from
`runs/pv_<arch>/eval_pv_test.json`.

**Selection.** On validation, EfficientNet-B0 had 24 errors, against 25 for
MobileNetV3 and 28 for ResNet-18. That is close to a tie. The test ranking is
the same, and the gap is larger there (23 / 41 / 49 errors).

## Where the errors are (EfficientNet-B0, test)

| True class | Predicted | Count |
|---|---|---:|
| Corn Northern leaf blight | Corn Cercospora / gray leaf spot | 5 |
| Corn Cercospora / gray leaf spot | Corn Northern leaf blight | 4 |
| Tomato bacterial spot | Tomato late blight | 3 |
| Potato late blight | Tomato late blight | 2 |
| Pepper bacterial spot | Tomato Septoria leaf spot | 2 |

The two corn leaf-spot diseases are confused in both directions. They are the
two classes with the lowest recall: 0.947 for gray leaf spot (76 test images)
and 0.966 for Northern leaf blight (147). The other two models make the same
corn confusion most often. They also confuse late blight across crops
(potato ↔ tomato). This is plausible because the disease is the same and the
leaves look alike.

## What these numbers do and do not mean

- **The task is easy.** Every photo is a single leaf on a plain background,
  taken under similar conditions. Above 99% accuracy says the lab photos are
  separable. It does not show that the model would work on a farmer's photo.
  Step 3 measures that on PlantDoc.
- **The grouped split is stricter than the official one.** The official split
  shows about 5% of its test images to the model during training (see
  splits.md). Published PlantVillage results (above 99% since Mohanty et al.
  2016) use other splits, so they cannot be compared one-to-one with this table.
  No run on the official split was made.
- **One seed, short schedule.** Each model was trained once (seed 0). The best
  validation epoch was the last epoch for all three, so longer training might
  still help. The ranking between models with fewer than 50 test errors each
  would need repeated seeds to be certain. The EfficientNet-B0 and ResNet-18
  accuracy intervals do not overlap, but EfficientNet-B0 and MobileNetV3 do.
- **160 px input.** This was chosen to fit an 8 GB M1 while other apps were
  running. Accuracy at 224 px was not measured.
