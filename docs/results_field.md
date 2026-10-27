# Lab → field: PlantVillage models on PlantDoc (step 3)

The three step-2 models (trained only on PlantVillage, nothing changed) were
scored on the official PlantDoc test set: 236 images in 27 classes, each
mapped to its PlantVillage class
([configs/plantdoc_to_plantvillage.csv](../configs/plantdoc_to_plantvillage.csv)).
EfficientNet-B0 is the model chosen on PlantVillage validation in step 2. The
other two are shown for comparison; no choice was made on PlantDoc.

Numbers come from `scripts/evaluate.py --set plantdoc` and
`scripts/analyse_field.py` and are saved in
[results/step3_plantdoc.json](../results/step3_plantdoc.json). 95% intervals
come from a 1,000-sample bootstrap over test images. With 4–12 images per
class, intervals are wide, and per-class numbers are rough.

## Headline

| Model | PlantVillage test acc | PlantDoc test acc [95% CI] | PlantDoc macro-F1 [95% CI] |
|---|---:|---|---|
| ResNet-18 | 0.994 | 0.284 [0.229, 0.343] | 0.281 [0.221, 0.322] |
| MobileNetV3-Large | 0.995 | 0.347 [0.288, 0.407] | 0.327 [0.263, 0.366] |
| **EfficientNet-B0** (selected) | **0.997** | **0.309 [0.250, 0.364]** | **0.300 [0.239, 0.339]** |

Accuracy falls from above 99% to about 30%. For reference, guessing the
largest class would give 5.1% (12 / 236), and chance over 38 classes is 2.6%.
So the models keep some signal, but most field photos are classified wrongly.
The PlantVillage ranking does not carry over. MobileNetV3 is highest on
PlantDoc, but all three intervals overlap, so the field ranking is not clear
either.

Removing the 16 test images that duplicate PlantDoc train images
([docs/data.md](data.md)) changes little: EfficientNet-B0 gets 0.314 on the
remaining 220.

## Easier versions of the question (EfficientNet-B0)

These views were defined before looking at the PlantDoc results. None of them
is tuned on test.

| View | What counts as right | Accuracy [95% CI] |
|---|---|---|
| 38-way | exact PlantVillage class | 0.309 [0.250, 0.364] |
| Restricted | exact class, but only the 28 classes PlantDoc has can be predicted | 0.352 [0.288, 0.411] |
| Crop | right plant (apple, corn, tomato…), any disease | 0.538 [0.466, 0.602] |
| Health | healthy vs diseased | 0.856 [0.809, 0.898] |

- 49 of 236 predictions (21%) are classes that do not exist in PlantDoc.
  Removing those options helps only a little (0.309 → 0.352).
- The model names the right crop only about half the time. In PlantVillage
  every crop's photos share a look, and the model seems to rely on that look
  rather than on leaf shape.
- Healthy vs diseased holds up best: 0.856, against 0.619 for always saying
  "diseased".

## Where it breaks

**One class collects many errors.** "Strawberry leaf scorch" is the most
predicted class, 31 times, and PlantDoc has no such images. By eye, the field
photos sent there are mostly leaves with dark spots or reddish colour, of any
crop: apple scab, tomato early blight, an autumn-red blueberry leaf. In
PlantVillage, leaf scorch is the class with purple-brown spots and red
margins. The model appears to have learned "spotty or reddish ⇒ leaf scorch"
rather than what a strawberry leaf looks like. Each model has such
catch-all classes, but not the same ones:
- ResNet-18: tomato late blight (35), tomato early blight (30), leaf scorch (26);
- MobileNetV3: tomato late blight (33), tomato early blight (26).

**Classes with 0% recall in all three models:** cherry healthy (10 images),
potato late blight (8), tomato bacterial spot (9), tomato mosaic virus (10)
and tomato healthy (8).

- The cherry and strawberry healthy photos are clean, single leaves on white,
  and still fail. Cherry goes to apple or blueberry healthy; strawberry to
  raspberry or grape healthy.
- Potato late blight is mostly predicted as tomato late blight, the same
  disease on a related plant.

**Most common confusions (EfficientNet-B0):**

| True | Predicted | Count |
|---|---|---:|
| Corn Northern leaf blight | Corn gray leaf spot | 8 |
| Corn common rust | Corn Northern leaf blight | 5 |
| Tomato early blight | Strawberry leaf scorch | 5 |
| Apple scab | Strawberry leaf scorch | 4 |
| Potato late blight | Tomato late blight | 4 |
| Soybean healthy | Blueberry healthy | 4 |
| Strawberry healthy | Raspberry healthy | 4 |
| Tomato bacterial spot | Tomato Septoria leaf spot | 4 |

Corn shows the opposite of the crop problem: corn photos always get a corn
label (crop right 100% for blight and rust), but the disease is often wrong.
That matches the corn confusion already seen on PlantVillage test.

**Confidence.** On PlantDoc the mean top probability is 0.64 when right and
0.51 when wrong; on PlantVillage test it is 0.89. Label smoothing caps
confidence near 0.9, so these values are not calibrated probabilities. Of the
23 PlantDoc photos with confidence above 0.9, only 9 (39%) are right. A
confident answer on a field photo should not be trusted.

## Is it the background?

The usual explanation is that field photos have cluttered backgrounds. To
test it, I labelled the background of every PlantDoc test image by eye. This
was done before looking at any predictions, from contact sheets without
labels or predictions
([configs/plantdoc_test_background.csv](../configs/plantdoc_test_background.csv)).
The categories:

- **plain** (87): isolated leaves on a uniform background, mostly
  white-background stock photos;
- **natural** (136): a real scene (on the plant, in a hand, on soil);
- **slide** (13): composites with text or several panels.

| Background | n | EfficientNet-B0 acc [95% CI] | MobileNetV3 acc | ResNet-18 acc |
|---|---:|---|---:|---:|
| plain | 87 | 0.345 [0.253, 0.437] | 0.356 | 0.299 |
| natural | 136 | 0.309 [0.228, 0.382] | 0.360 | 0.294 |
| slide | 13 | 0.077 [0.000, 0.231] | 0.154 | 0.077 |

Plain-background photos are not clearly easier than real scenes, for any of
the three models. The two groups hold different classes: 59 of the 87 plain
photos are healthy leaves, against 30 of the 136 natural ones. Comparing
inside each group shows no consistent effect either (EfficientNet-B0):
- diseased leaves: 0.25 plain (n = 28) vs 0.31 natural (n = 106);
- healthy leaves: 0.39 plain (n = 59) vs 0.30 natural (n = 30).

**Reading:** background clutter does not explain most of the gap. A clean leaf
on white is nearly as hard for these models as a leaf in the field. The more
likely cause is that every PlantVillage photo comes from one setting: the same
grey background, lighting, cameras and leaf collection. The models seem to
have learned that look rather than the disease. This is an interpretation;
step 3 does not prove it. For step 4, it suggests that background removal
alone will not close the gap. Changing the photo's colour and look in
training, and some field images, are more promising.

## About PlantDoc itself

PlantDoc was collected from the web, and the background labels show it. Only
136 of 236 test images (58%) are real-scene photos. 87 are stock-style photos
on plain backgrounds, many with watermarks. 13 are slides or composites,
including one cropped strip with almost no leaf. "Field accuracy" here
therefore means "accuracy on web images of plant leaves". Earlier checks also
found label conflicts between PlantDoc test and train duplicates
([data.md](data.md)). These limits apply to every number on this page.
