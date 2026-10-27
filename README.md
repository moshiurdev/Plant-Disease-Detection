# Plant Disease Detection: from lab photos to real fields and to Bangladeshi rice

**Status: step 1 of the plan (data checked, nothing trained yet). No results are reported here until they are measured.**

Plant disease classifiers often score close to 100% on PlantVillage, but those
photos are single leaves on a plain background in a lab. Farmers take photos in
the field: cluttered backgrounds, several leaves, shadows, blur. This project
measures how much accuracy is lost when a model trained on lab photos meets
field photos, tries simple ways to close that gap, and asks whether what the
model learned helps on a crop it has never seen (rice from Bangladesh) when
only a few labelled photos are available.

## Research questions

1. **Lab → field.** A model trained on PlantVillage is tested on PlantDoc field
   photos of the same diseases. How large is the drop, which classes break, and
   how much do stronger augmentation and a small amount of field fine-tuning
   recover?
2. **Low-data transfer to a new crop.** On Dhan-Shomadhan (Bangladeshi rice,
   5 diseases), does starting from a PlantVillage-trained model beat starting
   from ImageNet when only 10 / 25 / 50 / all images per class are labelled?
   If it does not help, that is reported as the result.
3. **Background shift inside rice.** Train on white-background rice photos,
   test on field-background photos, and the reverse.

## Datasets

All three are public and were downloaded and checked on 2026-10-27. See
[docs/data.md](docs/data.md) for counts, checksums and the leakage checks.

| Dataset | What it is | Size | License | Source |
|---|---|---|---|---|
| PlantVillage (color) | Single leaves on plain backgrounds, 14 crops, 38 classes | 54,305 images | CC BY-SA 3.0 | [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage) (Hughes & Salathé 2015; Mohanty et al. 2016) |
| PlantDoc | Field / web photos, 13 crops, 28 classes (classification version) | 2,578 images (2,342 train / 236 test) | CC BY 4.0 | [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (Singh et al., CODS-COMAD 2020) |
| Dhan-Shomadhan | Bangladeshi rice leaves, 5 diseases, white and field backgrounds | 1,106 images (769 white / 337 field) | CC BY 4.0 | [Mendeley Data, DOI 10.17632/znsxdctwtt.1](https://data.mendeley.com/datasets/znsxdctwtt/1) (Hossain et al., Daffodil International University) |

All 28 PlantDoc classes map one-to-one onto PlantVillage classes
([configs/plantdoc_to_plantvillage.csv](configs/plantdoc_to_plantvillage.csv)),
so a PlantVillage model can be scored on PlantDoc without retraining.

## Evaluation rules

Same standard as my earlier projects: splits that keep related images apart,
model selection on validation data only, and every number in this README
measured by a script in this repository.

- **PlantVillage:** several photos exist of the same physical leaf. Splits use the
  official leaf-grouped train/test lists, and the validation set is carved out
  of train by leaf ID, never by image.
- **PlantDoc:** its official test set is the field test set. PlantDoc is never
  used for model selection in question 1. Images in PlantDoc test that are near
  copies of PlantDoc train images or of PlantVillage images are reported and
  scored separately.
- **Dhan-Shomadhan:** small, so every rice result is averaged over repeated
  splits and seeds and reported as mean ± standard deviation. Near-duplicate
  photos are kept in the same fold.
- **Metrics:** accuracy and macro-F1 (classes are imbalanced), per-class recall
  and confusion matrices.

## Plan

| Step | What | Done when |
|---|---|---|
| 1 | Download and check all data | Counts, checksums and leakage checks written to `docs/data.md` |
| 2 | PlantVillage baselines (MobileNetV3 / EfficientNet / ResNet-18, ImageNet init) | In-domain test numbers on leaf-grouped splits |
| 3 | Lab → field on PlantDoc test, error analysis | Field accuracy, per-class drops, failure examples |
| 4 | Closing the gap: stronger augmentation, background handling, few-shot PlantDoc fine-tuning | Before/after table on PlantDoc test |
| 5 | Rice transfer: ImageNet vs PlantVillage init, 10/25/50/all per class | Learning curves with error bars |
| 6 | Rice background shift (white ↔ field) | Cross-background table |
| 7 | ONNX export, CPU timing, FastAPI demo (upload a leaf photo) | Demo runs on a laptop CPU |
| 8 | README with final results and limitations, GitHub release | Repo public (only after asking) |

Hardware: Apple M1, 8 GB RAM (PyTorch MPS for training, ONNX Runtime on CPU for the demo).

## Reproduce step 1

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh
.venv/bin/python scripts/check_data.py
```

## Licenses and credit

Code in this repository: MIT. The datasets keep their own licenses (above) and
are not redistributed here; the scripts download them from the original sources.
PlantVillage is CC BY-SA 3.0, so any images or derived datasets shared from it
must keep that license.
