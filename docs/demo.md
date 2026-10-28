# ONNX models and the CPU demo (step 7)

Three models are exported to ONNX and served by a small FastAPI app that runs
on the CPU with ONNX Runtime (no PyTorch needed to run it). Each model was
chosen on validation in its own step. None was picked by its test score.

| Model | Source run | What it is | Test accuracy |
|---|---|---|---|
| `lab` | `runs/pv_efficientnet_b0` | Step 2. PlantVillage only, 38 classes | 0.997 PlantVillage, 0.309 PlantDoc |
| `field` | `runs/ft/pv_kall_s1` | Step 4. PlantVillage start, fine-tuned on all PlantDoc train images; best seed on PlantDoc val | 0.636 PlantDoc (0.546 PlantVillage) |
| `rice` | `runs/rice/imagenet_kall_r4` | Step 5. ImageNet start (higher mean val than the PlantVillage start), all labels, repeat with the best val | 0.797 on its test split (white 0.806, field 0.776) |

The field model keeps 38 outputs but is only allowed to answer one of the 28
PlantDoc classes. Fine-tuning already made it ignore the other 10 (0 of 236
PlantDoc test predictions fell outside), so this changes no result. It just
makes the rule explicit.

The rice model's score is from one split (repeat 4, 222 test images). The
5-repeat average for this setup is 0.780 ± 0.021 (step 5). Choosing the best
repeat by val is a small optimistic step, and the average is the fairer
number. Its field score (0.776) is above the 5-repeat field average (0.674 ±
0.064), so expect weaker results on field photos.

## Export and checks

`scripts/export_onnx.py` exports each model (opset 17, dynamic batch) to
`models/<name>.onnx`. It writes `models/models.json` with classes,
preprocessing, scores, checks, timings and the SHA-256 of each file. The
`.onnx` files (16 MB each) are not committed; they are attached to the GitHub
release.

The demo's preprocessing (`pdd/preprocess.py`) uses only PIL and numpy. The
export script checks it against the torchvision transforms used in
evaluation, and checks ONNX Runtime against PyTorch:

| Check | lab | field | rice |
|---|---:|---:|---:|
| Max difference in the input tensor, 20 images | 0 | 0 | 0 |
| Max difference in logits, ONNX vs PyTorch | 2e-6 | 8e-6 | 1e-5 |
| Test images re-scored through the demo path | 1,000* | 236 | 222 |
| Same prediction as PyTorch | 100% | 100% | 100% |

\* The first 1,000 PlantVillage test images by path (accuracy 0.989 on that
subset; 0.997 on the full test set).

For rice, the demo also repeats the image caching from training
(`scripts/cache_rice.py`): turn upright, rotate to portrait, scale to 224 px
wide. Training images were decoded with JPEG draft mode, so a full-size phone
photo can differ very slightly from its cached copy.

## CPU timing

ONNX Runtime 1.31 on an Apple M1 (8 cores), one image, median of 100 runs
after 10 warm-up runs (90th percentile in brackets):

| Model | Input | File | 1 thread | All threads |
|---|---|---:|---:|---:|
| lab | 160 × 160 | 16.2 MB | 14.6 ms (14.8) | 5.1 ms (5.7) |
| field | 160 × 160 | 16.2 MB | 14.6 ms (14.8) | 5.1 ms (5.6) |
| rice | 256 × 128 | 16.0 MB | 18.0 ms (18.7) | 5.9 ms (6.4) |

Decoding and resizing an 8-megapixel phone photo takes longer than the model:
about 105 ms for a Dhan-Shomadhan original, measured once through the API.

## The demo

```bash
.venv/bin/pip install -r requirements-demo.txt   # or the full requirements.txt
.venv/bin/python scripts/export_onnx.py          # or download the .onnx files from the release into models/
.venv/bin/uvicorn app.main:app --port 8000       # open http://localhost:8000
```

- `GET /`: upload page. Pick a model, choose a leaf photo, see the top 5
  classes with probabilities and the time taken.
- `GET /models`: models, classes, scores and notes.
- `POST /predict`: form fields `file`, `model` (`lab`, `field` or `rice`) and
  optional `top`. Returns JSON.

```bash
curl -F file=@leaf.jpg -F model=field http://localhost:8000/predict
```

`tests/test_app.py` (pytest) covers the endpoints, bad uploads, that the field
model only answers PlantDoc classes, a real PlantDoc photo and a full-size
rice photo.

**Limits.** This is a research demo, not a diagnosis tool. The probabilities
come from models trained with label smoothing and are not calibrated. Every
model answers with one of its classes even for a photo of something else.
Steps 3 and 6 showed that photos unlike the training data can get confident,
wrong answers.
