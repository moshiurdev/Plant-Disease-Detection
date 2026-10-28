"""Check how EXIF rotation tags affect the leaf models in the demo (step 7 follow-up).

Training and evaluation of the lab and field models opened images without
applying the EXIF rotation tag (pdd/data.py), so tagged photos were seen as
stored. The demo turns photos upright first (ImageOps.exif_transpose), which
is right for phone photos but differs from evaluation. This script counts the
tagged photos and scores the PlantDoc test set both ways with the ONNX models.
The rice model applies the tag in both training (scripts/cache_rice.py) and
the demo, so it is not affected.

Writes results/step7_exif.json.

Usage: python scripts/check_exif.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdd import preprocess  # noqa: E402

PLANTDOC = ROOT / "data/raw/plantdoc"
PLANTVILLAGE = ROOT / "data/raw/plantvillage"


def orientation(path):
    """EXIF orientation tag (1 = no rotation, also when the tag is missing)."""
    with Image.open(path) as im:
        return im.getexif().get(274) or 1


def predict(sess, m, im):
    z = sess.run(None, {"image": preprocess.leaf_square(im)})[0][0]
    classes = np.array(m["classes"])
    if m["allowed_classes"]:
        z = np.where(np.isin(classes, m["allowed_classes"]), z, -np.inf)
    return classes[z.argmax()]


def main():
    meta = json.load(open(ROOT / "models/models.json"))["models"]
    test = pd.read_csv(ROOT / "runs/ft/pv_kall_s1/pred_plantdoc.csv", keep_default_na=False)
    test["orientation"] = [orientation(PLANTDOC / p) for p in test["path"]]
    rotated = (test["orientation"] != 1).values
    train = [p for p in (PLANTDOC / "train").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    pv = pd.read_csv(ROOT / "runs/pv_efficientnet_b0/pred_pv_test.csv", keep_default_na=False)

    res = {
        "plantdoc_train_files": len(train),
        "plantdoc_train_orientation_counts": pd.Series([orientation(p) for p in train]).value_counts().to_dict(),
        "plantdoc_test_files": len(test),
        "plantdoc_test_orientation_counts": test["orientation"].value_counts().to_dict(),
        "plantvillage_test_files": len(pv),
        "plantvillage_test_tagged": int(sum(orientation(PLANTVILLAGE / p) != 1 for p in pv["path"])),
        "plantdoc_test": {},
    }
    y = test["label"].values
    for name in ("field", "lab"):
        m = meta[name]
        sess = ort.InferenceSession(str(ROOT / "models" / m["file"]), providers=["CPUExecutionProvider"])
        stored, upright = [], []
        for p in test["path"]:
            with Image.open(PLANTDOC / p) as im:
                stored.append(predict(sess, m, im))
                upright.append(predict(sess, m, ImageOps.exif_transpose(im)))
        stored, upright = np.array(stored), np.array(upright)
        res["plantdoc_test"][name] = {
            "accuracy_as_stored": float((stored == y).mean()),
            "accuracy_upright": float((upright == y).mean()),
            "predictions_changed": int((stored != upright).sum()),
            "rotated_correct_as_stored": int(((stored == y) & rotated).sum()),
            "rotated_correct_upright": int(((upright == y) & rotated).sum()),
        }
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results/step7_exif.json", "w"), indent=1, default=int)
    print(json.dumps(res, indent=1, default=int))


if __name__ == "__main__":
    main()
