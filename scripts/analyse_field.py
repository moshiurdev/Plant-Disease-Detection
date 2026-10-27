"""Analyse lab -> field errors from saved logits (run scripts/evaluate.py first).

For each run, on PlantDoc test:
  38-way      the model's own prediction over all PlantVillage classes
  restricted  argmax over only the 28 classes PlantDoc has (fixed in advance
              from configs/plantdoc_to_plantvillage.csv, not tuned on test)
  crop        is the predicted crop right, whatever the disease?
  health      is healthy vs diseased right?
plus accuracy by background type (configs/plantdoc_test_background.csv,
labelled by eye without looking at predictions: plain = isolated leaves on a
uniform background, slide = composite with text or several panels,
natural = real scene), which classes the model predicts, and how confident it
is on field photos compared with PlantVillage test.

Writes results/step3_plantdoc.json.

Usage: python scripts/analyse_field.py runs/pv_efficientnet_b0 runs/pv_mobilenet_v3_large ...
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdd.metrics import summarize  # noqa: E402


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def crop(c):
    return c.split("___")[0]


def scored(y, p, names, mask):
    """summarize() on string labels mapped to integer ids."""
    idx = {c: i for i, c in enumerate(names)}
    r = summarize([idx[a] for a in y[mask]], [idx[b] for b in p[mask]], len(names))
    return {k: r[k] for k in ("n", "accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95", "errors")}


def analyse(run):
    ev = json.load(open(run / "eval_plantdoc.json"))
    classes = ev["classes"]
    pred = pd.read_csv(run / "pred_plantdoc.csv", keep_default_na=False)
    z = np.load(run / "logits_plantdoc.npy")
    dups = set(pd.read_csv(ROOT / "configs/plantdoc_test_train_duplicates.csv")["test_path"])
    clean = ~pred["path"].isin(dups).values
    pd_classes = sorted(pd.read_csv(ROOT / "configs/plantdoc_to_plantvillage.csv")["plantvillage_class"])
    keep = np.array([c in pd_classes for c in classes])

    y = pred["label"].values
    p38 = np.array(classes)[z.argmax(1)]
    pr = np.array(classes)[np.where(keep, z, -np.inf).argmax(1)]
    crops = sorted({crop(c) for c in classes})
    health = ["diseased", "healthy"]
    status = np.vectorize(lambda c: "healthy" if c.endswith("healthy") else "diseased")

    out = {"run": str(run.relative_to(ROOT)), "arch": ev["arch"]}
    for name, m in (("all", np.ones(len(y), bool)), ("without_train_duplicates", clean)):
        out[name] = {
            "38_way": scored(y, p38, classes, m),
            "restricted_to_plantdoc_classes": scored(y, pr, classes, m),
            "crop": scored(np.vectorize(crop)(y), np.vectorize(crop)(p38), crops, m),
            "health": scored(status(y), status(p38), health, m),
        }

    conf = softmax(z).max(1)
    right = p38 == y
    out["confidence"] = {
        "mean_when_right": float(conf[right].mean()),
        "mean_when_wrong": float(conf[~right].mean()),
        "share_above_0.9": float((conf > 0.9).mean()),
        "accuracy_when_above_0.9": float(right[conf > 0.9].mean()) if (conf > 0.9).any() else None,
    }
    pv = run / "logits_pv_test.npy"
    if pv.exists():
        out["confidence"]["pv_test_mean"] = float(softmax(np.load(pv)).max(1).mean())

    bg = pd.read_csv(ROOT / "configs/plantdoc_test_background.csv").set_index("path")["background"]
    bgv = bg.loc[pred["path"]].values
    out["by_background"] = {b: {"38_way": scored(y, p38, classes, bgv == b),
                                "restricted_to_plantdoc_classes": scored(y, pr, classes, bgv == b),
                                "crop": scored(np.vectorize(crop)(y), np.vectorize(crop)(p38), crops, bgv == b)}
                            for b in ("plain", "natural", "slide")}

    out["predictions_outside_plantdoc_classes"] = int((~np.isin(p38, pd_classes)).sum())
    out["most_predicted"] = Counter(p38).most_common(8)
    rec = {}
    for c in sorted(set(y)):
        m = y == c
        rec[c] = {"n": int(m.sum()), "recall_38_way": float(right[m].mean()),
                  "crop_right": float((np.vectorize(crop)(p38[m]) == crop(c)).mean()),
                  "top_predictions": Counter(p38[m]).most_common(3)}
    out["per_class"] = rec
    out["top_confusions"] = Counter(zip(y[~right], p38[~right])).most_common(12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/step3_plantdoc.json")
    args = ap.parse_args()
    res = [analyse(Path(r).resolve()) for r in args.runs]
    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    for r in res:
        a = r["all"]
        print(r["arch"], " ".join(f"{k}={v['accuracy']:.3f}" for k, v in a.items()),
              "| outside:", r["predictions_outside_plantdoc_classes"], "| conf:",
              {k: round(v, 3) if v is not None else None for k, v in r["confidence"].items()})
        for b, v in r["by_background"].items():
            print("   ", b, v["38_way"]["n"], " ".join(f"{k}={m['accuracy']:.3f} {np.round(m['accuracy_ci95'], 3).tolist()}"
                                                   for k, m in v.items()))


if __name__ == "__main__":
    main()
