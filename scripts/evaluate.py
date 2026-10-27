"""Evaluate a trained checkpoint on a held-out set.

Sets:
  pv_test   PlantVillage grouped test split (in-domain)
  pv_val    PlantVillage grouped validation split
  plantdoc  PlantDoc official test, mapped to PlantVillage classes; also
            reported without the test images that duplicate PlantDoc train
            images (configs/plantdoc_test_train_duplicates.csv)
  plantdoc_val  validation split carved from PlantDoc train
            (data/splits/plantdoc_grouped.csv)

Writes <run>/eval_<set>.json with accuracy, macro-F1 (95% bootstrap CIs),
per-class recall and the confusion matrix, predictions as CSV and the raw
logits as .npy (for scripts/analyse_field.py).

Usage: python scripts/evaluate.py runs/pv_resnet18 --set pv_test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdd import data, models  # noqa: E402
from pdd.metrics import summarize  # noqa: E402


def load_set(name, classes, size):
    tf = data.eval_transform(size)
    if name in ("pv_test", "pv_val"):
        ds = data.plantvillage_split(name.split("_")[1], tf)
        return ds, None
    if name == "plantdoc":
        ds = data.plantdoc_split("test", tf)
        dups = set(pd.read_csv(ROOT / "configs/plantdoc_test_train_duplicates.csv")["test_path"])
        return ds, ~pd.Series(ds.paths).isin(dups).values
    if name == "plantdoc_val":
        return data.plantdoc_split("val", tf), None
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--set", default="pv_test", choices=["pv_test", "pv_val", "plantdoc", "plantdoc_val"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    run = Path(args.run)
    ck = torch.load(run / "best.pt", map_location="cpu")
    classes, size = ck["classes"], ck["size"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = models.build(ck["arch"], len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    ds, clean_mask = load_set(args.set, classes, size)
    dl = DataLoader(ds, 128, shuffle=False, num_workers=args.workers)
    ys, zs = [], []
    with torch.no_grad():
        for x, y in dl:
            zs.append(model(x.to(device)).float().cpu())
            ys.append(y)
    y, z = torch.cat(ys).numpy(), torch.cat(zs).numpy()
    p = z.argmax(1)
    np.save(run / f"logits_{args.set}.npy", z)

    res = {"run": str(run), "arch": ck["arch"], "epoch": ck["epoch"], "set": args.set,
           "classes": classes, "all": summarize(y, p, len(classes))}
    if clean_mask is not None:
        res["without_train_duplicates"] = summarize(y[clean_mask], p[clean_mask], len(classes))
    json.dump(res, open(run / f"eval_{args.set}.json", "w"), indent=1)
    pd.DataFrame({"path": ds.paths, "label": [classes[i] for i in y], "pred": [classes[i] for i in p]}) \
        .to_csv(run / f"pred_{args.set}.csv", index=False)
    for k in ("all", "without_train_duplicates"):
        if k in res:
            r = res[k]
            print(f"{args.set} [{k}] n={r['n']} acc={r['accuracy']:.4f} {r['accuracy_ci95']} "
                  f"macroF1={r['macro_f1']:.4f} {r['macro_f1_ci95']} errors={r['errors']}")


if __name__ == "__main__":
    main()
