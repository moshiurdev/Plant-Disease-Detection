"""Collect step-5 rice results into results/step5_rice.json and print tables.

For each start point (ImageNet, PlantVillage) and k: mean and std over the 5
repeats of val macro-F1 (used for every choice) and test accuracy / macro-F1,
on all test images and per background. Also the paired PlantVillage - ImageNet
difference per repeat (same split, same training images).

Usage: python scripts/summarise_step5.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/rice"
KS = ("10", "25", "50", "all")


def row(run):
    t = json.load(open(run / "test.json"))
    return {"val_macro_f1": json.load(open(run / "best_val.json"))["val_macro_f1"],
            "best_step": json.load(open(run / "best_val.json"))["step"],
            "train_images": json.load(open(run / "config.json"))["train_images"],
            "test_acc": t["all"]["accuracy"], "test_macro_f1": t["all"]["macro_f1"],
            "test_acc_white": t["white"]["accuracy"], "test_acc_field": t["field"]["accuracy"]}


def agg(rows, keys):
    out = {"repeats": len(rows), "train_images": [r["train_images"] for r in rows],
           "best_steps": [r["best_step"] for r in rows]}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "runs": v.round(4).tolist()}
    return out


def main():
    keys = ("val_macro_f1", "test_acc", "test_macro_f1", "test_acc_white", "test_acc_field")
    res = {"lr_choice": json.load(open(OUT / "lr_choice.json")), "lr_search": {}, "runs": {}, "pv_minus_imagenet": {}}
    for d in sorted((OUT / "lr").iterdir()):
        res["lr_search"][d.name] = json.load(open(d / "best_val.json"))["val_macro_f1"]
    rows = {}
    for init in ("imagenet", "pv"):
        for k in KS:
            rows[init, k] = [row(r) for r in sorted(OUT.glob(f"{init}_k{k}_r*"))]
            res["runs"][f"{init}_k{k}"] = agg(rows[init, k], keys)
    for k in KS:
        res["pv_minus_imagenet"][f"k{k}"] = {
            m: [round(p[m] - i[m], 4) for p, i in zip(rows["pv", k], rows["imagenet", k])] for m in keys}

    json.dump(res, open(ROOT / "results/step5_rice.json", "w"), indent=1)
    print("lr search (val macro-F1):", {k: round(v, 3) for k, v in res["lr_search"].items()}, res["lr_choice"])
    for name, v in res["runs"].items():
        f = lambda m: f"{v[m]['mean']:.3f}±{v[m]['std']:.3f}"
        print(f"{name:14s} n={v['train_images']}  val F1 {f('val_macro_f1')}  test acc {f('test_acc')}"
              f"  F1 {f('test_macro_f1')}  white {f('test_acc_white')}  field {f('test_acc_field')}"
              f"  steps {v['best_steps']}")
    for k, d in res["pv_minus_imagenet"].items():
        print("PV - ImageNet", k, {m: d[m] for m in ("val_macro_f1", "test_acc", "test_acc_field")})


if __name__ == "__main__":
    main()
