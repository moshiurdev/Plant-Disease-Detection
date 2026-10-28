"""Collect step-6 results (rice background shift) into results/step6_background.json.

Rows: training background (white all, white 40/class, field all, and both
from step 5), columns: accuracy and macro-F1 on white and on field test
images, mean and std over the 5 repeats.

Usage: python scripts/summarise_step6.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/rice"
CONDITIONS = {"white_all": "bg/white_all_r*", "white_k40": "bg/white_k40_r*",
              "field_all": "bg/field_all_r*", "both_all (step 5)": "imagenet_kall_r*"}


def main():
    res = {}
    for name, pattern in CONDITIONS.items():
        rows = []
        for run in sorted(OUT.glob(pattern)):
            t = json.load(open(run / "test.json"))
            rows.append({"train_images": json.load(open(run / "config.json"))["train_images"],
                         "val_macro_f1": json.load(open(run / "best_val.json"))["val_macro_f1"],
                         **{f"{m}_{b}": t[b][k] for b in ("white", "field")
                            for m, k in (("acc", "accuracy"), ("macro_f1", "macro_f1"))}})
        out = {"repeats": len(rows), "train_images": [r["train_images"] for r in rows]}
        for key in rows[0]:
            if key != "train_images":
                v = np.array([r[key] for r in rows])
                out[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "runs": v.round(4).tolist()}
        res[name] = out
    json.dump(res, open(ROOT / "results/step6_background.json", "w"), indent=1)
    for name, v in res.items():
        f = lambda m: f"{v[m]['mean']:.3f}±{v[m]['std']:.3f}"
        print(f"{name:18s} n={v['train_images']}  val F1 {f('val_macro_f1')}  white acc {f('acc_white')}"
              f" F1 {f('macro_f1_white')}  field acc {f('acc_field')} F1 {f('macro_f1_field')}")


if __name__ == "__main__":
    main()
