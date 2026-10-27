"""Collect step-4 results into results/step4_gap.json and print tables.

Rows:
  - no fine-tuning: the step-2 model and the strong-augmentation model
    (runs/pv_efficientnet_b0[_strong]);
  - fine-tuned: each start point x k, mean and std over seeds of PlantDoc val
    macro-F1 (used for every choice) and PlantDoc test accuracy / macro-F1
    (all 236 images, and the 220 that are not copies of train images).

Usage: python scripts/summarise_step4.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "runs/ft"


def test_scores(run):
    ev = json.load(open(run / "eval_plantdoc.json"))
    return {"test_acc": ev["all"]["accuracy"], "test_macro_f1": ev["all"]["macro_f1"],
            "test_acc_ci95": ev["all"]["accuracy_ci95"],
            "test_acc_clean": ev["without_train_duplicates"]["accuracy"]}


def main():
    res = {"lr_choice": json.load(open(FT / "lr_choice.json")), "lr_search": {}, "no_finetune": {}, "finetune": {}}
    for d in sorted((FT / "lr").iterdir()):
        res["lr_search"][d.name] = json.load(open(d / "best_val.json"))["val_macro_f1"]

    for name, run in (("pv", "pv_efficientnet_b0"), ("pv_strong", "pv_efficientnet_b0_strong")):
        run = ROOT / "runs" / run
        val = json.load(open(run / "eval_plantdoc_val.json"))["all"]
        res["no_finetune"][name] = {"val_macro_f1": val["macro_f1"]} | test_scores(run)

    for init in ("imagenet", "pv", "pv_strong"):
        for k in ("5", "10", "25", "all"):
            runs = sorted(FT.glob(f"{init}_k{k}_s*"))
            rows = [{"val_macro_f1": json.load(open(r / "best_val.json"))["val_macro_f1"],
                     "best_step": json.load(open(r / "best_val.json"))["step"],
                     "train_images": json.load(open(r / "config.json"))["train_images"]} | test_scores(r)
                    for r in runs]
            agg = {"seeds": len(rows), "train_images": rows[0]["train_images"],
                   "best_steps": [r["best_step"] for r in rows]}
            for key in ("val_macro_f1", "test_acc", "test_macro_f1", "test_acc_clean"):
                v = np.array([r[key] for r in rows])
                agg[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "runs": v.round(4).tolist()}
            res["finetune"][f"{init}_k{k}"] = agg

    out = ROOT / "results/step4_gap.json"
    json.dump(res, open(out, "w"), indent=1)

    print("lr search (val macro-F1):", {k: round(v, 3) for k, v in res["lr_search"].items()}, res["lr_choice"])
    for k, v in res["no_finetune"].items():
        print(f"{k:18s} no fine-tune   val F1 {v['val_macro_f1']:.3f}  test acc {v['test_acc']:.3f}"
              f"  F1 {v['test_macro_f1']:.3f}  clean acc {v['test_acc_clean']:.3f}")
    for k, v in res["finetune"].items():
        f = lambda m: f"{v[m]['mean']:.3f}±{v[m]['std']:.3f}"
        print(f"{k:18s} n={v['train_images']:4d}  val F1 {f('val_macro_f1')}  test acc {f('test_acc')}"
              f"  F1 {f('test_macro_f1')}  clean acc {f('test_acc_clean')}  steps {v['best_steps']}")


if __name__ == "__main__":
    main()
