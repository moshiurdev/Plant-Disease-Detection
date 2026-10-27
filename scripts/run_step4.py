"""Step 4 experiment grid: few-shot PlantDoc fine-tuning.

1. Learning rate: for each start point (ImageNet, PlantVillage), try
   1e-4, 3e-4 and 1e-3 at k=10, seed 0, and keep the one with the best
   PlantDoc val macro-F1. The strong-augmentation PlantVillage model uses
   the PlantVillage choice.
2. Grid: start point x k in {5, 10, 25, all} x seeds {0, 1, 2}, each
   selected on PlantDoc val (scripts/finetune.py).
3. Every finished run is scored once on PlantDoc test (scripts/evaluate.py).

Finished runs (best_val.json / eval_plantdoc.json present) are skipped, so
the script can be restarted.

Usage: python scripts/run_step4.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FT = ROOT / "runs/ft"
INITS = {"imagenet": "imagenet", "pv": "runs/pv_efficientnet_b0", "pv_strong": "runs/pv_efficientnet_b0_strong"}
LRS = (1e-4, 3e-4, 1e-3)
SHOTS = (5, 10, 25, 0)
SEEDS = (0, 1, 2)


def run(cmd, log):
    print(" ".join(cmd), flush=True)
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def finetune(name, init, shots, seed, lr):
    out = FT / name
    if not (out / "best_val.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        run([PY, "scripts/finetune.py", "--init", INITS[init], "--shots", str(shots), "--seed", str(seed),
             "--lr", str(lr), "--out", str(out)], out / "train.log")
    return json.load(open(out / "best_val.json"))["val_macro_f1"]


def main():
    lr_choice = {}
    for init in ("imagenet", "pv"):
        scores = {lr: finetune(f"lr/{init}_lr{lr:g}", init, 10, 0, lr) for lr in LRS}
        lr_choice[init] = max(scores, key=scores.get)
        print(init, scores, "->", lr_choice[init], flush=True)
    lr_choice["pv_strong"] = lr_choice["pv"]
    json.dump(lr_choice, open(FT / "lr_choice.json", "w"), indent=2)

    for init in INITS:
        for shots in SHOTS:
            for seed in SEEDS:
                name = f"{init}_k{shots or 'all'}_s{seed}"
                finetune(name, init, shots, seed, lr_choice[init])
                if not (FT / name / "eval_plantdoc.json").exists():
                    run([PY, "scripts/evaluate.py", str(FT / name), "--set", "plantdoc"], FT / name / "eval.log")


if __name__ == "__main__":
    main()
