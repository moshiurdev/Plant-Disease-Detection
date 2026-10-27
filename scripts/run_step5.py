"""Step 5 experiment grid: rice transfer, ImageNet vs PlantVillage start.

1. Learning rate: for each start point, try 1e-4, 3e-4 and 1e-3 at k=10 on
   repeat 0, and keep the one with the best val macro-F1.
2. Grid: start point x k in {10, 25, 50, all} x repeats 0-4 (each repeat is
   its own grouped split and seed; scripts/finetune_rice.py). Each run picks
   its checkpoint on its val images and then scores its test images once.

Finished runs (test.json present) are skipped, so the script can be restarted.

Usage: python scripts/run_step5.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "runs/rice"
INITS = {"imagenet": "imagenet", "pv": "runs/pv_efficientnet_b0"}
LRS = (1e-4, 3e-4, 1e-3)
SHOTS = (10, 25, 50, 0)
REPEATS = range(5)


def finetune(name, init, shots, repeat, lr, extra=()):
    out = OUT / name
    if not (out / "test.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "scripts/finetune_rice.py", "--init", INITS[init], "--repeat", str(repeat),
               "--shots", str(shots), "--lr", str(lr), "--out", str(out), *extra]
        print(" ".join(cmd), flush=True)
        with open(out / "train.log", "w") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    return json.load(open(out / "best_val.json"))["val_macro_f1"]


def main():
    lr_choice = {}
    for init in INITS:
        scores = {lr: finetune(f"lr/{init}_lr{lr:g}", init, 10, 0, lr) for lr in LRS}
        lr_choice[init] = max(scores, key=scores.get)
        print(init, scores, "->", lr_choice[init], flush=True)
    json.dump(lr_choice, open(OUT / "lr_choice.json", "w"), indent=2)

    for init in INITS:
        for shots in SHOTS:
            for r in REPEATS:
                finetune(f"{init}_k{shots or 'all'}_r{r}", init, shots, r, lr_choice[init])


if __name__ == "__main__":
    main()
