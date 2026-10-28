"""Step 6 experiment grid: background shift inside rice (white <-> field).

ImageNet start, the step-5 learning rate, 5 repeats (same splits as step 5):
  white_all  train on all white-background pool images, select on white val
  white_k40  40 white images per class (200), about the size of the field pool
  field_all  train on all field pool images (~216), select on field val
Every run is scored on both white and field test images. The step-5 runs
trained on both backgrounds (runs/rice/imagenet_kall_r*) are the reference.

Usage: python scripts/run_step6.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_step5 import OUT, REPEATS, finetune  # noqa: E402

CONDITIONS = {"white_all": ("white", 0), "white_k40": ("white", 40), "field_all": ("field", 0)}


def main():
    lr = json.load(open(OUT / "lr_choice.json"))["imagenet"]
    for name, (bg, shots) in CONDITIONS.items():
        for r in REPEATS:
            finetune(f"bg/{name}_r{r}", "imagenet", shots, r, lr, ("--train-bg", bg))


if __name__ == "__main__":
    main()
