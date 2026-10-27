#!/usr/bin/env bash
# Step 2: PlantVillage baselines on the grouped split (160 px, 6 epochs, seed 0).
# Model selection uses the validation set only.
set -euo pipefail
cd "$(dirname "$0")/.."
for arch in resnet18 mobilenet_v3_large efficientnet_b0; do
  .venv/bin/python -u scripts/train.py --arch "$arch" --epochs 6 --out "runs/pv_${arch}" 2>&1 \
    | grep -v Warning | tee "runs/pv_${arch}.log"
done
