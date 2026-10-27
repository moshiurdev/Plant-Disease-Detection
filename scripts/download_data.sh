#!/usr/bin/env bash
# Download all three datasets into data/ and verify them.
# PlantVillage: Hugging Face mohanty/PlantVillage (CC BY-SA 3.0), color images only are extracted.
# PlantDoc: GitHub pratikkayal/PlantDoc-Dataset (CC BY 4.0), exported from git to avoid case collisions.
# Dhan-Shomadhan: Mendeley Data 10.17632/znsxdctwtt.1 (CC BY 4.0), each file checked against its SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
HF=https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main
PV_SHA256=fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84
PLANTDOC_COMMIT=5467f6012d78d1c446145d5f582da6096f852ae8

mkdir -p data/downloads/plantvillage data/raw
for f in README.md leaf_grouping/leaf-map.json splits/color_train.txt splits/color_test.txt plant_village.py; do
  curl -sfL --create-dirs -o "data/downloads/plantvillage/$f" "$HF/$f"
done
curl -fL -C - -o data/downloads/plantvillage/data.zip "$HF/data.zip"
echo "$PV_SHA256  data/downloads/plantvillage/data.zip" | shasum -a 256 -c -
$PY -I -c "
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
m = [i for i in z.infolist() if i.filename.startswith('raw/color/') and not i.filename.endswith('/')]
assert not any(i.filename.startswith('/') or '..' in i.filename.split('/') for i in m)
for i in m:
    z.extract(i, sys.argv[2])
print('PlantVillage color images:', len(m))
" data/downloads/plantvillage/data.zip data/raw/plantvillage

if [ ! -d data/downloads/plantdoc ]; then
  git clone -q https://github.com/pratikkayal/PlantDoc-Dataset.git data/downloads/plantdoc
fi
git -C data/downloads/plantdoc checkout -q "$PLANTDOC_COMMIT"
$PY -I scripts/export_plantdoc.py

$PY -I scripts/download_dhan_shomadhan.py
