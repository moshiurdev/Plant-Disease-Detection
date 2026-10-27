"""Leakage-aware PlantDoc splits for fine-tuning (step 4).

PlantDoc was collected from the web, so the same photo often appears more
than once, sometimes mirrored or under another class. Pairs come from
dedup.py + verify_pairs.py on all PlantDoc images and are linked with the same
rule as PlantVillage (make_splits.py). Then:
  - test images keep the official test split;
  - train images in a group that also holds a test image -> excluded (dup_of_test);
  - groups whose images carry different labels -> excluded (label_conflict);
  - the remaining train groups are split 80/20 into train / val, stratified by
    label (StratifiedGroupKFold, 5 folds, fold 0 = val).

Writes data/splits/plantdoc_grouped.csv (committed).

Usage:
  python scripts/dedup.py data/meta/plantdoc_for_dedup.csv data/raw/plantdoc \
      --out data/meta/plantdoc_pairs.csv --min-corr 0.90
  python scripts/verify_pairs.py data/meta/plantdoc_pairs.csv data/raw/plantdoc
  python scripts/make_plantdoc_splits.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_splits import ANY_LABEL_INLIERS, MIN_CORR, SAME_LABEL_INLIERS, SEED, UnionFind  # noqa: E402


def main():
    df = pd.read_csv(ROOT / "data/meta/plantdoc.csv", keep_default_na=False)
    df = df.rename(columns={"plantvillage_class": "label", "split": "official_split"})
    pairs = pd.read_csv(ROOT / "data/meta/plantdoc_pairs.csv", keep_default_na=False)
    idx = {p: i for i, p in enumerate(df["path"])}
    uf = UnionFind(len(df))
    same = pairs["label_a"] == pairs["label_b"]
    link = ((same & (pairs["inliers"] >= SAME_LABEL_INLIERS))
            | (pairs["inliers"] >= ANY_LABEL_INLIERS)
            | (pairs["corr"] >= MIN_CORR))
    for a, b in zip(pairs.loc[link, "a"], pairs.loc[link, "b"]):
        uf.union(idx[a], idx[b])
    df["group"] = [uf.find(i) for i in range(len(df))]

    has_test = df.groupby("group")["official_split"].agg(lambda s: (s == "test").any())
    mixed = df.groupby("group")["label"].nunique() > 1
    train = df["official_split"] == "train"
    df["split"] = df["official_split"]
    df.loc[train & df["group"].map(mixed), "split"] = "excluded_label_conflict"
    df.loc[train & df["group"].map(has_test), "split"] = "excluded_dup_of_test"

    pool = df[df["split"] == "train"]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, val = next(sgkf.split(pool, pool["label"], pool["group"]))
    df.loc[pool.index[val], "split"] = "val"

    out = df[["path", "label", "plantdoc_class", "group", "split", "official_split"]]
    (ROOT / "data/splits").mkdir(exist_ok=True)
    out.to_csv(ROOT / "data/splits/plantdoc_grouped.csv", index=False)
    stats = {
        "linked_pairs": int(link.sum()),
        "linked_pairs_with_different_labels": int((link & ~same).sum()),
        "split_images": out["split"].value_counts().to_dict(),
        "classes": {s: int(out[out.split == s]["label"].nunique()) for s in ("train", "val", "test")},
        "train_per_class": out[out.split == "train"]["label"].value_counts().describe().round(1).to_dict(),
        "val_per_class": out[out.split == "val"]["label"].value_counts().describe().round(1).to_dict(),
    }
    json.dump(stats, open(ROOT / "data/splits/plantdoc_grouped_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
