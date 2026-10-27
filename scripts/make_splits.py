"""Build leakage-aware PlantVillage train/val/test splits.

Images are grouped so that one physical leaf never lands on two sides:
  1. the official leaf map (leaf_id), including the loader's fallback IDs,
     which already join 'X.JPG' with 'X copy.jpg';
  2. verified near-duplicate pairs from dedup.py + verify_pairs.py. A pair is
     linked if it shares a label and has >= 6 ORB/RANSAC inliers, or has >= 20
     inliers whatever the labels, or a thumbnail correlation >= 0.985.
     Thresholds were set by looking at sampled pairs (docs/splits.md).
Groups are then split 70/15/15 with stratification by label
(StratifiedGroupKFold, 20 folds: 3 test, 3 val, 14 train).

Writes data/splits/plantvillage_grouped.csv (committed) with path, label,
group, split and the official split for comparison.

Usage: python scripts/make_splits.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
SAME_LABEL_INLIERS = 6
ANY_LABEL_INLIERS = 20
MIN_CORR = 0.985


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    df = pd.read_csv(ROOT / "data/meta/plantvillage.csv", keep_default_na=False)
    df = df.rename(columns={"split": "official_split"})
    pairs = pd.read_csv(ROOT / "data/meta/plantvillage_pairs.csv", keep_default_na=False)
    idx = {p: i for i, p in enumerate(df["path"])}
    uf = UnionFind(len(df))

    first = {}
    for i, leaf in enumerate(df["leaf_id"]):
        if leaf in first:
            uf.union(first[leaf], i)
        else:
            first[leaf] = i

    same = pairs["label_a"] == pairs["label_b"]
    link = ((same & (pairs["inliers"] >= SAME_LABEL_INLIERS))
            | (pairs["inliers"] >= ANY_LABEL_INLIERS)
            | (pairs["corr"] >= MIN_CORR))
    for a, b in zip(pairs.loc[link, "a"], pairs.loc[link, "b"]):
        uf.union(idx[a], idx[b])

    df["group"] = [uf.find(i) for i in range(len(df))]
    sizes = df.groupby("group").size()

    # Stratify each group by its most common label.
    g_label = df.groupby("group")["label"].agg(lambda s: s.value_counts().index[0])
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(df, df["group"].map(g_label), df["group"])):
        fold[te] = k
    df["split"] = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))

    out = df[["path", "label", "leaf_id", "group", "split", "official_split"]]
    (ROOT / "data/splits").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "data/splits/plantvillage_grouped.csv", index=False)

    linked = pairs[link]
    conflicts = linked[linked["label_a"] != linked["label_b"]]
    off_cross = out.groupby("group")["official_split"].nunique()
    stats = {
        "images": len(df),
        "groups": int(len(sizes)),
        "group_size": {"max": int(sizes.max()), "mean": round(float(sizes.mean()), 2),
                       "groups_over_20": int((sizes > 20).sum())},
        "linked_pairs": int(link.sum()),
        "linked_pairs_with_different_labels": int(len(conflicts)),
        "split_images": out["split"].value_counts().to_dict(),
        "split_classes": {s: int(out[out.split == s]["label"].nunique()) for s in ("train", "val", "test")},
        "groups_mixing_labels": int((df.groupby("group")["label"].nunique() > 1).sum()),
        "official_split_groups_crossing": int((off_cross > 1).sum()),
        "official_split_test_images_in_crossing_groups": int(
            out[out["group"].isin(off_cross[off_cross > 1].index) & (out["official_split"] == "test")].shape[0]),
    }
    conflicts.to_csv(ROOT / "data/splits/plantvillage_label_conflicts.csv", index=False)
    json.dump(stats, open(ROOT / "data/splits/plantvillage_grouped_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
