"""Group repeated shots and make 5 repeated grouped splits of Dhan-Shomadhan (steps 5-6).

Pairs come from scripts/rice_pairs.py (ORB inliers between every two images
with the same background). A pair is linked when it reaches the threshold
for its background, or a lower one if both images have the same label (see
docs/rice.md for how the thresholds were set). Linked images are joined into
groups (union-find), and a group is never split across pool / val / test.

For each repeat r = 0..4 (seed r), groups are split with StratifiedGroupKFold,
stratified by label x background: one fold of 5 (about 20%) is test, then
one fold of 5 of the rest (about 16%) is val, and the remaining ~64% is the
pool that training images are drawn from.

Writes data/splits/rice_grouped.csv (path, label, background, group,
split_r0..split_r4) and data/splits/rice_grouped_stats.json.

Usage: python scripts/make_rice_splits.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 5
# background: (inliers to link any pair, inliers to link a same-label pair).
# White-background photos are not linked: there, ORB inliers come from the
# parallel leaf veins and are higher for different-label pairs than for
# same-label pairs, and no pair is a near copy (docs/rice.md).
NEVER = 10**9
RULES = {"field": (20, 15), "white": (NEVER, NEVER)}


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def main():
    df = pd.read_csv(ROOT / "data/meta/dhan_cached.csv", keep_default_na=False)
    idx = {p: i for i, p in enumerate(df["path"])}
    pairs = pd.read_csv(ROOT / "data/meta/dhan_orb_pairs.csv")
    any_t = pairs["background"].map(lambda b: RULES[b][0])
    same_t = pairs["background"].map(lambda b: RULES[b][1])
    linked = pairs[(pairs["inliers"] >= any_t) | (pairs["same_label"] & (pairs["inliers"] >= same_t))]

    parent = list(range(len(df)))
    for a, b in zip(linked["a"], linked["b"]):
        ra, rb = find(parent, idx[a]), find(parent, idx[b])
        if ra != rb:
            parent[ra] = rb
    df["group"] = [find(parent, i) for i in range(len(df))]
    df["group"] = df["group"].map({g: n for n, g in enumerate(df["group"].unique())})
    mixed = df.groupby("group")["label"].nunique()
    sizes = df.groupby("group").size()

    strat = df["label"] + "|" + df["background"]
    for r in range(REPEATS):
        split = np.array(["pool"] * len(df), dtype=object)
        rest, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(df, strat, df["group"]))
        split[test] = "test"
        sub = df.iloc[rest]
        _, val = next(StratifiedGroupKFold(5, shuffle=True, random_state=100 + r)
                      .split(sub, strat.iloc[rest], sub["group"]))
        split[rest[val]] = "val"
        df[f"split_r{r}"] = split

    out = ROOT / "data/splits/rice_grouped.csv"
    df[["path", "label", "background", "group"] + [f"split_r{r}" for r in range(REPEATS)]].to_csv(out, index=False)
    stats = {
        "rules": RULES,
        "linked_pairs": {b: int((linked["background"] == b).sum()) for b in RULES},
        "linked_pairs_different_label": int((~linked["same_label"]).sum()),
        "groups": int(df["group"].nunique()),
        "groups_with_more_than_one_image": int((sizes > 1).sum()),
        "images_in_such_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "groups_with_mixed_labels": int((mixed > 1).sum()),
        "repeats": {f"r{r}": {s: {b: int(((df[f"split_r{r}"] == s) & (df["background"] == b)).sum())
                                  for b in ("white", "field")} for s in ("pool", "val", "test")}
                    for r in range(REPEATS)},
    }
    json.dump(stats, open(ROOT / "data/splits/rice_grouped_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
