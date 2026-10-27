"""Group near-duplicate images by perceptual hash (union-find over pHash pairs).

Reads an index CSV made by check_data.py (needs 'phash' and 'split' columns),
adds a 'dup_group' column (images within --max-d bits of each other, chained,
share a group) and prints how many groups cross the existing split. Used to
build leakage-free splits: a whole dup_group always goes to one side.

Usage: python scripts/near_duplicates.py data/meta/plantvillage.csv [--max-d 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_data import popcount64  # noqa: E402


def groups(hashes, max_d, chunk=256):
    h = np.array(hashes, dtype=np.uint64)
    parent = np.arange(len(h))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i0 in range(0, len(h), chunk):
        d = popcount64(h[i0:i0 + chunk, None] ^ h[None, :])
        for i, j in zip(*np.nonzero(d <= max_d)):
            i += i0
            if j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(len(h))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--max-d", type=int, default=6)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, keep_default_na=False)
    ok = df["phash"] != ""
    g = np.full(len(df), -1)
    g[ok.values] = groups([int(x, 16) for x in df.loc[ok, "phash"]], args.max_d)
    df["dup_group"] = g
    df.to_csv(args.csv, index=False)

    sizes = df[ok].groupby("dup_group").size()
    multi = sizes[sizes > 1]
    print(f"{len(df)} images, {len(multi)} groups with >1 image covering {multi.sum()} images")
    if "split" in df:
        spl = df[ok].groupby("dup_group")["split"].nunique()
        cross = spl[spl > 1].index
        x = df[df["dup_group"].isin(cross)]
        print(f"groups crossing splits: {len(cross)}; images in them: {len(x)}; "
              f"of which test: {(x['split'] == 'test').sum()}")
    if "label" in df:
        lab = df[ok].groupby("dup_group")["label"].nunique()
        print(f"groups mixing labels: {(lab > 1).sum()}")


if __name__ == "__main__":
    main()
