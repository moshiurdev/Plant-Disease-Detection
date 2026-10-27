"""Find near-duplicate images by pixel correlation of small thumbnails.

Perceptual hashes are too coarse for PlantVillage (different leaves on the
same background in the same pose hash alike; see docs/data.md). Here every
image becomes a 32x32 grayscale thumbnail, mean-centred and L2-normalised, so
the dot product of two thumbnails is their Pearson correlation. All pairs
are compared with chunked matrix products; pairs above --min-corr are written
out, also checking a horizontally flipped copy.

Usage:
  python scripts/dedup.py data/meta/plantvillage.csv data/raw/plantvillage \
      --out data/meta/plantvillage_pairs.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SIZE = 32


def thumb(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    return a


def normalise(x):
    x = x.reshape(len(x), -1)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def load_thumbs(paths, cache):
    if cache.exists():
        t = np.load(cache)
        if len(t) == len(paths):
            return t
    with ProcessPoolExecutor() as pool:
        t = np.stack(list(pool.map(thumb, paths, chunksize=128)))
    np.save(cache, t)
    return t


def pairs_above(x, y, min_corr, same, chunk=1024):
    """(i, j, corr) with x[i]·y[j] >= min_corr; j > i when x and y are the same set."""
    out = []
    for i0 in range(0, len(x), chunk):
        s = x[i0:i0 + chunk] @ y.T
        ii, jj = np.nonzero(s >= min_corr)
        for i, j in zip(ii, jj):
            if same and j <= i + i0:
                continue
            out.append((i + i0, j, float(s[i, j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-corr", type=float, default=0.95)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)
    root = Path(args.root)
    t = load_thumbs([root / p for p in df["path"]], Path(args.out).with_suffix(".thumbs.npy"))
    x = normalise(t)
    xf = normalise(t[:, :, ::-1].copy())

    found = {}
    for flip, y in ((False, x), (True, xf)):
        for i, j, c in pairs_above(x, y, args.min_corr, same=not flip):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key not in found or c > found[key][0]:
                found[key] = (c, flip)
    rows = [{"a": df["path"][i], "b": df["path"][j], "corr": round(c, 4), "flipped": f,
             "label_a": df["label"][i] if "label" in df else "",
             "label_b": df["label"][j] if "label" in df else ""}
            for (i, j), (c, f) in sorted(found.items())]
    pd.DataFrame(rows, columns=["a", "b", "corr", "flipped", "label_a", "label_b"]).to_csv(args.out, index=False)
    print(f"{len(rows)} pairs with corr >= {args.min_corr} ({sum(r['flipped'] for r in rows)} flipped)")


if __name__ == "__main__":
    main()
