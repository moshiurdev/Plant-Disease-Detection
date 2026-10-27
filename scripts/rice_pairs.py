"""Find repeated shots of the same rice leaf/plant in Dhan-Shomadhan (step 5).

Thumbnail correlation finds almost no field-background candidates (field
scenes vary too much), and on white backgrounds any two thin leaves
correlate. So here ORB keypoints are matched for *every* pair of images with
the same background (cached copies, scripts/cache_rice.py) and the RANSAC
inliers of a partial affine fit are counted.

On white backgrounds the grey sheet itself has marks that repeat across
photos, so keypoints there are only taken on leaf pixels (HSV saturation
> 40, dilated); field photos use the whole image.

Pairs with different disease labels cannot be the same leaf, so their inlier
counts show what "different leaves" score. The linking threshold is set
above that distribution (see docs/rice.md).

Writes data/meta/dhan_orb_pairs.csv (pairs with >= 8 inliers) and
data/meta/dhan_orb_hist.csv (all pairs, counted by inliers, background and
same/different label).

Usage: python scripts/rice_pairs.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/dhan_shomadhan"
KEEP = 8
FEATS = {}


def leaf_mask(bgr):
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    return cv2.dilate((sat > 40).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))


def features(path):
    bgr = cv2.imread(str(CACHE / path))
    mask = leaf_mask(bgr) if path.startswith("white/") else None
    k, d = cv2.ORB_create(nfeatures=1000).detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask)
    return np.float32([p.pt for p in k]), d


def init(paths):
    for p in paths:
        FEATS[p] = features(p)


def inliers(a, b, matcher):
    (pa, da), (pb, db) = FEATS[a], FEATS[b]
    if da is None or db is None or len(pa) < 8 or len(pb) < 8:
        return 0
    good = [m[0] for m in matcher.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 6:
        return len(good) // 2
    _, mask = cv2.estimateAffinePartial2D(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]],
                                          method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return int(mask.sum()) if mask is not None else 0


def score_row(args):
    i, paths = args
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return [(paths[i], paths[j], inliers(paths[i], paths[j], matcher)) for j in range(i + 1, len(paths))]


def main():
    df = pd.read_csv(ROOT / "data/meta/dhan_cached.csv", keep_default_na=False)
    label = dict(zip(df["path"], df["label"]))
    rows = []
    for bg, g in df.groupby("background"):
        paths = list(g["path"])
        with ProcessPoolExecutor(4, initializer=init, initargs=(paths,)) as pool:
            for part in pool.map(score_row, [(i, paths) for i in range(len(paths))], chunksize=8):
                rows += [(a, b, n, bg) for a, b, n in part]
        print(bg, len(paths), "images done", flush=True)
    p = pd.DataFrame(rows, columns=["a", "b", "inliers", "background"])
    p["same_label"] = p["a"].map(label) == p["b"].map(label)
    p[p["inliers"] >= KEEP].to_csv(ROOT / "data/meta/dhan_orb_pairs.csv", index=False)
    hist = p.groupby(["background", "same_label", "inliers"]).size().rename("pairs").reset_index()
    hist.to_csv(ROOT / "data/meta/dhan_orb_hist.csv", index=False)
    for (bg, same), h in hist.groupby(["background", "same_label"]):
        c = h.set_index("inliers")["pairs"]
        print(bg, "same label" if same else "diff label", "pairs", c.sum(), "max inliers", c.index.max(),
              {t: int(c[c.index >= t].sum()) for t in (8, 10, 15, 20, 30, 50)})


if __name__ == "__main__":
    main()
