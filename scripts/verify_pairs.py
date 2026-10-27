"""Verify candidate duplicate pairs with keypoint matching (ORB + RANSAC).

Thumbnail correlation (scripts/dedup.py) finds candidates, but on PlantVillage
different leaves of the same shape on the same background also correlate
highly. Two photos of the same physical leaf share vein and lesion details,
so many ORB keypoints match under a single geometric transform; different
leaves give few such matches. For each candidate pair this adds 'inliers'
(RANSAC inliers of a partial affine fit, the second image also tried
mirrored) to the pairs CSV.

Usage: python scripts/verify_pairs.py data/meta/plantvillage_pairs.csv data/raw/plantvillage
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SIDE = 256


def load(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (SIDE, SIDE), interpolation=cv2.INTER_AREA)


def inliers(a, b, orb, matcher):
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = []
    for m in matcher.knnMatch(da, db, k=2):
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < 6:
        return len(good) // 2
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return int(mask.sum()) if mask is not None else 0


def score(pair, root):
    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    a, b = load(root / pair[0]), load(root / pair[1])
    return max(inliers(a, b, orb, matcher), inliers(a, cv2.flip(b, 1), orb, matcher))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_csv")
    ap.add_argument("root")
    args = ap.parse_args()
    p = pd.read_csv(args.pairs_csv, keep_default_na=False)
    with ProcessPoolExecutor() as pool:
        p["inliers"] = list(pool.map(partial(score, root=Path(args.root)), zip(p["a"], p["b"]), chunksize=64))
    p.to_csv(args.pairs_csv, index=False)
    print(p["inliers"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
