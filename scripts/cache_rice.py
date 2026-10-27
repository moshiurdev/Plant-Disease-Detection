"""Cache small copies of the Dhan-Shomadhan photos (steps 5-6).

The originals are 8-megapixel phone photos (1952 x 4160), which are too slow
to decode on every training step. Each image is turned upright (EXIF), rotated
to portrait so the leaf runs top to bottom, and saved 224 px wide (about
224 x 477) as JPEG quality 95 under data/cache/dhan_shomadhan/.

Usage: python scripts/cache_rice.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/dhan_shomadhan"
DST = ROOT / "data/cache/dhan_shomadhan"
WIDTH = 224


def cache(path):
    out = DST / path
    if out.exists():
        return
    with Image.open(SRC / path) as im:
        im.draft("RGB", (im.width // 4, im.height // 4))
        im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width > im.height:
        im = im.transpose(Image.Transpose.ROTATE_90)
    im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=95)


def main():
    df = pd.read_csv(ROOT / "data/meta/dhan_shomadhan.csv", keep_default_na=False)
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(cache, df["path"], chunksize=16))
    df[["path", "label", "background"]].to_csv(ROOT / "data/meta/dhan_cached.csv", index=False)
    print(len(df), "images cached in", DST)


if __name__ == "__main__":
    main()
