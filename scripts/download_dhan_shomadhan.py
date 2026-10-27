"""Download Dhan-Shomadhan (Mendeley Data, DOI 10.17632/znsxdctwtt.1, CC BY 4.0).

Uses the public Mendeley Data API: lists every folder, downloads each image,
checks it against the SHA-256 the API publishes, and writes a manifest CSV.
Re-running skips files that are already present and verified.

Usage: python scripts/download_dhan_shomadhan.py [--out data/raw/dhan_shomadhan]
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/znsxdctwtt"
VERSION = 1
# Mendeley returns 403 to urllib's default User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (dataset download script)"}

# Folder names on Mendeley contain typos and trailing spaces; map them to clean names.
BACKGROUND = {"White Background": "white", "Field Background": "field"}
CLASS = {
    "Brown Spot": "brown_spot", "Browon Spot": "brown_spot",
    "Leaf Scaled": "leaf_scald",
    "Rice Blast": "rice_blast",
    "Rice Tungro": "rice_tungro", "Rice Turgro": "rice_tungro",
    "Shath Blight": "sheath_blight", "Sheath Blight": "sheath_blight",
}


def get_json(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(item, tries=4):
    dest, url, expected = item["dest"], item["url"], item["sha256"]
    if dest.exists() and sha256(dest) == expected:
        return "skip"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if sha256(tmp) != expected:
                raise ValueError("sha256 mismatch")
            tmp.rename(dest)
            return "ok"
        except Exception as e:
            if i == tries - 1:
                return f"FAIL {e}"
            time.sleep(2 ** i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/dhan_shomadhan")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)

    folders = get_json(f"{API}/folders/{VERSION}")
    by_id = {f["id"]: f for f in folders}
    items = []
    for f in folders:
        parent = by_id.get(f.get("parent_id"))
        if parent is None or parent["name"].strip() not in BACKGROUND:
            continue  # only leaf folders <background>/<class> hold images
        bg = BACKGROUND[parent["name"].strip()]
        cls = CLASS[f["name"].strip()]
        files = get_json(f"{API}/files?folder_id={f['id']}&version={VERSION}")
        (out / bg / cls).mkdir(parents=True, exist_ok=True)
        for x in files:
            cd = x["content_details"]
            items.append({
                "background": bg, "label": cls, "filename": x["filename"],
                "size": x["size"], "sha256": cd["sha256_hash"],
                "url": cd["download_url"], "dest": out / bg / cls / x["filename"],
                "mendeley_folder": f"{parent['name'].strip()}/{f['name'].strip()}",
            })
    print(f"{len(items)} files listed")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(download, items))
    failed = [(it["dest"], r) for it, r in zip(items, results) if r.startswith("FAIL")]
    print({k: results.count(k) for k in ("ok", "skip")}, "failed:", len(failed))
    for d, r in failed:
        print(" ", d, r)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "background", "label", "size", "sha256", "mendeley_folder"])
        for it in sorted(items, key=lambda i: str(i["dest"])):
            w.writerow([it["dest"].relative_to(out), it["background"], it["label"],
                        it["size"], it["sha256"], it["mendeley_folder"]])


if __name__ == "__main__":
    main()
