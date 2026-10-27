"""Check the three datasets and build index CSVs used by training.

For each dataset: count images per class, make sure every image decodes, and
look for exact (MD5) and near (perceptual hash) duplicates that could leak
between splits:
  - PlantVillage: are the official train/test splits leaf-disjoint?
  - PlantDoc: duplicates between the official train and test folders, and
    PlantDoc images that also appear in PlantVillage.
  - Dhan-Shomadhan: duplicates within a background and across
    white/field backgrounds.

Writes data/meta/{plantvillage,plantdoc,dhan_shomadhan}.csv and
data/meta/data_check.json. Usage: python scripts/check_data.py
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "data" / "downloads"
PV_DIR = ROOT / "data" / "raw" / "plantvillage"  # holds raw/color/<class>/<file>
PD_DIR = ROOT / "data" / "raw" / "plantdoc"  # from scripts/export_plantdoc.py
DS_DIR = ROOT / "data" / "raw" / "dhan_shomadhan"
META = ROOT / "data" / "meta"
NEAR = 6  # max pHash Hamming distance (of 64 bits) counted as a near-duplicate


def file_info(path):
    """MD5, 64-bit pHash, size and mode; error string if the image does not decode."""
    path = Path(path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as im:
            im.load()
            w, h, mode = im.width, im.height, im.mode
            ph = int(str(imagehash.phash(im.convert("RGB"))), 16)
        return {"md5": md5, "phash": ph, "width": w, "height": h, "mode": mode, "error": ""}
    except Exception as e:
        return {"md5": md5, "phash": None, "width": 0, "height": 0, "mode": "", "error": repr(e)}


def hash_all(paths):
    with ProcessPoolExecutor() as pool:
        return list(pool.map(file_info, paths, chunksize=64))


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def near_pairs(a, b, same=False, max_d=NEAR):
    """Index pairs (i, j, distance) with Hamming(a[i], b[j]) <= max_d."""
    a = np.array(a, dtype=np.uint64)
    b = np.array(b, dtype=np.uint64)
    out = []
    for i0 in range(0, len(a), 512):
        d = popcount64(a[i0:i0 + 512, None] ^ b[None, :])
        for i, j in zip(*np.nonzero(d <= max_d)):
            i += i0
            if same and j <= i:
                continue
            out.append((int(i), int(j), int(d[i - i0, j])))
    return out


def pv_leaf_id(rel_path, leaf_map):
    """Same rule as the official Hugging Face loader (plant_village.py)."""
    parts = rel_path.split("/")
    cls, name = parts[2], parts[3]
    ident = name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()
    sugg = leaf_map.get(ident.lower().strip())
    if not sugg:
        return f"fallback_{ident}"
    if len(sugg) == 1:
        return sugg[0]
    return next((s for s in sugg if cls in s), f"fallback_{ident}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def check_plantvillage():
    leaf_map = json.load(open(DL / "plantvillage" / "leaf_grouping" / "leaf-map.json"))
    rows = []
    for split in ("train", "test"):
        for line in open(DL / "plantvillage" / "splits" / f"color_{split}.txt"):
            rel = line.strip()
            if rel:
                rows.append({"path": rel, "label": rel.split("/")[2], "split": split,
                             "leaf_id": pv_leaf_id(rel, leaf_map)})
    missing = [r["path"] for r in rows if not (PV_DIR / r["path"]).exists()]
    on_disk = {str(p.relative_to(PV_DIR)) for p in (PV_DIR / "raw" / "color").rglob("*") if p.is_file()}
    not_in_splits = sorted(on_disk - {r["path"] for r in rows})
    infos = hash_all([PV_DIR / r["path"] for r in rows])
    for r, inf in zip(rows, infos):
        r.update({k: inf[k] for k in ("md5", "width", "height", "error")})
        r["phash"] = f"{inf['phash']:016x}" if inf["phash"] is not None else ""

    leaves = defaultdict(set)
    for r in rows:
        leaves[r["leaf_id"]].add(r["split"])
    md5_splits = defaultdict(set)
    for r in rows:
        md5_splits[r["md5"]].add(r["split"])
    write_csv(META / "plantvillage.csv", rows)
    return {
        "images_in_splits": len(rows),
        "per_split": dict(Counter(r["split"] for r in rows)),
        "classes": len({r["label"] for r in rows}),
        "per_class": dict(sorted(Counter(r["label"] for r in rows).items())),
        "missing_files": len(missing),
        "files_on_disk_not_in_splits": len(not_in_splits),
        "decode_errors": sum(bool(r["error"]) for r in rows),
        "leaf_ids": len(leaves),
        "fallback_leaf_ids": sum(k.startswith("fallback_") for k in leaves),
        "leaf_ids_in_both_splits": sum(len(s) > 1 for s in leaves.values()),
        "exact_duplicate_groups": sum(1 for r in Counter(r["md5"] for r in rows).values() if r > 1),
        "exact_duplicates_across_splits": sum(len(s) > 1 for s in md5_splits.values()),
    }, rows


def check_plantdoc(pv_rows):
    cmap = {r["plantdoc_class"]: r["plantvillage_class"]
            for r in csv.DictReader(open(ROOT / "configs" / "plantdoc_to_plantvillage.csv"))}
    rows = []
    for split in ("train", "test"):
        for p in sorted((PD_DIR / split).rglob("*")):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append({"path": str(p.relative_to(PD_DIR)), "split": split,
                             "plantdoc_class": p.parent.name,
                             "plantvillage_class": cmap.get(p.parent.name, "")})
    infos = hash_all([PD_DIR / r["path"] for r in rows])
    for r, inf in zip(rows, infos):
        r.update({k: inf[k] for k in ("md5", "width", "height", "mode", "error")})
        r["phash_int"] = inf["phash"]
    ok = [r for r in rows if r["phash_int"] is not None]

    # Duplicates between official train and test.
    tr = [r for r in ok if r["split"] == "train"]
    te = [r for r in ok if r["split"] == "test"]
    tr_md5 = {r["md5"] for r in tr}
    cross = near_pairs([r["phash_int"] for r in te], [r["phash_int"] for r in tr])
    te_near = {i for i, _, _ in cross}
    exact_within = sum(c - 1 for c in Counter(r["md5"] for r in ok).values() if c > 1)
    near_within = near_pairs([r["phash_int"] for r in ok], [r["phash_int"] for r in ok], same=True)

    # PlantDoc images that also appear in PlantVillage.
    pv_ok = [r for r in pv_rows if r["phash"]]
    pv_md5 = {r["md5"] for r in pv_ok}
    pv_hash = [int(r["phash"], 16) for r in pv_ok]
    in_pv = near_pairs([r["phash_int"] for r in ok], pv_hash)
    pd_in_pv = {i for i, _, _ in in_pv}

    for i, r in enumerate(te):
        r["test_dup_of_train"] = r["md5"] in tr_md5 or i in te_near
    for i, r in enumerate(ok):
        r["near_dup_in_plantvillage"] = i in pd_in_pv or r["md5"] in pv_md5
    for r in rows:
        r["phash"] = f"{r.pop('phash_int'):016x}" if r.get("phash_int") is not None else ""
        r.setdefault("test_dup_of_train", "")
        r.setdefault("near_dup_in_plantvillage", "")
    write_csv(META / "plantdoc.csv", rows)
    return {
        "images": len(rows),
        "per_split": dict(Counter(r["split"] for r in rows)),
        "classes_train": len({r["plantdoc_class"] for r in rows if r["split"] == "train"}),
        "classes_test": len({r["plantdoc_class"] for r in rows if r["split"] == "test"}),
        "unmapped_classes": sorted({r["plantdoc_class"] for r in rows if not r["plantvillage_class"]}),
        "per_class": {s: dict(sorted(Counter(r["plantdoc_class"] for r in rows if r["split"] == s).items()))
                      for s in ("train", "test")},
        "decode_errors": [r["path"] for r in rows if r["error"]],
        "exact_duplicate_extra_copies": exact_within,
        f"near_duplicate_pairs_within_all(d<={NEAR})": len(near_within),
        "test_images_exact_dup_of_train": sum(r["md5"] in tr_md5 for r in te),
        f"test_images_near_dup_of_train(d<={NEAR})": len(te_near),
        f"images_near_dup_in_plantvillage(d<={NEAR})": len(pd_in_pv),
        "images_exact_dup_in_plantvillage": sum(r["md5"] in pv_md5 for r in ok),
        "min_side_px": int(min(min(r["width"], r["height"]) for r in ok)),
    }


def check_dhan():
    rows = list(csv.DictReader(open(DS_DIR / "manifest.csv")))
    infos = hash_all([DS_DIR / r["path"] for r in rows])
    for r, inf in zip(rows, infos):
        r.update({k: inf[k] for k in ("md5", "width", "height", "mode", "error")})
        r["phash_int"] = inf["phash"]
    ok = [r for r in rows if r["phash_int"] is not None]
    pairs = near_pairs([r["phash_int"] for r in ok], [r["phash_int"] for r in ok], same=True)
    kinds = Counter()
    for i, j, _ in pairs:
        a, b = ok[i], ok[j]
        k = "white-field" if a["background"] != b["background"] else a["background"]
        kinds[k + (" same-label" if a["label"] == b["label"] else " diff-label")] += 1
    for r in rows:
        r["phash"] = f"{r.pop('phash_int'):016x}" if r.get("phash_int") is not None else ""
    write_csv(META / "dhan_shomadhan.csv", rows)
    sizes = Counter(f"{r['width']}x{r['height']}" for r in ok)
    return {
        "images": len(rows),
        "per_background_class": {bg: dict(sorted(Counter(r["label"] for r in rows if r["background"] == bg).items()))
                                 for bg in ("white", "field")},
        "decode_errors": [r["path"] for r in rows if r["error"]],
        "exact_duplicate_extra_copies": sum(c - 1 for c in Counter(r["md5"] for r in ok).values() if c > 1),
        f"near_duplicate_pairs(d<={NEAR})": dict(kinds),
        "most_common_sizes": sizes.most_common(5),
    }


def main():
    report = {}
    pv, pv_rows = check_plantvillage()
    report["plantvillage"] = pv
    print("plantvillage done")
    report["plantdoc"] = check_plantdoc(pv_rows)
    print("plantdoc done")
    report["dhan_shomadhan"] = check_dhan()
    print("dhan_shomadhan done")
    META.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(META / "data_check.json", "w"), indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
