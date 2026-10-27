"""Export PlantDoc images from its git history into data/raw/plantdoc.

A normal checkout on macOS (case-insensitive file system) drops files whose
names differ only in case, e.g. 'CAR1.jpg' and 'car1.jpg'. This reads every
blob straight from git and gives the second file of such a pair a '__2'
suffix, so no image is lost. Writes manifest.csv with the original git path.

Usage: python scripts/export_plantdoc.py [--repo data/downloads/plantdoc] [--out data/raw/plantdoc]
"""
import argparse
import csv
import subprocess
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="data/downloads/plantdoc")
    ap.add_argument("--out", default="data/raw/plantdoc")
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)

    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    listing = subprocess.check_output(["git", "-C", repo, "ls-tree", "-r", "-z", "HEAD"], text=True)
    entries = []
    for rec in listing.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        _, kind, sha = meta.split()
        top = path.split("/")[0]
        if kind == "blob" and top in ("train", "test") and Path(path).suffix.lower() in EXTS:
            entries.append((path, sha))

    taken = set()
    rows = []
    for path, sha in sorted(entries):
        dest = Path(path)
        n = 1
        while str(dest).lower() in taken:
            n += 1
            dest = Path(path).with_name(f"{Path(path).stem}__{n}{Path(path).suffix}")
        taken.add(str(dest).lower())
        data = subprocess.check_output(["git", "-C", repo, "cat-file", "blob", sha])
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        (out / dest).write_bytes(data)
        rows.append({"path": str(dest), "git_path": path, "git_blob": sha, "renamed": str(dest) != path})

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "SOURCE.txt").write_text(
        f"https://github.com/pratikkayal/PlantDoc-Dataset commit {commit}\nLicense: CC BY 4.0\n")
    print(f"exported {len(rows)} images, renamed {sum(r['renamed'] for r in rows)} case-colliding files")


if __name__ == "__main__":
    main()
