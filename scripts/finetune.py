"""Fine-tune on a few labelled PlantDoc images (step 4).

k images per class are drawn from the PlantDoc train split
(data/splits/plantdoc_grouped.csv), one image per duplicate group, with the
seed deciding which. Training runs for a fixed number of steps; every
--eval-every steps the model is scored on PlantDoc val, and the checkpoint
with the best val macro-F1 is kept. PlantDoc test is not touched here.

The model keeps the 38 PlantVillage classes as outputs, so the result can be
scored exactly like the step-2/3 models.

Usage:
  python scripts/finetune.py --init runs/pv_efficientnet_b0 --shots 10 --seed 0 --out runs/ft/pv_k10_s0
  python scripts/finetune.py --init imagenet --arch efficientnet_b0 --shots 10 --seed 0 --out runs/ft/in_k10_s0
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pdd import data, models  # noqa: E402
from pdd.metrics import summarize  # noqa: E402
from train import predict  # noqa: E402


def sample_shots(shots, seed):
    """k images per class from PlantDoc train, at most one per duplicate group."""
    df = pd.read_csv(ROOT / "data/splits/plantdoc_grouped.csv", keep_default_na=False)
    df = df[df["split"] == "train"]
    if shots == 0:
        return df
    rng = np.random.default_rng(seed)
    picked = []
    for _, g in df.groupby("label"):
        groups = rng.permutation(g["group"].unique())[:shots]
        for gid in groups:
            rows = g[g["group"] == gid]
            picked.append(rows.iloc[rng.integers(len(rows))])
    return pd.DataFrame(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="'imagenet' or a run directory with best.pt")
    ap.add_argument("--arch", default="efficientnet_b0", choices=list(models.ARCHS))
    ap.add_argument("--shots", type=int, required=True, help="images per class; 0 = all of PlantDoc train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--aug", default="default", choices=["default", "strong"])
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = data.plantvillage_classes()

    if args.init == "imagenet":
        arch = args.arch
        model = models.build(arch, len(classes))
    else:
        ck = torch.load(Path(args.init) / "best.pt", map_location="cpu")
        assert ck["classes"] == classes and ck["size"] == args.size
        arch = ck["arch"]
        model = models.build(arch, len(classes), pretrained=False)
        model.load_state_dict(ck["state_dict"])
    model.to(device)

    subset = sample_shots(args.shots, args.seed)
    subset[["path", "label", "group"]].to_csv(out / "train_subset.csv", index=False)
    tf = data.strong_train_transform if args.aug == "strong" else data.train_transform
    train_ds = data.ImageList(ROOT / "data/raw/plantdoc", subset["path"], subset["label"], classes, tf(args.size))
    val_ds = data.plantdoc_split("val", data.eval_transform(args.size))
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                          num_workers=args.workers, persistent_workers=True,
                          worker_init_fn=data.seed_worker, generator=g)
    val_dl = DataLoader(val_ds, 64, shuffle=False, num_workers=args.workers, persistent_workers=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    json.dump(vars(args) | {"arch": arch, "train_images": len(train_ds), "val_images": len(val_ds),
                            "device": str(device)}, open(out / "config.json", "w"), indent=2)

    history, best, step, loss_sum, seen = [], -1.0, 0, 0.0, 0
    while step < args.steps:
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            loss_sum += loss.item() * len(y)
            seen += len(y)
            if step % args.eval_every == 0 or step == args.steps:
                m = summarize(*predict(model, val_dl, device), len(classes), n_boot=0)
                row = {"step": step, "train_loss": loss_sum / seen, "val_acc": m["accuracy"],
                       "val_macro_f1": m["macro_f1"]}
                history.append(row)
                print(json.dumps(row), flush=True)
                loss_sum, seen = 0.0, 0
                if m["macro_f1"] > best:
                    best = m["macro_f1"]
                    torch.save({"arch": arch, "classes": classes, "size": args.size, "epoch": step,
                                "state_dict": model.state_dict()}, out / "best.pt")
                json.dump(history, open(out / "history.json", "w"), indent=2)
                model.train()
            if step >= args.steps:
                break
    json.dump(max(history, key=lambda r: r["val_macro_f1"]), open(out / "best_val.json", "w"), indent=2)


if __name__ == "__main__":
    main()
