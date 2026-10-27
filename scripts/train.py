"""Fine-tune an ImageNet backbone on PlantVillage (grouped split).

The best epoch is chosen by validation macro-F1; the test set is not
touched here (use scripts/evaluate.py once training is final).

Usage: python scripts/train.py --arch resnet18 --epochs 8 --out runs/pv_resnet18
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdd import data, models  # noqa: E402
from pdd.metrics import summarize  # noqa: E402


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(models.ARCHS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-col", default="split", help="'split' (grouped) or 'official_split'")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train images")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    classes = data.plantvillage_classes()
    train_ds = data.plantvillage_split("train", data.train_transform(args.size), args.split_col)
    val_ds = data.plantvillage_split("val", data.eval_transform(args.size), args.split_col)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        for ds, n in ((train_ds, args.limit), (val_ds, args.limit // 4)):
            keep = rng.permutation(len(ds))[:n]
            ds.paths = [ds.paths[i] for i in keep]
            ds.targets = [ds.targets[i] for i in keep]
    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0,
              worker_init_fn=data.seed_worker)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, 128, shuffle=False, **kw)

    model = models.build(args.arch, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_dl)
    warm = len(train_dl) // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    config = vars(args) | {"device": str(device), "params": models.count_params(model),
                           "train_images": len(train_ds), "val_images": len(val_ds), "classes": classes}
    json.dump(config, open(out / "config.json", "w"), indent=2)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        train_time = time.time() - t0
        yv, pv = predict(model, val_dl, device)
        m = summarize(yv, pv, len(classes), n_boot=0)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "val_acc": m["accuracy"],
               "val_macro_f1": m["macro_f1"], "val_errors": m["errors"],
               "train_sec": round(train_time, 1), "img_per_sec": round(seen / train_time, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save({"arch": args.arch, "classes": classes, "size": args.size, "epoch": epoch,
                        "state_dict": model.state_dict()}, out / "best.pt")
        json.dump(history, open(out / "history.json", "w"), indent=2)
    best_row = max(history, key=lambda r: r["val_macro_f1"])
    json.dump(best_row, open(out / "best_val.json", "w"), indent=2)
    print("best", json.dumps(best_row))


if __name__ == "__main__":
    main()
