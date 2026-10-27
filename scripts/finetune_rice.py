"""Train a rice disease classifier on a few labelled Dhan-Shomadhan photos (steps 5-6).

Uses one repeat of data/splits/rice_grouped.csv (scripts/make_rice_splits.py):
k images per class are drawn from that repeat's pool, at most one per
near-duplicate group, with the repeat number as seed. --train-bg limits the
pool (and validation) to one background, for the step-6 background study.

Training runs for a fixed number of steps. Every --eval-every steps the model
is scored on the repeat's val images (same background as training), and the
checkpoint with the best val macro-F1 is kept. Only after training is that
checkpoint scored on the repeat's test images, split by background.

--init imagenet starts from ImageNet weights; a run directory (e.g.
runs/pv_efficientnet_b0) starts from that PlantVillage model with a new
5-class head.

Usage:
  python scripts/finetune_rice.py --init runs/pv_efficientnet_b0 --repeat 0 --shots 10 --out runs/rice/pv_k10_r0
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


def load_split(repeat, train_bg):
    df = pd.read_csv(ROOT / "data/splits/rice_grouped.csv", keep_default_na=False)
    df["split"] = df[f"split_r{repeat}"]
    src = df if train_bg == "all" else df[df["background"] == train_bg]
    return df, src[src["split"] == "pool"], src[src["split"] == "val"], df[df["split"] == "test"]


def sample_shots(pool, shots, seed):
    """k images per class, at most one per near-duplicate group (all groups if fewer)."""
    if shots == 0:
        return pool
    rng = np.random.default_rng(seed)
    picked = []
    for _, g in pool.groupby("label"):
        for gid in rng.permutation(g["group"].unique())[:shots]:
            rows = g[g["group"] == gid]
            picked.append(rows.iloc[rng.integers(len(rows))])
    return pd.DataFrame(picked)


def build_model(init, arch, n_classes):
    if init == "imagenet":
        return arch, models.build(arch, n_classes)
    ck = torch.load(Path(init) / "best.pt", map_location="cpu")
    model = models.build(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    head = model.fc if ck["arch"] == "resnet18" else model.classifier[-1]
    new = nn.Linear(head.in_features, n_classes)
    if ck["arch"] == "resnet18":
        model.fc = new
    else:
        model.classifier[-1] = new
    return ck["arch"], model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="'imagenet' or a run directory with best.pt")
    ap.add_argument("--arch", default="efficientnet_b0", choices=list(models.ARCHS))
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--shots", type=int, required=True, help="images per class; 0 = the whole pool")
    ap.add_argument("--train-bg", default="all", choices=["all", "white", "field"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=0, help="0 is fastest here: the images are small and RAM is tight")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed = args.repeat
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = data.rice_classes()
    arch, model = build_model(args.init, args.arch, len(classes))
    model.to(device)

    _, pool, val, test = load_split(args.repeat, args.train_bg)
    subset = sample_shots(pool, args.shots, seed)
    subset[["path", "label", "background", "group"]].to_csv(out / "train_subset.csv", index=False)
    train_ds = data.rice_list(subset, data.rice_train_transform())
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                          num_workers=args.workers, persistent_workers=args.workers > 0,
                          worker_init_fn=data.seed_worker, generator=g)
    # val and test are small: loaded in the main process (worker start-up costs more than it saves)
    val_dl = DataLoader(data.rice_list(val, data.rice_eval_transform()), 64)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    json.dump(vars(args) | {"arch": arch, "train_images": len(train_ds), "val_images": len(val),
                            "test_images": len(test), "device": str(device)},
              open(out / "config.json", "w"), indent=2)

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
                    torch.save({"arch": arch, "classes": classes, "size": list(data.RICE_SIZE), "epoch": step,
                                "state_dict": model.state_dict()}, out / "best.pt")
                json.dump(history, open(out / "history.json", "w"), indent=2)
                model.train()
            if step >= args.steps:
                break
    json.dump(max(history, key=lambda r: r["val_macro_f1"]), open(out / "best_val.json", "w"), indent=2)

    # Test, once, with the checkpoint chosen on val.
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["state_dict"])
    test_dl = DataLoader(data.rice_list(test, data.rice_eval_transform()), 64)
    y, p = predict(model, test_dl, device)
    pd.DataFrame({"path": test["path"].values, "background": test["background"].values,
                  "label": np.array(classes)[y], "pred": np.array(classes)[p]}).to_csv(out / "pred_test.csv", index=False)
    bg = test["background"].values
    res = {"classes": classes, "all": summarize(y, p, len(classes), n_boot=0)}
    for b in ("white", "field"):
        res[b] = summarize(y[bg == b], p[bg == b], len(classes), n_boot=0)
    json.dump(res, open(out / "test.json", "w"), indent=1)
    print("test", {k: round(v["accuracy"], 4) for k, v in res.items() if k != "classes"}, flush=True)


if __name__ == "__main__":
    main()
