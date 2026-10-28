"""Export the demo models to ONNX, check them, and time them on the CPU (step 7).

Models (each chosen on validation in its step):
  lab    runs/pv_efficientnet_b0      step 2, 38 PlantVillage classes
  field  runs/ft/pv_kall_s1           step 4, PlantVillage start fine-tuned on
                                      all PlantDoc train images; only the 28
                                      PlantDoc classes are allowed at inference
  rice   runs/rice/imagenet_kall_r4   step 5, ImageNet start (higher mean val
                                      than the PlantVillage start), repeat with
                                      the best val macro-F1

Checks, for each model:
  1. pdd.preprocess (PIL + numpy, used by the demo) gives the same input as
     the torchvision transform used in evaluation;
  2. ONNX Runtime logits match PyTorch logits;
  3. ONNX predictions on the model's test images match the saved PyTorch
     predictions (lab: first 1,000 PlantVillage test images; field: PlantDoc
     test; rice: repeat-4 test);
  4. CPU latency for one image (ONNX Runtime, 1 thread and all threads;
     median of 100 runs after 10 warm-up runs).

Writes models/<name>.onnx (not committed; attached to the GitHub release) and
models/models.json (classes, preprocessing, scores, checks, timing, sha256).

Usage: python scripts/export_onnx.py
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdd import data, models, preprocess  # noqa: E402

OUT = ROOT / "models"


def load(run):
    ck = torch.load(ROOT / run / "best.pt", map_location="cpu")
    m = models.build(ck["arch"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def plantdoc_classes():
    return sorted(pd.read_csv(ROOT / "configs/plantdoc_to_plantvillage.csv")["plantvillage_class"])


def spec():
    pv_test = pd.read_csv(ROOT / "runs/pv_efficientnet_b0/pred_pv_test.csv", keep_default_na=False)
    lab_ev = json.load(open(ROOT / "runs/pv_efficientnet_b0/eval_pv_test.json"))["all"]
    lab_pd = json.load(open(ROOT / "runs/pv_efficientnet_b0/eval_plantdoc.json"))["all"]
    field_pd = json.load(open(ROOT / "runs/ft/pv_kall_s1/eval_plantdoc.json"))["all"]
    field_pv = json.load(open(ROOT / "runs/ft/pv_kall_s1/eval_pv_test.json"))["all"]
    rice = json.load(open(ROOT / "runs/rice/imagenet_kall_r4/test.json"))
    rice_pred = pd.read_csv(ROOT / "runs/rice/imagenet_kall_r4/pred_test.csv", keep_default_na=False)
    pd_pred = lambda run: pd.read_csv(ROOT / run / "pred_plantdoc.csv", keep_default_na=False)
    return {
        "lab": dict(run="runs/pv_efficientnet_b0", kind="leaf", allowed=None,
                    title="Lab model (PlantVillage, 38 classes)",
                    note="Trained on PlantVillage lab photos only. Near-perfect on lab photos, "
                         "about 31% on field photos.",
                    scores={"plantvillage_test_acc": lab_ev["accuracy"], "plantdoc_test_acc": lab_pd["accuracy"]},
                    check=(ROOT / "data/raw/plantvillage", pv_test.head(1000))),
        "field": dict(run="runs/ft/pv_kall_s1", kind="leaf", allowed=plantdoc_classes(),
                      title="Field model (PlantDoc, 28 classes)",
                      note="PlantVillage model fine-tuned on PlantDoc field/web photos. About 64% on "
                           "field photos; forgets lab photos and the 10 classes PlantDoc lacks.",
                      scores={"plantdoc_test_acc": field_pd["accuracy"], "plantvillage_test_acc": field_pv["accuracy"]},
                      check=(ROOT / "data/raw/plantdoc", pd_pred("runs/ft/pv_kall_s1"))),
        "rice": dict(run="runs/rice/imagenet_kall_r4", kind="rice", allowed=None,
                     title="Rice model (Dhan-Shomadhan, 5 diseases)",
                     note="Bangladeshi rice leaves, white-sheet and field photos. About 80% on its "
                          "test split; weaker on field photos.",
                     scores={"rice_test_acc": rice["all"]["accuracy"], "rice_test_acc_white": rice["white"]["accuracy"],
                             "rice_test_acc_field": rice["field"]["accuracy"]},
                     check=(ROOT / "data/cache/dhan_shomadhan", rice_pred)),
    }


def prep_np(kind, im, cached=False):
    return preprocess.leaf_square(im) if kind == "leaf" else preprocess.rice_tall(im, cached=cached)


def prep_torch(kind, im):
    tf = data.eval_transform(160) if kind == "leaf" else data.rice_eval_transform()
    return tf(im.convert("RGB"))[None].numpy()


def predict_allowed(z, classes, allowed):
    if allowed:
        z = np.where(np.isin(classes, allowed), z, -np.inf)
    return np.array(classes)[z.argmax(-1)]


def latency(path, x, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    for _ in range(10):
        s.run(None, {"image": x})
    t = []
    for _ in range(100):
        t0 = time.perf_counter()
        s.run(None, {"image": x})
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 90))


def main():
    OUT.mkdir(exist_ok=True)
    meta = {"created_with": {"torch": torch.__version__, "onnxruntime": ort.__version__,
                             "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()},
            "models": {}}
    for name, s in spec().items():
        model, ck = load(s["run"])
        classes = ck["classes"]
        root, ref = s["check"]
        h, w = (160, 160) if s["kind"] == "leaf" else data.RICE_SIZE
        path = OUT / f"{name}.onnx"
        torch.onnx.export(model, torch.zeros(1, 3, h, w), str(path), input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                          dynamo=False)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        # 1-2: preprocessing and logits on 20 images
        prep_diff, logit_diff = 0.0, 0.0
        for p in ref["path"].head(20):
            with Image.open(root / p) as im:
                a, b = prep_np(s["kind"], im, cached=True), prep_torch(s["kind"], im)
            prep_diff = max(prep_diff, float(np.abs(a - b).max()))
            with torch.no_grad():
                zt = model(torch.from_numpy(b)).numpy()
            logit_diff = max(logit_diff, float(np.abs(sess.run(None, {"image": b})[0] - zt).max()))

        # 3: prediction agreement over the test images, demo preprocessing end to end
        preds = []
        for p in ref["path"]:
            with Image.open(root / p) as im:
                z = sess.run(None, {"image": prep_np(s["kind"], im, cached=True)})[0][0]
            preds.append(predict_allowed(z, classes, s["allowed"]))
        preds = np.array(preds)
        agree = float((preds == ref["pred"].values).mean())
        acc = float((preds == ref["label"].values).mean())

        # 4: CPU latency, batch of 1
        x = np.zeros((1, 3, h, w), np.float32)
        t1, t1_90 = latency(path, x, 1)
        ta, ta_90 = latency(path, x, 0)

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        meta["models"][name] = {
            "file": path.name, "sha256": digest, "size_mb": round(path.stat().st_size / 1e6, 1),
            "title": s["title"], "note": s["note"], "source_run": s["run"], "arch": ck["arch"],
            "input": {"kind": s["kind"], "height": h, "width": w, "mean": preprocess.MEAN.tolist(),
                      "std": preprocess.STD.tolist()},
            "classes": classes, "allowed_classes": s["allowed"], "scores": s["scores"],
            "checks": {"max_preprocess_diff": prep_diff, "max_logit_diff": logit_diff,
                       "test_images_checked": len(ref), "prediction_agreement_with_pytorch": agree,
                       "onnx_accuracy_on_checked_images": acc},
            "cpu_latency_ms": {"1_thread_median": t1, "1_thread_p90": t1_90,
                               "all_threads_median": ta, "all_threads_p90": ta_90},
        }
        print(name, json.dumps({k: meta["models"][name][k] for k in ("size_mb", "checks", "cpu_latency_ms")}), flush=True)
    json.dump(meta, open(OUT / "models.json", "w"), indent=1)


if __name__ == "__main__":
    main()
