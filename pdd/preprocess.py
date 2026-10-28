"""Image preprocessing for the ONNX models, with PIL and numpy only (no torch).

These reproduce the evaluation transforms used in training
(pdd.data.eval_transform and pdd.data.rice_eval_transform);
scripts/export_onnx.py checks that they give the same input tensor.
"""
import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_input(im):
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # 1 x 3 x H x W


def leaf_square(im, size=160):
    """Shorter side to size*256/224, then a centred size x size crop (PlantVillage / PlantDoc models)."""
    im = im.convert("RGB")
    short = int(size * 256 / 224)
    w, h = im.size
    if w <= h:
        new = (short, int(short * h / w))
    else:
        new = (int(short * w / h), short)
    im = im.resize(new, Image.BILINEAR)
    w, h = im.size
    top, left = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    return _to_input(im.crop((left, top, left + size, top + size)))


def rice_cache(im, width=224):
    """Same steps as scripts/cache_rice.py: upright, portrait, 224 px wide."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width > im.height:
        im = im.transpose(Image.Transpose.ROTATE_90)
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def rice_tall(im, size=(256, 128), cached=False):
    """Rice model input: the cache step (unless the image is already cached), then resize to height x width."""
    if not cached:
        im = rice_cache(im)
    return _to_input(im.convert("RGB").resize((size[1], size[0]), Image.BILINEAR))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
