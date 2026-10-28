"""Tests for the demo API (needs models/ from scripts/export_onnx.py or the release).

Run: python -m pytest tests -q
"""
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import META, SESSIONS, app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def png(w=300, h=200, seed=0):
    rgb = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def post(data, model, name="leaf.png"):
    return client.post("/predict", files={"file": (name, data)}, data={"model": model})


def test_page_and_models():
    assert client.get("/").status_code == 200
    m = client.get("/models").json()
    assert set(m) == set(SESSIONS) == {"lab", "field", "rice"}
    assert len(m["field"]["classes"]) == 28 and len(m["rice"]["classes"]) == 5


@pytest.mark.parametrize("model", ["lab", "field", "rice"])
def test_predict_shape(model):
    r = post(png(), model)
    assert r.status_code == 200
    j = r.json()
    probs = [p["probability"] for p in j["predictions"]]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-6


def test_field_model_only_plantdoc_classes():
    allowed = set(META["field"]["allowed_classes"])
    for seed in range(5):
        top = post(png(seed=seed), "field").json()["predictions"]
        assert {p["class"] for p in top} <= allowed


def test_errors():
    assert post(b"not an image", "field").status_code == 400
    assert post(png(), "nope").status_code == 400


def _first(pattern):
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        pytest.skip(f"no data for {pattern}")
    return hits[0]


def test_real_plantdoc_photo():
    path = _first("data/raw/plantdoc/test/Tomato leaf late blight/*")
    j = post(path.read_bytes(), "field", path.name).json()
    assert j["predictions"][0]["class"].startswith("Tomato___")


def test_real_full_size_rice_photo():
    path = _first("data/raw/dhan_shomadhan/**/*.jpg")
    j = post(path.read_bytes(), "rice", path.name).json()
    assert len(j["predictions"]) == 5
    print(path.name, j["predictions"][0], j["timing_ms"])
