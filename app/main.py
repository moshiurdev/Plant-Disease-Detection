"""FastAPI demo: upload a leaf photo, get the top predictions from an ONNX model (step 7).

Runs on the CPU with ONNX Runtime; torch is not needed. The models and their
metadata come from models/ (scripts/export_onnx.py or the GitHub release).

  uvicorn app.main:app --port 8000      then open http://localhost:8000

Endpoints:
  GET  /          upload page
  GET  /models    the available models, their scores and notes
  POST /predict   form fields: file (image), model (lab | field | rice), top (default 5)
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdd import preprocess  # noqa: E402

MODELS_DIR = ROOT / "models"
MAX_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000

META = json.load(open(MODELS_DIR / "models.json"))["models"]
SESSIONS = {name: ort.InferenceSession(str(MODELS_DIR / m["file"]), providers=["CPUExecutionProvider"])
            for name, m in META.items() if (MODELS_DIR / m["file"]).exists()}

app = FastAPI(title="Plant Disease Detection demo")


def pretty(name):
    if "___" in name:
        crop, disease = name.split("___", 1)
        crop = crop.replace("_(including_sour)", "").replace("_(maize)", "").replace(",_bell", " (bell)")
        return f"{crop.replace('_', ' ')}: {disease.replace('_', ' ').strip()}"
    return name.replace("_", " ").capitalize()


@app.get("/models")
def list_models():
    return {name: {k: m[k] for k in ("title", "note", "scores", "cpu_latency_ms")} | {
                "classes": m["allowed_classes"] or m["classes"]}
            for name, m in META.items() if name in SESSIONS}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("field"), top: int = Form(5)):
    if model not in SESSIONS:
        raise HTTPException(400, f"unknown model '{model}'; choose from {sorted(SESSIONS)}")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image larger than 20 MB")
    t0 = time.perf_counter()
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "could not read the file as an image")
    m = META[model]
    if m["input"]["kind"] == "rice":
        x = preprocess.rice_tall(im)
    else:
        x = preprocess.leaf_square(ImageOps.exif_transpose(im))
    t1 = time.perf_counter()
    z = SESSIONS[model].run(None, {"image": x})[0][0]
    t2 = time.perf_counter()

    classes = np.array(m["classes"])
    if m["allowed_classes"]:
        z = np.where(np.isin(classes, m["allowed_classes"]), z, -np.inf)
    p = preprocess.softmax(z)
    order = np.argsort(-p)[: max(1, min(top, len(classes)))]
    return {
        "model": model,
        "title": m["title"],
        "predictions": [{"class": str(classes[i]), "label": pretty(str(classes[i])), "probability": float(p[i])}
                        for i in order],
        "timing_ms": {"preprocess": round((t1 - t0) * 1000, 1), "inference": round((t2 - t1) * 1000, 1)},
        "note": m["note"],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    options = "".join(f'<option value="{n}"{" selected" if n == "field" else ""}>{META[n]["title"]}</option>'
                      for n in SESSIONS)
    return PAGE.replace("{{options}}", options)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plant Disease Detection</title>
<style>
:root { --bg:#f7f8f5; --card:#fff; --ink:#1d2a1f; --muted:#5c6b5e; --line:#dfe5dc; --accent:#2f7d4a; --bar:#cfe6d6; }
@media (prefers-color-scheme: dark) { :root { --bg:#121712; --card:#1b221c; --ink:#e6efe7; --muted:#9aaa9c;
  --line:#2c362d; --accent:#6cc28a; --bar:#2b4a35; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:760px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:1.5rem; margin:0 0 4px; } .sub { color:var(--muted); margin:0 0 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px; }
label { display:block; font-weight:600; margin:0 0 6px; } select, input[type=file] { width:100%; font:inherit; }
select { padding:8px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--ink); }
button { margin-top:14px; font:inherit; font-weight:600; padding:10px 18px; border:0; border-radius:8px;
  background:var(--accent); color:#fff; cursor:pointer; } button:disabled { opacity:.5; cursor:default; }
.row { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.4fr); gap:16px; align-items:start; }
@media (max-width:600px) { .row { grid-template-columns:1fr; } }
img { max-width:100%; max-height:360px; border-radius:8px; display:block; margin:0 auto; }
.pred { margin:8px 0; } .pred .name { display:flex; justify-content:space-between; gap:8px; font-size:.95rem; }
.track { height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
.fill { height:100%; background:var(--accent); } .note, .meta { color:var(--muted); font-size:.875rem; }
.err { color:#c0392b; }
</style></head><body><main>
<h1>Plant Disease Detection</h1>
<p class="sub">Upload a photo of a single leaf. The model runs on the CPU with ONNX Runtime.</p>
<form class="card" id="f">
  <label for="model">Model</label><select id="model" name="model">{{options}}</select>
  <p class="note" id="mnote"></p>
  <label for="file">Leaf photo</label><input id="file" name="file" type="file" accept="image/*" required>
  <button id="go" type="submit">Predict</button>
</form>
<div class="card row" id="out" hidden><div><img id="prev" alt="uploaded leaf"></div><div id="res"></div></div>
<p class="note">Research demo, not a diagnosis tool. Probabilities come from a model trained with label
smoothing and are not calibrated; on photos unlike its training data the model can be confidently wrong.</p>
</main><script>
const notes = {}; fetch('/models').then(r => r.json()).then(m => { for (const k in m) notes[k] = m[k].note; showNote(); });
const sel = document.getElementById('model'); sel.onchange = showNote;
function showNote() { document.getElementById('mnote').textContent = notes[sel.value] || ''; }
const res = document.getElementById('res');
document.getElementById('f').onsubmit = async e => {
  e.preventDefault(); const f = document.getElementById('file').files[0]; if (!f) return;
  const go = document.getElementById('go'); go.disabled = true;
  document.getElementById('prev').src = URL.createObjectURL(f); document.getElementById('out').hidden = false;
  res.textContent = 'Running...';
  const fd = new FormData(); fd.append('file', f); fd.append('model', sel.value);
  try {
    const r = await fetch('/predict', {method:'POST', body:fd}); const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    res.innerHTML = '';
    for (const p of j.predictions) {
      const d = document.createElement('div'); d.className = 'pred';
      const pct = (p.probability * 100).toFixed(1);
      d.innerHTML = '<div class="name"><span></span><span>' + pct + '%</span></div><div class="track"><div class="fill"></div></div>';
      d.querySelector('.name span').textContent = p.label; d.querySelector('.fill').style.width = pct + '%';
      res.appendChild(d);
    }
    const m = document.createElement('p'); m.className = 'meta';
    m.textContent = 'Preprocess ' + j.timing_ms.preprocess + ' ms, model ' + j.timing_ms.inference + ' ms';
    res.appendChild(m);
  } catch (err) { res.innerHTML = ''; const p = document.createElement('p'); p.className = 'err';
    p.textContent = 'Error: ' + err.message; res.appendChild(p); }
  go.disabled = false;
};
</script></body></html>
"""
