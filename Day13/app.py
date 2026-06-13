"""
Seafood Freshness — Flask App (YOLO)
=====================================
Upload a fish image → get freshness prediction

Usage:
    pip install flask ultralytics Pillow numpy
    python app.py
    Open: http://localhost:5000
"""

import os
import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Fish_grading_model\yolo_results\fish_freshness\weights\best.pt"
IMG_SIZE   = 224

FRESHNESS_MAP = {
    "eyes_fresh":    { "verdict": "FRESH",    "emoji": "✅", "color": "#02C39A", "tip": "Eyes are clear and bright — fish is fresh. Safe to consume." },
    "eyes_spoiled":  { "verdict": "SPOILED",  "emoji": "❌", "color": "#EF4444", "tip": "Eyes look cloudy or sunken — sign of spoilage. Do not consume." },
    "gills_fresh":   { "verdict": "FRESH",    "emoji": "✅", "color": "#02C39A", "tip": "Gills are red/pink and healthy — fish is fresh. Safe to consume." },
    "gills_spoiled": { "verdict": "SPOILED",  "emoji": "❌", "color": "#EF4444", "tip": "Gills appear grey/brown — sign of spoilage. Do not consume." },
}
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names
print(f"Model loaded ✅  Classes: {list(CLASS_NAMES.values())}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fish Freshness Inspector</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #050D12;
    --surface:  #0B1820;
    --surface2: #0F2233;
    --border:   #1C3A50;
    --accent:   #02C39A;
    --accent2:  #065A82;
    --red:      #EF4444;
    --text:     #E2EEF4;
    --muted:    #6B8FA3;
    --font-head: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* animated bg grid */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(2,195,154,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(2,195,154,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 24px;
  }

  /* Header */
  header {
    text-align: center;
    margin-bottom: 48px;
  }

  .logo-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .logo-icon {
    font-size: 2.4rem;
    filter: drop-shadow(0 0 12px rgba(2,195,154,0.6));
  }

  h1 {
    font-family: var(--font-head);
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    color: var(--accent);
    letter-spacing: -1px;
    text-shadow: 0 0 30px rgba(2,195,154,0.3);
  }

  .subtitle {
    font-size: 0.95rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
  }

  .badge-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 16px;
    flex-wrap: wrap;
  }

  .badge {
    font-family: var(--font-head);
    font-size: 0.7rem;
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: 2px;
    color: var(--muted);
    letter-spacing: 1px;
  }

  /* Upload area */
  .upload-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 32px;
    margin-bottom: 28px;
  }

  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: 4px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
  }

  .drop-zone::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at center, rgba(2,195,154,0.05) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.25s;
  }

  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(2,195,154,0.03);
  }

  .drop-zone:hover::after, .drop-zone.drag-over::after {
    opacity: 1;
  }

  .drop-icon { font-size: 3rem; margin-bottom: 12px; }

  .drop-text {
    font-size: 1rem;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .drop-sub {
    font-size: 0.8rem;
    color: var(--border);
    font-family: var(--font-head);
  }

  #fileInput { display: none; }

  /* Preview */
  #preview-section { display: none; margin-top: 20px; }

  .preview-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    align-items: start;
  }

  .preview-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .preview-label {
    font-family: var(--font-head);
    font-size: 0.7rem;
    color: var(--muted);
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    letter-spacing: 1px;
  }

  #previewImg {
    width: 100%;
    height: 220px;
    object-fit: cover;
    display: block;
  }

  .btn-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }

  .btn {
    flex: 1;
    padding: 14px;
    border: none;
    border-radius: 3px;
    font-family: var(--font-head);
    font-size: 0.85rem;
    cursor: pointer;
    letter-spacing: 1px;
    transition: all 0.2s;
  }

  .btn-primary {
    background: var(--accent);
    color: #050D12;
    font-weight: 700;
  }

  .btn-primary:hover { background: #03e8b3; transform: translateY(-1px); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  .btn-secondary {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
  }

  .btn-secondary:hover { border-color: var(--muted); color: var(--text); }

  /* Result */
  #result-section { display: none; }

  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 20px;
  }

  .result-header {
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 1px solid var(--border);
  }

  .result-emoji { font-size: 2.8rem; }

  .result-verdict {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -1px;
  }

  .result-class {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 2px;
    font-family: var(--font-head);
  }

  .result-confidence {
    margin-left: auto;
    text-align: right;
  }

  .conf-value {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
  }

  .conf-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .result-tip {
    padding: 16px 24px;
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.6;
    border-bottom: 1px solid var(--border);
  }

  /* Score bars */
  .scores {
    padding: 20px 24px;
  }

  .scores-title {
    font-family: var(--font-head);
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 2px;
    margin-bottom: 16px;
  }

  .score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .score-name {
    font-family: var(--font-head);
    font-size: 0.72rem;
    color: var(--muted);
    width: 130px;
    flex-shrink: 0;
  }

  .score-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
  }

  .score-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--accent);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    width: 0%;
  }

  .score-pct {
    font-family: var(--font-head);
    font-size: 0.75rem;
    color: var(--text);
    width: 42px;
    text-align: right;
  }

  /* Loader */
  .loader {
    display: none;
    text-align: center;
    padding: 32px;
  }

  .loader.active { display: block; }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .loader-text {
    font-family: var(--font-head);
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 2px;
  }

  /* Footer */
  footer {
    text-align: center;
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--muted);
    font-family: var(--font-head);
    letter-spacing: 1px;
  }

  .error-msg {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #FCA5A5;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 0.9rem;
    margin-top: 12px;
    display: none;
  }

  @media (max-width: 600px) {
    .preview-grid { grid-template-columns: 1fr; }
    .result-header { flex-wrap: wrap; }
    .result-confidence { margin-left: 0; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo-row">
      <span class="logo-icon">🐟</span>
      <h1>FISH_INSPECTOR</h1>
    </div>
    <p class="subtitle">AI-Powered Seafood Freshness Detection</p>
    <div class="badge-row">
      <span class="badge">YOLOv8n-cls</span>
      <span class="badge">4 CLASSES</span>
      <span class="badge">~91% ACCURACY</span>
      <span class="badge">REAL-TIME</span>
    </div>
  </header>

  <!-- Upload -->
  <div class="upload-card">
    <div class="drop-zone" id="dropZone">
      <div class="drop-icon">📷</div>
      <p class="drop-text">Drop a fish image here or click to upload</p>
      <p class="drop-sub">JPG · PNG · BMP · max 10MB</p>
      <input type="file" id="fileInput" accept="image/*">
    </div>

    <div id="preview-section">
      <div class="preview-grid">
        <div class="preview-box">
          <div class="preview-label">UPLOADED IMAGE</div>
          <img id="previewImg" src="" alt="Preview">
        </div>
        <div style="display:flex; flex-direction:column; gap:12px; padding-top:4px;">
          <div style="background:var(--surface2); border:1px solid var(--border); border-radius:4px; padding:16px;">
            <div style="font-family:var(--font-head); font-size:0.7rem; color:var(--muted); margin-bottom:8px;">FILE INFO</div>
            <div id="fileInfo" style="font-size:0.85rem; color:var(--text); line-height:1.8;"></div>
          </div>
          <div style="font-family:var(--font-head); font-size:0.75rem; color:var(--muted); line-height:1.8; padding:12px; border:1px solid var(--border); border-radius:4px;">
            MODEL INSPECTS:<br>
            👁 Eye clarity<br>
            🐟 Gill colour<br>
            ✋ Surface texture
          </div>
        </div>
      </div>

      <div class="btn-row">
        <button class="btn btn-primary" id="analyzeBtn" onclick="analyze()">
          🔍 &nbsp;ANALYZE FRESHNESS
        </button>
        <button class="btn btn-secondary" onclick="reset()">RESET</button>
      </div>
      <div class="error-msg" id="errorMsg"></div>
    </div>
  </div>

  <!-- Loader -->
  <div class="loader" id="loader">
    <div class="spinner"></div>
    <p class="loader-text">ANALYZING IMAGE...</p>
  </div>

  <!-- Result -->
  <div id="result-section">
    <div class="result-card">
      <div class="result-header">
        <span class="result-emoji" id="resEmoji"></span>
        <div>
          <div class="result-verdict" id="resVerdict"></div>
          <div class="result-class" id="resClass"></div>
        </div>
        <div class="result-confidence">
          <div class="conf-value" id="resConf"></div>
          <div class="conf-label">confidence</div>
        </div>
      </div>
      <div class="result-tip" id="resTip"></div>
      <div class="scores">
        <div class="scores-title">ALL CLASS SCORES</div>
        <div id="scoreBars"></div>
      </div>
    </div>
    <div style="text-align:center;">
      <button class="btn btn-secondary" style="width:200px;" onclick="reset()">
        ↩ &nbsp;ANALYZE ANOTHER
      </button>
    </div>
  </div>

  <footer>
    FISH_INSPECTOR v1.0 &nbsp;·&nbsp; YOLOv8n-cls &nbsp;·&nbsp; eyes_fresh · eyes_spoiled · gills_fresh · gills_spoiled
  </footer>
</div>

<script>
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const previewSec = document.getElementById('preview-section');
const previewImg = document.getElementById('previewImg');
const fileInfo   = document.getElementById('fileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const loader     = document.getElementById('loader');
const resultSec  = document.getElementById('result-section');
const errorMsg   = document.getElementById('errorMsg');

let selectedFile = null;

// Click to open file dialog
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

// Drag & drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    fileInfo.innerHTML = `
      <b>Name:</b> ${file.name}<br>
      <b>Size:</b> ${(file.size/1024).toFixed(1)} KB<br>
      <b>Type:</b> ${file.type}
    `;
    previewSec.style.display = 'block';
    resultSec.style.display  = 'none';
    errorMsg.style.display   = 'none';
  };
  reader.readAsDataURL(file);
}

async function analyze() {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  loader.classList.add('active');
  resultSec.style.display = 'none';
  errorMsg.style.display  = 'none';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) throw new Error(data.error);

    // populate result
    document.getElementById('resEmoji').textContent   = data.emoji;
    document.getElementById('resVerdict').textContent = data.verdict;
    document.getElementById('resVerdict').style.color = data.color;
    document.getElementById('resClass').textContent   = data.label.replace(/_/g,' ').toUpperCase();
    document.getElementById('resConf').textContent    = data.confidence + '%';
    document.getElementById('resTip').textContent     = '💡 ' + data.tip;

    // score bars
    const bars = document.getElementById('scoreBars');
    bars.innerHTML = '';
    data.scores.forEach(s => {
      bars.innerHTML += `
        <div class="score-row">
          <span class="score-name">${s.name.replace(/_/g,' ')}</span>
          <div class="score-bar-bg">
            <div class="score-bar-fill" id="bar_${s.name}" style="width:0%"></div>
          </div>
          <span class="score-pct">${s.pct}%</span>
        </div>`;
    });

    loader.classList.remove('active');
    resultSec.style.display = 'block';

    // animate bars after render
    setTimeout(() => {
      data.scores.forEach(s => {
        const el = document.getElementById('bar_' + s.name);
        if (el) el.style.width = s.pct + '%';
      });
    }, 100);

  } catch (err) {
    loader.classList.remove('active');
    errorMsg.textContent = '⚠ Error: ' + err.message;
    errorMsg.style.display = 'block';
  }

  analyzeBtn.disabled = false;
}

function reset() {
  selectedFile = null;
  fileInput.value = '';
  previewSec.style.display = 'none';
  resultSec.style.display  = 'none';
  errorMsg.style.display   = 'none';
  previewImg.src = '';
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")

        results    = model.predict(source=img, imgsz=IMG_SIZE, verbose=False)
        probs      = results[0].probs
        class_idx  = int(probs.top1)
        confidence = round(float(probs.top1conf) * 100, 1)
        label      = CLASS_NAMES[class_idx]

        info = FRESHNESS_MAP.get(label, {
            "verdict": "UNKNOWN", "emoji": "❓",
            "color": "#6B7280", "tip": "Could not determine freshness."
        })

        # all scores sorted descending
        scores_arr = probs.data.cpu().numpy()
        sorted_idx = np.argsort(scores_arr)[::-1]
        scores = [
            {"name": CLASS_NAMES[i], "pct": round(float(scores_arr[i]) * 100, 1)}
            for i in sorted_idx
        ]

        return jsonify({
            "label":      label,
            "verdict":    info["verdict"],
            "emoji":      info["emoji"],
            "color":      info["color"],
            "tip":        info["tip"],
            "confidence": confidence,
            "scores":     scores,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🐟 Fish Freshness Inspector — Flask App")
    print("   Open browser at: http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)