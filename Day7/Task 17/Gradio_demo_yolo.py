"""
Seafood Freshness — YOLO Gradio Live Demo
==========================================
Usage:
    pip install ultralytics gradio Pillow
    python gradio_demo_yolo.py
"""

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Fish_grading_model\yolo_results\fish_freshness\weights\best.pt"
IMG_SIZE   = 224

FRESHNESS_MAP = {
    "eyes_fresh":    ("✅ FRESH",   "👁️ Eyes are clear and bright — fish is fresh. Safe to consume."),
    "eyes_spoiled":  ("❌ SPOILED", "👁️ Eyes look cloudy or sunken — sign of spoilage. Do not consume."),
    "gills_fresh":   ("✅ FRESH",   "🐟 Gills are red/pink and healthy — fish is fresh. Safe to consume."),
    "gills_spoiled": ("❌ SPOILED", "🐟 Gills appear grey/brown — sign of spoilage. Do not consume."),
}
# ─────────────────────────────────────────────────────────────────────────────

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names   # {0: 'eyes_fresh', 1: 'eyes_spoiled', ...}
print(f"Model loaded ✅  Classes: {list(CLASS_NAMES.values())}")


def classify_fish(image):
    if image is None:
        return "Upload an image to classify", "", ""

    # run YOLO inference
    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        verbose=False
    )

    probs      = results[0].probs
    class_idx  = int(probs.top1)
    confidence = float(probs.top1conf) * 100
    label      = CLASS_NAMES[class_idx]

    verdict, tip = FRESHNESS_MAP.get(label, ("❓ UNKNOWN", "Could not determine freshness."))

    # all scores sorted
    all_scores = "\n".join([
        f"  {CLASS_NAMES[i]:<16}: {float(probs.data[i])*100:.1f}%"
        for i in np.argsort(probs.data.cpu().numpy())[::-1]
    ])

    result = f"{verdict}  —  {label.replace('_', ' ').title()}\nConfidence: {confidence:.1f}%"
    scores = f"All class scores:\n{all_scores}"
    note   = tip if confidence >= 70 else tip + "\n\n⚠️ Low confidence — try a clearer image."

    return result, scores, note


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🐟 Fish Freshness Classifier — YOLO") as demo:

    gr.Markdown("""
    # 🐟 Fish Freshness Classifier — YOLO
    ### Upload a fish eye or gill image → get instant freshness grade
    *YOLOv8n-cls · 4 classes · Trained on real fish data*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input  = gr.Image(type="pil", label="📷 Upload Fish Image")
            classify_btn = gr.Button("🔍 Classify Freshness", variant="primary")

        with gr.Column(scale=1):
            result_out = gr.Textbox(label="🏷️ Freshness Verdict",  lines=2)
            scores_out = gr.Textbox(label="📊 Confidence Scores",  lines=6)
            tip_out    = gr.Textbox(label="💡 Inspector's Note",   lines=3)

    classify_btn.click(
        fn=classify_fish,
        inputs=image_input,
        outputs=[result_out, scores_out, tip_out]
    )
    image_input.change(
        fn=classify_fish,
        inputs=image_input,
        outputs=[result_out, scores_out, tip_out]
    )

    gr.Markdown("**Model:** YOLOv8n-cls · **Classes:** `eyes_fresh` · `eyes_spoiled` · `gills_fresh` · `gills_spoiled`")

print("\nStarting Gradio demo...")
print("Open browser at: http://127.0.0.1:7860\n")
demo.launch(share=True, theme=gr.themes.Soft())