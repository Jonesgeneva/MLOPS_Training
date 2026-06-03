

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_DIR  = r"C:\Fish_grading_model\Datasets"        # your 4 class folders
YOLO_DIR     = r"C:\Fish_grading_model\yolo_dataset"   # auto-created
MODEL_OUTPUT = r"C:\Fish_grading_model\yolo_results"   # saved model + plots

CLASS_NAMES  = ["eyes_fresh", "eyes_spoiled", "gills_fresh", "gills_spoiled"]
IMG_SIZE     = 224
EPOCHS       = 50
BATCH_SIZE   = 16
SPLIT_RATIO  = 0.8    # 80% train, 20% val
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────


# ── Step 1: Build YOLO folder structure ──────────────────────────────────────
# YOLO classification expects:
#   yolo_dataset/
#     train/
#       eyes_fresh/
#       eyes_spoiled/
#       gills_fresh/
#       gills_spoiled/
#     val/
#       eyes_fresh/
#       ...

def build_yolo_dataset():
    print("\nStep 1: Building YOLO dataset structure...")

    yolo_path = Path(YOLO_DIR)

    # clean old yolo_dataset if exists
    if yolo_path.exists():
        shutil.rmtree(yolo_path)

    total_train = 0
    total_val   = 0

    for cls in CLASS_NAMES:
        src_dir = Path(DATASET_DIR) / cls
        if not src_dir.exists():
            print(f"  [skip] Folder not found: {src_dir}")
            continue

        images = [f for f in src_dir.iterdir()
                  if f.suffix.lower() in IMG_EXTS]

        if not images:
            print(f"  [skip] No images in: {src_dir}")
            continue

        # shuffle and split
        random.seed(42)
        random.shuffle(images)
        split     = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split]
        val_imgs   = images[split:]

        # create destination folders
        train_dst = yolo_path / "train" / cls
        val_dst   = yolo_path / "val"   / cls
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        # copy images
        for f in train_imgs:
            shutil.copy(str(f), str(train_dst / f.name))
        for f in val_imgs:
            shutil.copy(str(f), str(val_dst / f.name))

        total_train += len(train_imgs)
        total_val   += len(val_imgs)
        print(f"  {cls:<20}: {len(train_imgs)} train | {len(val_imgs)} val")

    print(f"\n  Total train : {total_train}")
    print(f"  Total val   : {total_val}")
    print(f"  Saved to    : {YOLO_DIR}")
    return yolo_path


# ── Step 2: Train YOLO ───────────────────────────────────────────────────────
def train_yolo(yolo_path):
    print("\nStep 2: Loading YOLOv8 classification model...")

    # yolov8n-cls = nano classification — fastest, good for CPU
    model = YOLO("yolov8n-cls.pt")

    print(f"  Model       : YOLOv8n-cls (nano — fast on CPU)")
    print(f"  Image size  : {IMG_SIZE}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"\nStep 3: Training...\n")

    results = model.train(
        data=str(yolo_path),          # path to yolo_dataset/
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=MODEL_OUTPUT,         # save folder
        name="fish_freshness",        # run name
        exist_ok=True,
        verbose=True,
        patience=10,                  # early stopping
        device="cpu",                 # use CPU
        workers=0,                    # Windows fix — avoid multiprocessing errors
    )

    return results


# ── Step 3: Validate and print results ───────────────────────────────────────
def evaluate(yolo_path):
    print("\nStep 4: Evaluating best model...")

    best_model_path = Path(MODEL_OUTPUT) / "fish_freshness" / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"  [!] best.pt not found at {best_model_path}")
        return

    model = YOLO(str(best_model_path))
    metrics = model.val(
        data=str(yolo_path),
        imgsz=IMG_SIZE,
        device="cpu",
        workers=0,
    )

    print(f"""
  ── Validation Results ──────────────────────────────
  Top-1 Accuracy : {metrics.top1:.4f}  ({metrics.top1*100:.1f}%)
  Top-5 Accuracy : {metrics.top5:.4f}  ({metrics.top5*100:.1f}%)
  ────────────────────────────────────────────────────
    """)

    print(f"""
═══════════════════════════════════════════════════
  YOLO Training Complete!

  Best model → {best_model_path}
  All results → {MODEL_OUTPUT}\\fish_freshness\\

  Inside results folder you will find:
    weights\\best.pt      ← use this for demo
    weights\\last.pt      ← last epoch model
    results.png          ← training curves
    confusion_matrix.png ← per class accuracy
═══════════════════════════════════════════════════

Next: Run gradio_demo_yolo.py for live demo
""")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Seafood Freshness — YOLO Classification ──────────────")
    print(f"   Dataset  : {DATASET_DIR}")
    print(f"   Classes  : {CLASS_NAMES}")
    print(f"   Epochs   : {EPOCHS}")
    print(f"   Device   : CPU")
    print(f"─────────────────────────────────────────────────────────")

    # check dataset exists
    if not Path(DATASET_DIR).exists():
        print(f"\n[ERROR] Dataset folder not found: {DATASET_DIR}")
        print("  Make sure your 4 class folders are inside C:\\Fish_grading_model\\dataset\\")
        exit()

    yolo_path = build_yolo_dataset()
    train_yolo(yolo_path)
    evaluate(yolo_path)