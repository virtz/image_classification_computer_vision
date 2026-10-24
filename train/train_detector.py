"""Step 4: train the office-item detector (transfer learning, with a scratch baseline).

Runs three experiments on detect_dataset/:
  1. Stage 1  frozen backbone   (COCO weights, only the new head trains)
  2. Stage 2  full fine-tuning  (everything unfrozen, 10x lower learning rate)
  3. Scratch  random weights    (control: proves whether pretraining helped)
Epochs: 15 + 35 for transfer, 50 for scratch, so the comparison is fair.

Usage:
    python train/train_detector.py             # full run
    SMOKE=1 python train/train_detector.py     # 1 epoch each, to test the pipeline
    DEVICE=cpu python train/train_detector.py  # force a device (0 = first GPU)
"""
import csv
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
DATA = str(PROJECT_ROOT / "preprocessing" / "detect_dataset" / "data.yaml")
PROJECT = str(ROOT / "runs" / "detect")
SMOKE = os.environ.get("SMOKE") == "1"
DEVICE = os.environ.get("DEVICE")          # None lets Ultralytics choose
IMGSZ = 480

E1, E2, ES = (1, 1, 1) if SMOKE else (15, 35, 50)

COMMON = dict(data=DATA, imgsz=IMGSZ, batch=16, optimizer="AdamW",
              seed=0, project=PROJECT, patience=0)   # patience=0: no early stop, epochs stay equal
if DEVICE:
    COMMON["device"] = DEVICE


def best_val_score(run_dir):
    """Highest validation mAP50-95 reached in a run (from its results.csv)."""
    with open(Path(run_dir) / "results.csv") as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[0] if k.strip() == "metrics/mAP50-95(B)")
    return max(float(r[key]) for r in rows)


def main():
    if not Path(DATA).exists():
        raise SystemExit(
            "Detector data.yaml is missing. The existing classification dataset "
            "is data/office_items_dataset/. Run preprocessing/autolabel.py to "
            "generate detector images, bounding-box labels, and data.yaml."
        )

    from ultralytics import YOLO

    tag = "smoke_" if SMOKE else ""

    # Stage 1: frozen backbone. In a YOLOv8 detector, layers 0-9 are the backbone.
    m1 = YOLO("yolov8n.pt")
    m1.train(**COMMON, epochs=E1, freeze=10, lr0=1e-3, pretrained=True,
             name=f"{tag}det8_stage1_frozen")
    run1 = m1.trainer.save_dir

    # Stage 2: start from stage 1 weights, unfreeze everything, low learning rate.
    m2 = YOLO(str(m1.trainer.best))
    m2.train(**COMMON, epochs=E2, freeze=0, lr0=1e-4, name=f"{tag}det8_stage2_fine_tune")
    run2 = m2.trainer.save_dir

    # Baseline: same architecture, random weights.
    ms = YOLO("yolov8n.yaml")
    ms.train(**COMMON, epochs=ES, pretrained=False, lr0=1e-3,
             name=f"{tag}det8_scratch_baseline")
    runs = ms.trainer.save_dir

    # Pick the final transfer model by VALIDATION score (never the test set).
    s1, s2 = best_val_score(run1), best_val_score(run2)
    final_run, final_name = (run2, "stage 2") if s2 >= s1 else (run1, "stage 1")
    print(f"\nVal mAP50-95  stage1={s1:.3f}  stage2={s2:.3f}  scratch={best_val_score(runs):.3f}")
    print(f"Final model: {final_name}")

    if not SMOKE:
        dest = ROOT / "final_model"
        dest.mkdir(exist_ok=True)
        shutil.copy(Path(final_run) / "weights" / "best.pt", dest / "best.pt")
        print(f"Copied to {dest / 'best.pt'}")

        # Test split, touched once, after every choice has been made.
        for label, run in (("transfer", final_run), ("scratch", runs)):
            r = YOLO(str(Path(run) / "weights" / "best.pt")).val(
                data=DATA, split="test", imgsz=IMGSZ, project=PROJECT,
                name=f"test_{label}", **({"device": DEVICE} if DEVICE else {}))
            print(f"{label}: test mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}")


if __name__ == "__main__":
    main()
