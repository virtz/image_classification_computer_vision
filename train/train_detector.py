"""Train the office-item object detector.

The script runs three experiments on detect_dataset/: a frozen-backbone stage
starting from COCO weights, full fine-tuning from that stage, and a scratch
baseline with random weights. They run for 15, 35, and 50 epochs respectively.

Run from the repository root:
    python train/train_detector.py

To force CPU use:
    DEVICE=cpu python train/train_detector.py
"""
import csv
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
DATA = str(PROJECT_ROOT / "preprocessing" / "detect_dataset" / "data.yaml")
PROJECT = str(ROOT / "runs" / "detect")
DEVICE = os.environ.get("DEVICE")          # None lets Ultralytics choose
IMGSZ = 480

COMMON = dict(data=DATA, imgsz=IMGSZ, batch=16, optimizer="AdamW",
              seed=0, project=PROJECT, patience=0)   # patience=0: no early stop, epochs stay equal
if DEVICE:
    COMMON["device"] = DEVICE


def best_val_score(run_dir):
    """Highest validation mAP50-95 reached in a run (from its results.csv)."""
    # Read the validation scores saved for this run.
    with open(Path(run_dir) / "results.csv") as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[0] if k.strip() == "metrics/mAP50-95(B)")
    return max(float(r[key]) for r in rows)


def main():
    # Make sure the detector dataset has been created before training.
    if not Path(DATA).exists():
        raise SystemExit(
            "Detector data.yaml is missing. The existing classification dataset "
            "is data/office_items_dataset/. Run preprocessing/autolabel.py to "
            "generate detector images, bounding-box labels, and data.yaml."
        )

    from ultralytics import YOLO

    # Stage 1: freeze the early layers and train from pretrained weights.
    m1 = YOLO("yolov8n.pt")
    m1.train(**COMMON, epochs=15, freeze=10, lr0=1e-3, pretrained=True,
             name="det8_stage1_frozen")
    run1 = m1.trainer.save_dir

    # Stage 2: continue from stage 1, now training all layers at a lower rate.
    m2 = YOLO(str(m1.trainer.best))
    m2.train(**COMMON, epochs=35, freeze=0, lr0=1e-4, name="det8_stage2_fine_tune")
    run2 = m2.trainer.save_dir

    # Baseline: same architecture, random weights.
    ms = YOLO("yolov8n.yaml")
    ms.train(**COMMON, epochs=50, pretrained=False, lr0=1e-3,
             name="det8_scratch_baseline")
    runs = ms.trainer.save_dir

    # Choose between the two transfer runs using validation scores.
    s1, s2 = best_val_score(run1), best_val_score(run2)
    # Leave the test set out of this choice.
    final_run, final_name = (run2, "stage 2") if s2 >= s1 else (run1, "stage 1")
    print(f"\nVal mAP50-95  stage1={s1:.3f}  stage2={s2:.3f}  scratch={best_val_score(runs):.3f}")
    print(f"Final model: {final_name}")

    dest = ROOT / "final_model"
    dest.mkdir(exist_ok=True)
    shutil.copy(Path(final_run) / "weights" / "best.pt", dest / "best.pt")
    print(f"Copied to {dest / 'best.pt'}")

    # Run the final test evaluation after choosing the checkpoint.
    for label, run in (("transfer", final_run), ("scratch", runs)):
        r = YOLO(str(Path(run) / "weights" / "best.pt")).val(
            data=DATA, split="test", imgsz=IMGSZ, project=PROJECT,
            name=f"test_{label}", **({"device": DEVICE} if DEVICE else {}))
        print(f"{label}: test mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}")


if __name__ == "__main__":
    main()
