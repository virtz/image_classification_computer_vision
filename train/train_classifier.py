"""Train the office-item classifier (YOLOv8n-cls).

Three experiments, 30 epochs of training each, so the comparison is fair:
  1. Stage 1: pretrained backbone frozen, only the new head trains (10 epochs)
  2. Stage 2: fine-tune the whole model from stage 1's best weights (20 epochs)
  3. Scratch baseline: same architecture, random weights (30 epochs)

The better of stage 1 and stage 2 is chosen on VALIDATION accuracy (never test)
and copied to train/final_model/classifier_best.pt.

Place this file in the repo's train/ folder and run from the repo root:
    .venv/bin/python train/train_classifier.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

# Fix 1: resolve the repo root from this file's location (this file lives in train/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = str(REPO_ROOT / "data" / "office_items_dataset")
LABELS_FILE = REPO_ROOT / "train" / "class_names.txt"

# Fix 2: always write runs next to the detector runs, wherever the script is launched from
PROJECT = str(REPO_ROOT / "train" / "runs" / "classify")
FINAL_DIR = REPO_ROOT / "train" / "final_model"


def val_accuracy(results):
    """Top-1 validation accuracy from a finished training run."""
    metrics = results.results_dict
    return metrics.get("metrics/accuracy_top1", metrics.get("fitness", 0.0))


# Stage 1: feature extraction. Pretrained backbone frozen, only the new head trains.
model_stage1 = YOLO("yolov8n-cls.pt")
results_stage1 = model_stage1.train(
    data=DATA,
    project=PROJECT,
    name="office_stage1_frozen",
    epochs=10,
    batch=64,
    freeze=9,
    imgsz=224,
    lr0=1e-3,
    seed=0,
    optimizer="AdamW",
    pretrained=True,
    exist_ok=True,
)

# Stage 2: fine-tune everything, starting from stage 1's best weights.
best_weights_stage1 = Path(results_stage1.save_dir) / "weights" / "best.pt"
model_stage2 = YOLO(str(best_weights_stage1))
results_stage2 = model_stage2.train(  # Fix 3: keep the result so the stages can be compared
    data=DATA,
    project=PROJECT,
    name="office_stage2_fine_tune",
    epochs=20,
    batch=64,
    freeze=0,
    imgsz=224,
    lr0=1e-4,
    seed=0,
    optimizer="AdamW",
    exist_ok=True,
)

# Stage 3: baseline trained from scratch, for the pretrained-vs-scratch comparison.
scratch = YOLO("yolov8n-cls.yaml")
scratch.train(
    data=DATA,
    project=PROJECT,
    name="office_scratch_baseline",
    pretrained=False,
    epochs=30,
    batch=64,
    imgsz=224,
    lr0=1e-3,
    seed=0,
    optimizer="AdamW",
    exist_ok=True,
)

# Fix 3: choose the better transfer stage on validation accuracy and save it.
acc1 = val_accuracy(results_stage1)
acc2 = val_accuracy(results_stage2)
chosen = results_stage2 if acc2 >= acc1 else results_stage1
FINAL_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(Path(chosen.save_dir) / "weights" / "best.pt", FINAL_DIR / "classifier_best.pt")
dataset_classes = sorted(
    path.name for path in (Path(DATA) / "train").iterdir() if path.is_dir()
)
classifier = YOLO(str(FINAL_DIR / "classifier_best.pt"))
class_names = [classifier.names[index] for index in range(len(classifier.names))]
if class_names != dataset_classes:
    raise RuntimeError(
        f"Checkpoint classes {class_names} do not match training folders {dataset_classes}"
    )
LABELS_FILE.write_text(
    "".join(f"  {index}: {name}\n" for index, name in enumerate(class_names)),
    encoding="utf-8",
)
print(
    f"Validation top-1: stage1={acc1:.3f}  stage2={acc2:.3f}. "
    f"Copied {Path(chosen.save_dir).name} to {FINAL_DIR / 'classifier_best.pt'}; "
    f"wrote class labels to {LABELS_FILE}"
)
