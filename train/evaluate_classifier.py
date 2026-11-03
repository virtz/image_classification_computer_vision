"""Evaluate the office-item classifier on the held-out test folder.

Scores the chosen transfer model and the scratch baseline, and writes, per model:
  train/<tag>_metrics.txt            accuracy, macro-F1, per-class precision/recall/F1
  train/<tag>_misclassified.txt      every wrong prediction, for the error analysis
  train/<tag>_confusion_matrix.png   confusion matrix

Run ONCE, after training is final. Do not retrain or tune based on these numbers.
Place in the train/ folder and run from the repo root:
    .venv/bin/python train/evaluate_classifier.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "data" / "office_items_dataset" / "test"
OUT_DIR = REPO_ROOT / "train"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MODELS = {
    "classifier": REPO_ROOT / "train" / "final_model" / "classifier_best.pt",
    "classifier_scratch": REPO_ROOT / "train" / "runs" / "classify"
    / "office_scratch_baseline" / "weights" / "best.pt",
}


def evaluate(weights: Path, tag: str, classes: list) -> None:
    # A missing optional baseline is reported and skipped so the available model can be scored.
    if not weights.is_file():
        print(f"[{tag}] weights not found: {weights}. Skipping.")
        return
    model = YOLO(str(weights))
    if sorted(model.names.values()) != classes:
        raise SystemExit(
            f"[{tag}] class names differ from the test folders.\n"
            f"  model:   {sorted(model.names.values())}\n"
            f"  folders: {classes}\nRetrain after renaming or changing classes."
        )

    y_true, y_pred, wrong = [], [], []
    # Evaluate only supported image formats and retain wrong cases for manual error analysis.
    for cls in classes:
        for image in sorted((TEST_DIR / cls).iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            result = model(str(image), verbose=False)[0]
            predicted = result.names[result.probs.top1]
            y_true.append(cls)
            y_pred.append(predicted)
            if predicted != cls:
                wrong.append(
                    f"{image.relative_to(REPO_ROOT)}  true={cls}  pred={predicted}  "
                    f"conf={float(result.probs.top1conf):.2f}"
                )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
    report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    (OUT_DIR / f"{tag}_metrics.txt").write_text(
        f"test images: {len(y_true)}\naccuracy: {accuracy:.3f}\nmacro-F1: {macro_f1:.3f}\n\n{report}",
        encoding="utf-8",
    )
    (OUT_DIR / f"{tag}_misclassified.txt").write_text("\n".join(wrong), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 8))
    # Use the same class ordering for the confusion matrix and its axis labels.
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=classes), display_labels=classes
    ).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"[{tag}] {len(y_true)} test images  accuracy {accuracy:.3f}  macro-F1 {macro_f1:.3f}  "
          f"wrong {len(wrong)}")


def main() -> None:
    if not TEST_DIR.is_dir():
        raise SystemExit(f"Test folder not found: {TEST_DIR}")
    classes = sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())
    # Evaluate transfer and scratch checkpoints against the same held-out folders.
    for tag, weights in MODELS.items():
        evaluate(weights, tag, classes)


if __name__ == "__main__":
    main()