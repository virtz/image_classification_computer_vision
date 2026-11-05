"""Evaluate the classifier using the held-out test images.

It scores the selected transfer model and scratch baseline, then saves a
metrics report, a list of wrong predictions, and a confusion matrix for each.

Run after training is finished. These test results shouldn't be used to tune
the models. Run from the repository root:
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
  
    if not weights.is_file():   # The scratch checkpoint is optional, so keep going if it's missing.
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
    # Score each test image and keep a note of the mistakes.
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
    # Keep the matrix labels in the same order as the class list.
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
    # Score both checkpoints on the same test folders.
    for tag, weights in MODELS.items():
        evaluate(weights, tag, classes)


if __name__ == "__main__":
    main()