"""Step 5: accuracy, macro-F1 and confusion matrix on the held-out test images.

Uses the folder name as the true class and the highest-confidence box as the
prediction. An image with no box counts as the extra class 'none' (wrong).
Output: confusion_matrix.png, misclassified.txt, metrics.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "final_model" / "best.pt"
TEST = ROOT.parent / "data" / "office_items_dataset" / "test"
CONF = 0.25


def main():
    # Refuse to evaluate stale or missing artifacts instead of producing misleading metrics.
    if not WEIGHTS.exists():
        raise SystemExit("final_model/best.pt not found. Run train_detector.py first.")
    if not TEST.is_dir():
        raise SystemExit(f"Test dataset not found: {TEST}")

    model = YOLO(str(WEIGHTS))
    classes = [model.names[i] for i in range(len(model.names))]
    expected_classes = sorted(path.name for path in TEST.iterdir() if path.is_dir())
    if classes != expected_classes:
        raise SystemExit(
            "The final model's classes do not match data/office_items_dataset. "
            "Retrain the detector on the current dataset before evaluation."
        )
    labels = classes + ["none"]

    y_true, y_pred, wrong = [], [], []
    # Treat each image's most confident detection as its prediction; no box maps to "none".
    for cls in classes:
        for f in sorted((TEST / cls).glob("*.jpg")):
            r = model(str(f), conf=CONF, verbose=False)[0]
            pred = int(r.boxes.cls[r.boxes.conf.argmax()]) if len(r.boxes) else len(classes)
            y_true.append(classes.index(cls)); y_pred.append(pred)
            if pred != classes.index(cls):
                wrong.append(f"{f.name} -> {labels[pred]}")

    idx = list(range(len(labels)))
    report = (f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n"
              f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.3f}\n\n"
              + classification_report(y_true, y_pred, labels=idx,
                                      target_names=labels, zero_division=0))
    print(report)
    # Persist the text report and error list alongside the plotted confusion matrix.
    (ROOT / "metrics.txt").write_text(report)
    (ROOT / "misclassified.txt").write_text("\n".join(wrong))

    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=idx),
                           display_labels=labels).plot(xticks_rotation=45)
    plt.tight_layout(); plt.savefig(ROOT / "confusion_matrix.png", dpi=150)
    print(f"{len(wrong)} misclassified, listed in misclassified.txt")


if __name__ == "__main__":
    main()
