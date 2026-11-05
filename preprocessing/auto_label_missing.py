"""Add box labels for images that autolabel.py didn't label.

Unlike autolabel.py, this script leaves the existing detector dataset alone.
It uses the same class IDs and filenames, with a lower confidence threshold,
so check the new boxes by hand.

Run from the repository root:
    .venv/bin/python preprocessing/autolabel_missing.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data" / "office_items_dataset"
OUTPUT = ROOT / "detect_dataset"
PROMPT_CONFIDENCE = 0.05          # autolabel.py used 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_CLASSES = set()
PROMPTS = {                       # clearer text than the folder names
    "flash_drive": "USB flash drive",
    "usb_cord": "USB cable",
    "extension_box": "power strip",
}


def main():
    # The first autolabel run needs to have created data.yaml.
    if not (OUTPUT / "data.yaml").is_file():
        raise SystemExit(f"{OUTPUT / 'data.yaml'} not found. Run autolabel.py first.")

    # Keep class IDs in the same sorted order as autolabel.py.
    classes = sorted(p.name for p in (SOURCE / "train").iterdir() if p.is_dir())
    model = YOLOWorld("yolov8s-worldv2.pt")
    added = {name: 0 for name in classes}
    still_missing = []

    for split in ("train", "val", "test"):
        # Make missing folders, but leave existing labels alone.
        for subfolder in ("images", "labels"):
            (OUTPUT / subfolder / split).mkdir(parents=True, exist_ok=True)
        for class_id, name in enumerate(classes):
            if name in SKIP_CLASSES:
                continue
            model.set_classes([PROMPTS.get(name, name.replace("_", " "))])
            for image_path in sorted((SOURCE / split / name).iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                # Use the same filename pattern so we can spot images already added.
                output_name = f"{class_id}_{name}_{image_path.stem}{image_path.suffix.lower()}"
                if (OUTPUT / "images" / split / output_name).exists():
                    continue
                result = model(str(image_path), conf=PROMPT_CONFIDENCE, verbose=False)[0]
                if len(result.boxes) == 0:
                    still_missing.append(str(image_path))
                    continue
                best = int(result.boxes.conf.argmax())
                # Save the highest-confidence box in YOLO's normalized format.
                x, y, width, height = result.boxes.xywhn[best].tolist()
                shutil.copy2(image_path, OUTPUT / "images" / split / output_name)
                label_path = OUTPUT / "labels" / split / f"{Path(output_name).stem}.txt"
                label_path.write_text(
                    f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n",
                    encoding="utf-8",
                )
                added[name] += 1

    (OUTPUT / "skipped_after_retry.txt").write_text("\n".join(still_missing), encoding="utf-8")
    print("Added pseudo-labeled images per class:", added)
    print(f"{len(still_missing)} images still have no box; see skipped_after_retry.txt.")
    print("Review the new boxes by hand before training, especially val and test.")


if __name__ == "__main__":
    main()