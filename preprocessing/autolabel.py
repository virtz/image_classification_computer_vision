"""Use YOLO-World to create starting box labels for the dataset images."""
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data" / "office_items_dataset"
OUTPUT = ROOT / "detect_dataset"
PROMPT_CONFIDENCE = 0.25
PROMPTS = {}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    # Check the source splits before making a new detector dataset.
    if not SOURCE.is_dir():
        raise SystemExit(f"Classification dataset not found: {SOURCE}")
    if not all((SOURCE / split).is_dir() for split in ("train", "val", "test")):
        raise SystemExit(
            f"Expected train/, val/, and test/ folders under {SOURCE}."
        )

    classes = sorted(path.name for path in (SOURCE / "train").iterdir() if path.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found under {SOURCE / 'train'}.")

    for split in ("train", "val", "test"):
        split_classes = {
            path.name for path in (SOURCE / split).iterdir() if path.is_dir()
        }
        if split_classes != set(classes):
            raise SystemExit(
                f"Class folders in {SOURCE / split} do not match train/. "
                f"Missing: {sorted(set(classes) - split_classes)}; "
                f"unexpected: {sorted(split_classes - set(classes))}."
            )
        for name in classes:
            image_files = [
                path for path in (SOURCE / split / name).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise SystemExit(f"No supported images found in {SOURCE / split / name}.")

    model = YOLOWorld("yolov8s-worldv2.pt")
    # Count saved images and keep a list of the ones with no box.
    kept = {name: 0 for name in classes}
    skipped = []

    if OUTPUT.exists():
        backup = OUTPUT.with_name(f"detect_dataset_backup_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.move(str(OUTPUT), str(backup))
        print(f"Previous output moved to {backup}")

    for split in ("train", "val", "test"):
        # Make image and label folders for this split.
        for subfolder in ("images", "labels"):
            (OUTPUT / subfolder / split).mkdir(parents=True, exist_ok=True)
        for class_id, name in enumerate(classes):
            source_class = SOURCE / split / name
            if not source_class.is_dir():
                raise SystemExit(f"Missing class folder for {split}: {source_class}")
            model.set_classes([PROMPTS.get(name, name.replace("_", " "))])
            image_paths = sorted(
                path for path in source_class.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in image_paths:
                # Use the highest-confidence box for this image.
                result = model(str(image_path), conf=PROMPT_CONFIDENCE, verbose=False)[0]
                if len(result.boxes) == 0:
                    skipped.append(str(image_path))
                    continue
                best = int(result.boxes.conf.argmax())
                x, y, width, height = result.boxes.xywhn[best].tolist()
                extension = image_path.suffix.lower()
                output_name = f"{class_id}_{name}_{image_path.stem}{extension}"
                shutil.copy2(image_path, OUTPUT / "images" / split / output_name)
                label_path = OUTPUT / "labels" / split / f"{Path(output_name).stem}.txt"
                label_path.write_text(
                    f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n",
                    encoding="utf-8",
                )
                kept[name] += 1

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    # Tell Ultralytics where to find each split and its class names.
    (OUTPUT / "data.yaml").write_text(
        f"path: {OUTPUT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    (OUTPUT / "skipped.txt").write_text("\n".join(skipped), encoding="utf-8")
    print("Kept pseudo-labeled images per class:", kept)
    print(f"Skipped {len(skipped)} images; review {OUTPUT / 'skipped.txt'}.")
    print(f"Review generated boxes with preprocessing/check_labels.py before training.")


if __name__ == "__main__":
    main()
