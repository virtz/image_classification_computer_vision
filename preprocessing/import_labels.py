"""Import YOLO labels exported from makesense.ai into preprocessing/detect_dataset.

Two cases are handled automatically, by matching each exported file name to an image:

1. The image is already in detect_dataset (you loaded detect_dataset/images/<split>/ to fix
   auto-added boxes): the corrected label replaces the old one.
2. The image only exists in data/office_items_dataset/<split>/<class>/ (you drew a box for an
   image that autolabel skipped): the image is copied into detect_dataset with the same
   naming as autolabel.py (<class_id>_<class>_<name>.<ext>) and its label is written.

Safety checks: every line must be "<class_id> <cx> <cy> <w> <h>" with values in [0, 1], and
the class in the label must match the image's class folder (or file-name prefix). Files that
fail a check are reported and skipped. Use --dry-run first to see what would happen.

Place in preprocessing/ and run from the repo root:
    .venv/bin/python preprocessing/import_labels.py path/to/export_folder_or_zip --dry-run
    .venv/bin/python preprocessing/import_labels.py path/to/export_folder_or_zip
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data" / "office_items_dataset"
OUTPUT = ROOT / "detect_dataset"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IGNORED_NAMES = {"labels", "classes", "class_names"}


def parse_label(text: str, n_classes: int):
    """Return a list of (class_id, line) or None if any line is invalid."""
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            return None
        if len(parts) != 5 or not 0 <= class_id < n_classes or not all(0 <= v <= 1 for v in values):
            return None
        boxes.append((class_id, line.strip()))
    return boxes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "export", help="folder of exported .txt files, or the exported .zip")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only, change nothing")
    args = parser.parse_args()

    export = Path(args.export).expanduser()
    if not export.exists():
        raise SystemExit(f"Not found: {export}")
    tmp = None
    if export.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(export) as archive:
            archive.extractall(tmp.name)
        export = Path(tmp.name)

    classes = sorted(p.name for p in (
        SOURCE / "train").iterdir() if p.is_dir())
    class_ids = {name: index for index, name in enumerate(classes)}

    source_index = {}  # stem -> list of (split, class_name, image_path)
    for split in SPLITS:
        for class_name in classes:
            folder = SOURCE / split / class_name
            if folder.is_dir():
                for image in folder.iterdir():
                    if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                        source_index.setdefault(image.stem, []).append(
                            (split, class_name, image))

    # stem -> (split, image_path) for images already in detect_dataset
    existing_index = {}
    for split in SPLITS:
        folder = OUTPUT / "images" / split
        if folder.is_dir():
            for image in folder.iterdir():
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    existing_index[image.stem] = (split, image)

    done_existing = done_new = 0
    problems = []
    for label_file in sorted(export.rglob("*.txt")):
        stem = label_file.stem
        if stem.lower() in IGNORED_NAMES:
            continue
        boxes = parse_label(label_file.read_text(
            encoding="utf-8"), len(classes))
        if boxes is None:
            problems.append(
                f"{label_file.name}: invalid line (needs 5 numbers, valid class id, values 0 to 1)")
            continue
        if not boxes:
            problems.append(
                f"{label_file.name}: no boxes in this file, skipped")
            continue

        if stem in existing_index:
            split, image = existing_index[stem]
            prefix = stem.split("_", 1)[0]
            expected = int(prefix) if prefix.isdigit() else None
            out_stem, copy_from = stem, None
        elif stem in source_index:
            matches = source_index[stem]
            if len(matches) > 1:
                problems.append(
                    f"{label_file.name}: image name found in more than one folder, skipped")
                continue
            split, class_name, image = matches[0]
            expected = class_ids[class_name]
            out_stem = f"{expected}_{class_name}_{stem}"
            copy_from = image
        else:
            problems.append(f"{label_file.name}: no matching image found")
            continue

        if expected is not None and not any(class_id == expected for class_id, _ in boxes):
            problems.append(
                f"{label_file.name}: class in label does not match the image's class ({classes[expected]}), skipped"
            )
            continue

        if not args.dry_run:
            (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            if copy_from is not None:
                (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
                shutil.copy2(copy_from, OUTPUT / "images" / split /
                             f"{out_stem}{copy_from.suffix.lower()}")
            (OUTPUT / "labels" / split / f"{out_stem}.txt").write_text(
                "\n".join(line for _, line in boxes) + "\n", encoding="utf-8"
            )
        if copy_from is None:
            done_existing += 1
        else:
            done_new += 1

    mode = "Would import" if args.dry_run else "Imported"
    print(f"{mode}: {done_existing} corrected labels for existing images, {done_new} new images with labels.")
    if problems:
        print(f"{len(problems)} files need attention:")
        for message in problems:
            print("  -", message)
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
