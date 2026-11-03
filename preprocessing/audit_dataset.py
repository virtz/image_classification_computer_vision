"""One check for the whole detector dataset (replaces the audit, count, verify and find-missing scripts).

For every split and class it shows:
  folder   photos in data/office_items_dataset
  labelled photos that already have a box in detect_dataset
  missing  photos that still need a box (also written to detect_dataset/missing_boxes.txt)
  boxes    boxes of that class in the label files
It also verifies that images and labels match and that every label line is valid, and
warns about classes that are too thin to train on or to score.

Missing photos are computed fresh from the files, so the numbers stay correct after you
import hand-drawn boxes (skipped.txt and skipped_after_retry.txt go out of date).

Place in preprocessing/ and run from the repo root:
    .venv/bin/python preprocessing/audit_dataset.py
"""
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data" / "office_items_dataset"
DET = ROOT / "detect_dataset"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_TRAIN = 30   # warn if a class has fewer boxes than this in train
MIN_EVAL = 5     # warn if a class has fewer boxes than this in val or test


def load_names():
    # Accept both YAML name encodings: a sequence or an ID-keyed mapping.
    raw = yaml.safe_load((DET / "data.yaml").read_text(encoding="utf-8"))["names"]
    return raw if isinstance(raw, list) else [raw[key] for key in sorted(raw)]


def images_in(folder: Path):
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def main():
    names = load_names()
    # Compare configured class IDs with the source folders before auditing each split.
    classes = sorted(p.name for p in (SOURCE / "train").iterdir() if p.is_dir())
    if names != classes:
        print("WARNING: data.yaml classes differ from the dataset folders")
        print("  data.yaml:", names)
        print("  folders:  ", classes)

    warnings, missing_paths = [], []
    for split in SPLITS:
        # Stems are the shared key used to pair detector images with YOLO label files.
        det_stems = {p.stem for p in images_in(DET / "images" / split)}
        label_files = {p.stem: p for p in (DET / "labels" / split).glob("*.txt")}

        # Boxes per class, and any invalid lines
        boxes = Counter()
        bad_lines = []
        for stem, path in label_files.items():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    # YOLO rows contain a class ID and four normalized box coordinates.
                    ok = (len(parts) == 5 and 0 <= class_id < len(names)
                          and all(0 <= float(v) <= 1 for v in parts[1:]))
                except ValueError:
                    ok = False
                if ok:
                    boxes[names[class_id]] += 1
                else:
                    bad_lines.append(f"{path.name}: {line}")

        # Folder photos vs detector images
        print(f"\n{split.upper()}")
        print(f"  {'class':<14}{'folder':>7}{'labelled':>9}{'missing':>8}{'boxes':>7}")
        for class_id, name in enumerate(classes):
            sources = images_in(SOURCE / split / name)
            labelled = 0
            for image in sources:
                if f"{class_id}_{name}_{image.stem}" in det_stems:
                    labelled += 1
                else:
                    missing_paths.append(str(image))
            missing = len(sources) - labelled
            n_boxes = boxes.get(name, 0)
            print(f"  {name:<14}{len(sources):>7}{labelled:>9}{missing:>8}{n_boxes:>7}")
            limit = MIN_TRAIN if split == "train" else MIN_EVAL
            if n_boxes < limit:
                warnings.append(f"{split}/{name}: only {n_boxes} boxes (want at least {limit})")

        # Images and labels must pair up
        no_label = sorted(det_stems - label_files.keys())
        no_image = sorted(label_files.keys() - det_stems)
        if no_label:
            warnings.append(f"{split}: {len(no_label)} images without a label, e.g. {no_label[:3]}")
        if no_image:
            warnings.append(f"{split}: {len(no_image)} labels without an image, e.g. {no_image[:3]}")
        if bad_lines:
            warnings.append(f"{split}: {len(bad_lines)} invalid label lines, e.g. {bad_lines[:3]}")

    # Rebuild the missing list from current files instead of trusting stale retry logs.
    (DET / "missing_boxes.txt").write_text("\n".join(missing_paths), encoding="utf-8")
    print(f"\n{len(missing_paths)} photos still need a box (listed in {DET / 'missing_boxes.txt'}).")
    if warnings:
        print("\nWARNINGS")
        for message in warnings:
            print("  -", message)
    else:
        print("\nNo warnings: images and labels match and every class has enough boxes.")


if __name__ == "__main__":
    main()
