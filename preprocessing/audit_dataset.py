"""Check the detector dataset and report any missing or invalid labels.

For each class and split, the report shows the source photos, labelled photos,
missing labels, and box counts. It also checks that image and label files pair
up and that each label line is valid.

Missing photos are found from the current files, not from old skip lists. Run
from the repository root:
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
    # data.yaml can store names as a list or as an ID-keyed mapping.
    raw = yaml.safe_load((DET / "data.yaml").read_text(encoding="utf-8"))["names"]
    return raw if isinstance(raw, list) else [raw[key] for key in sorted(raw)]


def images_in(folder: Path):
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def main():
    names = load_names()
    # Check that the YAML class order matches the source folders.
    classes = sorted(p.name for p in (SOURCE / "train").iterdir() if p.is_dir())
    if names != classes:
        print("WARNING: data.yaml classes differ from the dataset folders")
        print("  data.yaml:", names)
        print("  folders:  ", classes)

    warnings, missing_paths = [], []
    for split in SPLITS:
        # Match image and label files by their filename stem.
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
                    # A YOLO row has a class ID and four coordinates between 0 and 1.
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

    # Recreate this list from the files so it isn't out of date.
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
