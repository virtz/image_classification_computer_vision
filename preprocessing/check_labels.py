"""Render a sample of generated boxes for manual pseudo-label review."""
import random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent / "detect_dataset"
PREVIEW = ROOT / "preview"
SAMPLE = 200
SPLIT = "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    yaml_path = ROOT / "data.yaml"
    if not yaml_path.is_file():
        raise SystemExit(
            "preprocessing/detect_dataset/data.yaml not found. Run autolabel.py first.")
    PREVIEW.mkdir(exist_ok=True)
    names = [
        line.split(": ", 1)[1]
        for line in yaml_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("  ") and ": " in line
    ]

    images = sorted(
        path for path in (ROOT / "images" / SPLIT).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.seed(1)
    multi_box_images = []
    single_box_images = []
    for image_path in images:
        label_path = ROOT / "labels" / SPLIT / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        row_count = sum(
            bool(line.strip())
            for line in label_path.read_text(encoding="utf-8").splitlines()
        )
        (multi_box_images if row_count > 1 else single_box_images).append(image_path)

    selected = random.sample(
        multi_box_images, min(SAMPLE, len(multi_box_images)))
    remaining = SAMPLE - len(selected)
    selected.extend(random.sample(single_box_images,
                    min(remaining, len(single_box_images))))
    selected.sort()

    for old_preview in PREVIEW.iterdir():
        if old_preview.is_file() and old_preview.suffix.lower() in IMAGE_EXTENSIONS:
            old_preview.unlink()

    for image_path in selected:
        label_path = ROOT / "labels" / SPLIT / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or not label_path.is_file():
            continue
        height, width = image.shape[:2]
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: expected 5 YOLO fields, "
                    f"found {len(fields)}"
                )
            try:
                class_id = int(fields[0])
                x, y, box_width, box_height = map(float, fields[1:])
            except ValueError as error:
                raise ValueError(
                    f"{label_path}:{line_number}: invalid YOLO label row"
                ) from error
            if not 0 <= class_id < len(names):
                raise ValueError(
                    f"{label_path}:{line_number}: class ID {class_id} is outside "
                    f"the {len(names)} configured classes"
                )
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
                raise ValueError(
                    f"{label_path}:{line_number}: box coordinates must be normalized "
                    "to [0, 1] with positive width and height"
                )

            center_x, center_y = x * width, y * height
            pixel_box_width, pixel_box_height = box_width * width, box_height * height
            point1 = (
                int(center_x - pixel_box_width / 2),
                int(center_y - pixel_box_height / 2),
            )
            point2 = (
                int(center_x + pixel_box_width / 2),
                int(center_y + pixel_box_height / 2),
            )
            color = (44, 160, 44)
            cv2.rectangle(image, point1, point2, color, 3)
            cv2.putText(
                image, names[class_id], (point1[0], max(20, point1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
        if not cv2.imwrite(str(PREVIEW / image_path.name), image):
            raise RuntimeError(f"Could not write preview for {image_path}")
    print(f"Wrote previews to {PREVIEW}")


if __name__ == "__main__":
    main()
