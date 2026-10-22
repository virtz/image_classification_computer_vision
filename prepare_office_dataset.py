#!/usr/bin/env python3
"""Advanced, non-destructive office-item image dataset curation.

Dependencies:
    python -m pip install numpy pillow opencv-python matplotlib

Run from the repository root (script may be copied there):
    python advanced_office_dataset.py --raw raw_dataset --output code/data/office_items_dataset

Produces 100 files/class (70 train, 15 val, 15 test) when at least
--min-real-for-100 usable distinct REAL photos/class remain after curation.
Only TRAIN is augmented; validation and test ALWAYS contain real photos.
If a class cannot be safely filled to 100, it remains smaller and the
shortfall is explicitly recorded. Raw images are never changed/deleted.

The script cannot guarantee correct labels or genuine new images; review
contact sheets and the audit report before training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, UnidentifiedImageError

CLASSES = (
    "extension_box", "flash_drive", "keyboard", "mouse", "mug",
    "notebook", "pen", "phone", "usd_cord", "water_bottle",
)
ALIASES = {"usb_cord": "usd_cord"}  # Rename only after agreeing with your teammates.
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
REPORT_COLUMNS = [
    "source", "class_name", "status", "reason", "sha256", "phash", "dhash",
    "width", "height", "quality_score", "sharpness", "contrast", "brightness",
    "entropy", "clipped_fraction",
]

@dataclass
class Photo:
    source: Path
    label: str
    sha: str
    phash: int
    dhash: int
    width: int
    height: int
    quality: float
    sharpness: float
    contrast: float
    brightness: float
    entropy: float
    clipped: float


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_image(path: Path) -> Image.Image:
    # Check file integrity, apply camera EXIF rotation, and force RGB.
    with Image.open(path) as f:
        f.verify()
    with Image.open(path) as f:
        rgb = ImageOps.exif_transpose(f).convert("RGB")
        rgb.load()
    return rgb


def label_from_path(file: Path, root: Path) -> str | None:
    parts = (ALIASES.get(s.lower(), s.lower()) for s in file.relative_to(root).parts[:-1])
    possible = set(parts) & set(CLASSES)
    return next(iter(possible)) if len(possible) == 1 else None


def hash_perceptual(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    frequencies = cv2.dct(np.float32(resized))[:8, :8]
    values = frequencies.flatten()[1:]  # Remove DC term
    median = float(np.median(values))
    bits = frequencies.flatten() > median
    bits[0] = False
    result = 0
    for bit in bits:
        result = (result << 1) | bool(bit)
    return result


def hash_difference(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    result = 0
    for bit in (x[:, :-1] > x[:, 1:]).flat:
        result = (result << 1) | bool(bit)
    return result


def inspect(path: Path, label: str, min_side: int) -> tuple[Photo | None, str]:
    try:
        photo = load_image(path)
        w, h = photo.size
        if min(w, h) < min_side:
            return None, f"too_small_{w}x{h}"
        if max(w, h) / min(w, h) > 4:
            return None, "extreme_aspect_ratio"

        # Analyze a fixed-resolution preview so sharpness is comparable.
        photo.thumbnail((512, 512), Image.Resampling.LANCZOS)
        rgb = np.asarray(photo)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
        prob = hist[hist > 0] / hist.sum()
        entropy = float(-(prob * np.log2(prob)).sum())
        clipped = float(np.mean((gray <= 4) | (gray >= 251)))

        # Reject only obvious technical failures; natural smooth product
        # backgrounds can be valid. Never auto-delete ambiguous labels.
        if contrast < 5 and entropy < 3:
            return None, "near_blank"
        if sharpness < 5 and contrast < 13:
            return None, "severely_blurry_or_blank"
        if clipped > 0.97 and entropy < 2:
            return None, "mostly_clipped"

        # Quality is a heuristic for relative ranking, NOT learned aesthetics.
        detail_score = min(1, math.log1p(sharpness) / math.log1p(450))
        contrast_score = min(1, contrast / 66)
        entropy_score = min(1, entropy / 7.4)
        resolution_score = min(1, min(w, h) / 960)
        exposure_score = max(0, 1 - abs(brightness - 126) / 175)
        clipping_score = max(0, 1 - clipped * 2.5)
        quality = 100 * (
            0.26 * detail_score + 0.17 * contrast_score +
            0.12 * entropy_score + 0.24 * resolution_score +
            0.12 * exposure_score + 0.09 * clipping_score
        )
        return Photo(
            path, label, file_sha256(path), hash_perceptual(rgb),
            hash_difference(rgb), w, h, round(quality, 2),
            round(sharpness, 2), round(contrast, 2), round(brightness, 2),
            round(entropy, 2), round(clipped, 4),
        ), "valid"
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, cv2.error) as e:
        return None, f"unreadable_{type(e).__name__}"


def audit_row(photo: Photo, status: str, reason: str) -> dict:
    return {
        "source": str(photo.source), "class_name": photo.label,
        "status": status, "reason": reason, "sha256": photo.sha,
        "phash": f"{photo.phash:016x}", "dhash": f"{photo.dhash:016x}",
        "width": photo.width, "height": photo.height,
        "quality_score": photo.quality, "sharpness": photo.sharpness,
        "contrast": photo.contrast, "brightness": photo.brightness,
        "entropy": photo.entropy, "clipped_fraction": photo.clipped,
    }


def near_duplicate(a: Photo, b: Photo, max_phash: int) -> bool:
    if a.sha == b.sha:
        return True
    hd_p = (a.phash ^ b.phash).bit_count()
    hd_d = (a.dhash ^ b.dhash).bit_count()
    # Two complementary hashes reduce false positives for product images
    # with simple backgrounds; threshold is intentionally conservative.
    return hd_p <= max_phash and hd_d <= 7


def select_photos(candidates: dict[str, list[Photo]], maximum: int,
                  phash_distance: int, audit: list[dict]) -> dict[str, list[Photo]]:
    # If an exactly identical file is labeled as two different classes,
    # exclude all occurrences and flag a potential labeling error.
    sha_labels: dict[str, set[str]] = defaultdict(set)
    for pictures in candidates.values():
        for p in pictures:
            sha_labels[p.sha].add(p.label)
    conflicted = {sha for sha, names in sha_labels.items() if len(names) > 1}

    selected: dict[str, list[Photo]] = {label: [] for label in CLASSES}
    global_kept: list[Photo] = []
    for label in CLASSES:
        # Stable deterministic quality-first selection.
        ranked = sorted(candidates.get(label, []), key=lambda x: (-x.quality, str(x.source)))
        for p in ranked:
            if p.sha in conflicted:
                audit.append(audit_row(p, "rejected", "exact_duplicate_conflicting_class_labels"))
                continue
            if any(p.sha == prev.sha for prev in global_kept):
                audit.append(audit_row(p, "rejected", "exact_duplicate"))
                continue
            if any(near_duplicate(p, prev, phash_distance) for prev in global_kept):
                audit.append(audit_row(p, "rejected", "possible_near_duplicate_review_if_needed"))
                continue
            if len(selected[label]) >= maximum:
                audit.append(audit_row(p, "excluded", "lower_quality_above_100_cap"))
                continue
            selected[label].append(p)
            global_kept.append(p)
    return selected


def smart_enhance(src: Image.Image, mode: str, max_dimension: int) -> Image.Image:
    """Conservative photo improvements: never synthesize object detail."""
    image = src.copy().convert("RGB")
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    if mode == "off":
        return image

    rgb = np.asarray(image).copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    median = float(np.median(lightness))

    # Only correct clearly dark or bright photographs. Avoid dramatic shifts.
    if median < 80:
        gamma = 0.88
    elif median > 190:
        gamma = 1.08
    else:
        gamma = 1.0
    if gamma != 1.0:
        lut = np.array([np.clip((i / 255.0) ** gamma * 255, 0, 255)
                        for i in range(256)], dtype=np.uint8)
        lightness = cv2.LUT(lightness, lut)

    # Mild local contrast in LAB minimizes color distortion. Blend to preserve
    # the photograph's authentic textures, colors and lighting.
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(lightness)
    lab[:, :, 0] = cv2.addWeighted(lightness, 0.72, enhanced_l, 0.28, 0)
    improved = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Gentle sharpening only if the source already has some reliable detail.
    gray = cv2.cvtColor(improved, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if 40 < lap_var < 550:
        blur = cv2.GaussianBlur(improved, (0, 0), sigmaX=0.75)
        improved = cv2.addWeighted(improved, 1.10, blur, -0.10, 0)

    return Image.fromarray(improved)


def augment_train(image: Image.Image, rng: random.Random) -> Image.Image:
    """Train-only augmentations; NEVER for val/test and not independent photos."""
    img = image.convert("RGB")
    angle = rng.uniform(-6, 6)
    # Use same original dimensions. Small geometry and lighting variation.
    fill = tuple(int(v) for v in np.asarray(img.resize((1, 1)))[0, 0])
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC,
                     expand=False, fillcolor=fill)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.88, 1.12))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.90, 1.12))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.92, 1.08))
    # No flips (logos, text, USB connectors). No aggressive crop or upscaling.
    return img


def save_jpeg(img: Image.Image, path: Path, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=jpeg_quality, subsampling=0, optimize=True)


def make_before_after(pairs: list[tuple[Path, Path, float]], path: Path,
                      class_name: str, sample_size: int = 8) -> None:
    if not pairs:
        return
    pairs = pairs[:sample_size]
    tile_w, tile_h, gap = 250, 230, 14
    width = tile_w * 2 + gap * 3
    height = 70 + (tile_h + gap) * len(pairs)
    sheet = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 12), f"{class_name}: original / improved", fill="#152338")
    draw.text((gap, 43), "ORIGINAL", fill="#525c6a")
    draw.text((gap * 2 + tile_w, 43), "PROCESSED", fill="#525c6a")
    for row, (raw, cleaned, score) in enumerate(pairs):
        y = 70 + row * (tile_h + gap)
        for col, file in enumerate((raw, cleaned)):
            x = gap + col * (tile_w + gap)
            with Image.open(file) as f:
                thumb = ImageOps.exif_transpose(f).convert("RGB")
                thumb.thumbnail((tile_w, tile_h - 26), Image.Resampling.LANCZOS)
                px = x + (tile_w - thumb.width) // 2
                py = y + (tile_h - 26 - thumb.height) // 2
                sheet.paste(thumb, (px, py))
            draw.text((x + 5, y + tile_h - 23), f"score {score:.1f} | {raw.name[:19]}", fill="#303747")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "PNG", optimize=True)


def plot_distribution(summary: list[dict], path: Path) -> None:
    # A simple publication-style high-resolution figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s["class_name"] for s in summary]
    train = np.array([s["train_real"] + s["train_augmented"] for s in summary])
    val = np.array([s["val_real"] for s in summary])
    test = np.array([s["test_real"] for s in summary])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    ax.barh(y, train, color="#356bb0", label="Train (includes augmentations)")
    ax.barh(y, val, left=train, color="#e0a043", label="Validation (real)")
    ax.barh(y, test, left=train + val, color="#308a74", label="Test (real)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Images")
    ax.set_title("Office items | class balance after curation", fontweight="bold", pad=15)
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def split_counts(n: int, fill100: bool) -> tuple[int, int, int]:
    if fill100:
        return 70, 15, 15
    if n < 3:
        return max(0, n - 1), min(1, n), 0
    n_val = max(1, round(n * 0.15))
    n_test = max(1, round(n * 0.15))
    while n_val + n_test >= n:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    return n - n_val - n_test, n_val, n_test


def main() -> None:
    default_project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw", type=Path, default=default_project / "raw_dataset")
    parser.add_argument("--output", type=Path,
                        default=default_project / "code/data/office_items_dataset")
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--min-side", type=int, default=160)
    parser.add_argument("--max-dimension", type=int, default=1600)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhance", choices=("natural", "off"), default="natural")
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument("--min-real-for-100", type=int, default=60,
                        help="Do not artificially fill very small classes; collect more real photos")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable train-only filling; leave undersized classes as-is")
    parser.add_argument("--overwrite", action="store_true",
                        help="Back up earlier generated outputs before writing new ones")
    args = parser.parse_args()

    raw = args.raw.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not raw.is_dir():
        parser.error(f"Input folder does not exist: {raw}")
    if raw == out or raw in out.parents or out in raw.parents:
        parser.error("Raw and output directories must be separate and non-nested")
    if not (31 <= args.min_real_for_100 <= args.max_per_class):
        parser.error("--min-real-for-100 must be between 31 and --max-per-class")
    if args.max_per_class != 100:
        parser.error("This coursework script is configured for exactly 100/class (70/15/15)")
    if not (0 <= args.phash_distance <= 10 and 40 <= args.jpeg_quality <= 100):
        parser.error("Invalid pHash distance or JPEG quality")
    if args.min_side < 1 or args.max_dimension < args.min_side:
        parser.error("Invalid image dimension limits")

    rng = random.Random(args.seed)
    generated = [out / s for s in (*SPLITS, "reports", "figures")]
    existing = [p for p in generated if p.exists()]
    if existing and not args.overwrite:
        parser.error("Output already has generated folders. Add --overwrite to back them up first")
    if existing:
        backup = out / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
        backup.mkdir(parents=True, exist_ok=False)
        for old in existing:
            shutil.move(str(old), str(backup / old.name))
        print(f"Previous generated files backed up: {backup}")

    audit: list[dict] = []
    found = Counter()
    candidates: dict[str, list[Photo]] = defaultdict(list)
    files = sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    print(f"Scanning {len(files)} image files in {raw} ...")
    for index, file in enumerate(files, 1):
        label = label_from_path(file, raw)
        if label is None:
            audit.append({"source": str(file), "class_name": "", "status": "skipped",
                          "reason": "class_not_identified_from_folder"})
            continue
        found[label] += 1
        photo, issue = inspect(file, label, args.min_side)
        if photo is None:
            audit.append({"source": str(file), "class_name": label,
                          "status": "rejected", "reason": issue})
        else:
            candidates[label].append(photo)
        if index % 200 == 0:
            print(f"Inspected {index}/{len(files)} images")

    selected = select_photos(candidates, args.max_per_class,
                             args.phash_distance, audit)
    manifest: list[dict] = []
    summary: list[dict] = []
    shortfalls: list[dict] = []

    for label in CLASSES:
        group = selected[label][:]
        rng.shuffle(group)
        n = len(group)
        fill_to_100 = not args.no_augment and args.min_real_for_100 <= n < 100
        train_target, val_n, test_n = split_counts(n, n == 100 or fill_to_100)
        val = group[:val_n]
        test = group[val_n:val_n + test_n]
        train = group[val_n + test_n:]
        # Only a few originals in train with large artificial fill is risky.
        if fill_to_100 and len(train) < 30:
            print(f"CAUTION: {label} has only {len(train)} independent training photographs")

        for split in SPLITS:
            (out / split / label).mkdir(parents=True, exist_ok=True)
        comparisons: list[tuple[Path, Path, float]] = []
        train_paths: list[Path] = []
        for split, subset in (("train", train), ("val", val), ("test", test)):
            for i, photo in enumerate(subset, 1):
                dest = out / split / label / f"{label}_{i:03d}_{photo.sha[:12]}.jpg"
                image = smart_enhance(load_image(photo.source), args.enhance,
                                      args.max_dimension)
                save_jpeg(image, dest, args.jpeg_quality)
                if split == "train":
                    train_paths.append(dest)
                if len(comparisons) < 8:
                    comparisons.append((photo.source, dest, photo.quality))
                manifest.append({
                    "class_name": label, "split": split, "source": str(photo.source),
                    "output": str(dest), "sha256": photo.sha,
                    "quality_score": photo.quality, "augmented": "no",
                    "parent_original_sha256": photo.sha,
                })
                audit.append(audit_row(photo, "accepted", split))

        n_aug = 0
        if fill_to_100:
            n_aug = train_target - len(train_paths)
            for idx in range(n_aug):
                parent = train_paths[idx % len(train_paths)]
                with Image.open(parent) as f:
                    augmented = augment_train(f, rng)
                dest = out / "train" / label / f"{label}_train_AUG_{idx + 1:03d}.jpg"
                save_jpeg(augmented, dest, args.jpeg_quality)
                parent_row = next(m for m in manifest if m["output"] == str(parent))
                manifest.append({
                    "class_name": label, "split": "train",
                    "source": parent_row["source"], "output": str(dest),
                    "sha256": "", "quality_score": "", "augmented": "yes",
                    "parent_original_sha256": parent_row["parent_original_sha256"],
                })

        before = len(candidates.get(label, []))
        count_total = n + n_aug
        need = max(0, 100 - count_total)
        if need:
            shortfalls.append({"class_name": label, "additional_real_images_needed": need,
                               "note": "Acquire diverse, licensed real photos; augmentation not appropriate for this shortage"})
        row = {
            "class_name": label, "files_discovered": found[label],
            "technically_valid": before, "unique_real_selected": n,
            "rejected_or_excluded": found[label] - n,
            "train_real": len(train), "train_augmented": n_aug,
            "val_real": len(val), "test_real": len(test),
            "total_final": count_total, "shortfall": need,
        }
        summary.append(row)
        make_before_after(comparisons, out / "figures" / f"{label}_before_after.png", label)
        print(f"{label:16} valid_unique={n:3d} | train {len(train):3d}+{n_aug:2d} aug | "
              f"val {len(val):2d} | test {len(test):2d} | total={count_total:3d} | missing={need}")

    reports = out / "reports"
    save_csv(reports / "image_audit.csv", audit, REPORT_COLUMNS)
    save_csv(reports / "dataset_manifest.csv", manifest,
             ["class_name", "split", "source", "output", "sha256",
              "quality_score", "augmented", "parent_original_sha256"])
    save_csv(reports / "dataset_summary.csv", summary,
             ["class_name", "files_discovered", "technically_valid", "unique_real_selected",
              "rejected_or_excluded", "train_real", "train_augmented",
              "val_real", "test_real", "total_final", "shortfall"])
    save_csv(reports / "images_to_source.csv", shortfalls,
             ["class_name", "additional_real_images_needed", "note"])
    plot_distribution(summary, out / "figures" / "class_distribution.png")

    # Final checks: file count, readibility, and original photo family leakage.
    for label in CLASSES:
        expected = next(s for s in summary if s["class_name"] == label)
        for split in SPLITS:
            actual = sorted((out / split / label).glob("*.jpg"))
            correct = expected["train_real"] + expected["train_augmented"] if split == "train" else expected[f"{split}_real"]
            if len(actual) != correct:
                raise RuntimeError(f"Wrong {split} count for {label}: {len(actual)} != {correct}")
            for file in actual:
                with Image.open(file) as check:
                    check.verify()
        by_split: dict[str, set[str]] = defaultdict(set)
        for m in manifest:
            if m["class_name"] == label:
                by_split[m["split"]].add(m["parent_original_sha256"])
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            if by_split[first] & by_split[second]:
                raise RuntimeError(f"Original-image leakage: {label} {first}/{second}")

    print(f"\nFinished. Final dataset: {out}")
    print(f"Statistics / audit: {reports}")
    print(f"Professional QA figures: {out / 'figures'}")
    if shortfalls:
        print("Some classes are below 100. See reports/images_to_source.csv.")
    if any(s["train_augmented"] for s in summary):
        print("IMPORTANT: augmented samples are NOT independent real images.")
    print("Review contact sheets and labels manually before training.")


if __name__ == "__main__":
    main()
