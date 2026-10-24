from pathlib import Path
src = Path("data/office_items_dataset")
det = Path("preprocessing/detect_dataset/images")
skipped = Path("preprocessing/detect_dataset/skipped.txt")
print("skipped.txt lines:", len(skipped.read_text().splitlines()) if skipped.exists() else "none")
for split in ("train", "val", "test"):
    kept = {p.name for p in (det / split).glob("*.*")}
    for cls in sorted(p.name for p in (src / split).iterdir() if p.is_dir()):
        n = len([p for p in (src / split / cls).glob("*.*")])
        print(split, cls, "folder:", n)
    print(split, "images in detect_dataset:", len(kept))