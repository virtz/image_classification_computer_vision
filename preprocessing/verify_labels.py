
from pathlib import Path
import yaml

root = Path("preprocessing/detect_dataset")
names = yaml.safe_load((root / "data.yaml").read_text())["names"]
n_classes = len(names)
for split in ("train", "val", "test"):
    imgs = {p.stem for p in (root / "images" / split).glob("*.*")}
    labels = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
    print(split, "images", len(imgs), "labels", len(labels))
    print("  images with no label:", sorted(imgs - labels.keys())[:10])
    print("  labels with no image:", sorted(labels.keys() - imgs)[:10])
    for stem, p in labels.items():
        for line in p.read_text().splitlines():
            parts = line.split()
            ok = len(parts) == 5 and 0 <= int(parts[0]) < n_classes \
                 and all(0 <= float(v) <= 1 for v in parts[1:])
            if not ok:
                print("  bad line:", p.name, line)