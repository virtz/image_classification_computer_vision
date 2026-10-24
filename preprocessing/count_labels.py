
from pathlib import Path
from collections import Counter
import yaml
root = Path("preprocessing/detect_dataset")
names = yaml.safe_load((root / "data.yaml").read_text())["names"]
names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
for split in ("train", "val", "test"):
    c = Counter()
    for f in (root / "labels" / split).glob("*.txt"):
        for line in f.read_text().splitlines():
            c[names[int(line.split()[0])]] += 1
    print(split, {n: c.get(n, 0) for n in names})
