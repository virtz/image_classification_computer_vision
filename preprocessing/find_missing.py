from pathlib import Path
from collections import Counter
lines = Path("preprocessing/detect_dataset/skipped_after_retry.txt").read_text().splitlines()
print(Counter((Path(l).parent.parent.name, Path(l).parent.name) for l in lines))