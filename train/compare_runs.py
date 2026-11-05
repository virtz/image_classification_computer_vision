"""Compare the detector runs using their validation mAP50 scores.

The plot joins transfer stages 1 and 2 into one curve and compares it with
the scratch run. It saves training_comparison.png and comparison_table.txt.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RUNS = ROOT / "runs" / "detect"
KEY = "metrics/mAP50(B)"


def curve(name):
    d = RUNS / name
    # Ultralytics adds a number if a run name is already taken.
    if not d.exists():          # Ultralytics may have added a number (name2, name3...)
        cands = sorted(RUNS.glob(f"{name}*"))
        if not cands:
            raise SystemExit(f"No run folder for {name}")
        d = cands[-1]
    with open(d / "results.csv") as f:
        rows = list(csv.DictReader(f))
    k = next(c for c in rows[0] if c.strip() == KEY)
    return [float(r[k]) for r in rows]


def main():
    # Put the two transfer stages together to compare with the scratch run.
    s1, s2, sc = (curve(n) for n in
                  ("det8_stage1_frozen", "det8_stage2_fine_tune", "det8_scratch_baseline"))
    transfer = s1 + s2
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(transfer) + 1), transfer, label="transfer (frozen then fine-tuned)")
    plt.plot(range(1, len(sc) + 1), sc, label="scratch")
    plt.axvline(len(s1) + 0.5, ls="--", c="grey", lw=1)
    # Show where fine-tuning starts on the transfer curve.
    plt.text(len(s1) + 1, 0.02, "unfrozen", fontsize=8, color="grey")
    plt.xlabel("epoch"); plt.ylabel("validation mAP50"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(ROOT / "training_comparison.png", dpi=150)

    lines = ["run            best val mAP50   mAP50 at epoch 5",
             f"stage 1        {max(s1):.3f}            {s1[min(4, len(s1)-1)]:.3f}",
             f"stage 2        {max(s2):.3f}            -",
             f"scratch        {max(sc):.3f}            {sc[min(4, len(sc)-1)]:.3f}"]
    (ROOT / "comparison_table.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
