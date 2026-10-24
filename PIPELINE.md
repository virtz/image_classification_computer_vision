# Office-item dataset and object-detection pipeline

This document describes how the repository's curated image dataset becomes a
YOLO object-detection dataset, how the detector is trained, and what the
resulting metrics do and do not establish.

## Pipeline at a glance

```text
Optional new raw photos
  preprocessing/raw_dataset/<class_name>/*
                 |
                 | prepare_office_dataset.py (optional curation)
                 v
  data/office_items_dataset/{train,val,test}/<class_name>/*
                 |
                 | preprocessing/autolabel.py (YOLO-World pseudo-boxes)
                 v
  preprocessing/detect_dataset/
    images/{train,val,test}/*
    labels/{train,val,test}/*.txt
    data.yaml
    skipped.txt
                 |
                 | preprocessing/check_labels.py (visual QA)
                 | manually correct/remove bad image-label pairs
                 v
  train/train_detector.py
    frozen transfer -> fine-tuning -> scratch baseline
                 |
                 +--> train/runs/detect/
                 +--> train/final_model/best.pt
                 +--> held-out detector metrics
                 |
                 | train/evaluate.py (class-level test predictions)
                 v
  train/metrics.txt, train/misclassified.txt,
  train/confusion_matrix.png
```

The existing `data/office_items_dataset/` already contains the curated
train/validation/test image folders. For that dataset, skip the optional raw
photo curation step and start with `autolabel.py`.

## 1. Source dataset layout

The source is a classification dataset: each image's class is encoded by the
folder it is in.

```text
data/office_items_dataset/
  train/<class_name>/*.{jpg,jpeg,png,webp,bmp}
  val/<class_name>/*.{jpg,jpeg,png,webp,bmp}
  test/<class_name>/*.{jpg,jpeg,png,webp,bmp}
```

The curated dataset contains eight classes:
`extension_box`, `flash_drive`, `mug`, `notebook`, `pen`, `phone`,
`usb_cord`, and `water_bottle`.

This layout supplies image-level class labels only. A detector additionally
needs a bounding box around each object; those boxes are not part of the
classification folders.

The classifier training script writes the ordered class mapping to the
repository-root `labels.txt` after it saves the selected checkpoint. Class
IDs follow the sorted training-folder names; keep this mapping aligned with
the checkpoint when displaying predictions.

## 2. Optional: curate a new raw-image collection

Use [prepare_office_dataset.py](prepare_office_dataset.py) only when preparing
new raw photos or replacing the existing curated dataset. Put photos in
class-named folders:

```text
preprocessing/raw_dataset/
  mug/photo1.jpg
  pen/photo2.jpg
  ...
```

From the repository root, run:

```bash
.venv/bin/python prepare_office_dataset.py
```

Its defaults read from `preprocessing/raw_dataset/` and write to
`data/office_items_dataset/`. The curation script validates and scores images,
rejects exact/near duplicates and technically poor files, caps selection at
100 distinct photos per class, and makes class-folder train/val/test splits.
Where enough real images are available, the target is 70 train, 15 validation,
and 15 test images per class. With 60-99 selected real photos, it can augment
the training portion only; validation and test remain real photos. Small
classes may remain below 100, with shortfalls reported.

It also writes audit/manifest/summary CSVs and comparison/distribution
figures under `data/office_items_dataset/reports/` and `figures/`. Existing
generated split/report/figure folders are preserved by default: use
`--overwrite` to move them into a timestamped backup before replacing them.
Raw source images are not modified.

## 3. Generate detector labels from the existing splits

Install the project dependencies if needed:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Then run:

```bash
.venv/bin/python preprocessing/autolabel.py
```

The script reads `data/office_items_dataset/` directly. It checks that all
three splits exist, have the same class folders, and contain supported images.
It sorts class folder names to establish class IDs, then uses
`yolov8s-worldv2.pt` (downloaded on first use) to prompt YOLO-World with the
class name for each image. It keeps the highest-confidence predicted box at
confidence 0.25. Images with no box are omitted and listed in `skipped.txt`.

Generated files go to:

```text
preprocessing/detect_dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
  skipped.txt
```

Each image has a matching text label with one YOLO normalized bounding box:

```text
<class_id> <center_x> <center_y> <width> <height>
```

The YAML lists the absolute dataset root, relative image paths for each split,
and class-ID/name mapping. The current mapping is also recorded in the
generated `data.yaml`. The detector trainer reads that YAML from
`preprocessing/detect_dataset/data.yaml`.

If `autolabel.py` is rerun, the prior generated detector dataset is moved to a
timestamped sibling backup before a fresh one is generated.

## 4. Inspect and correct pseudo-labels

Run the preview generator:

```bash
.venv/bin/python preprocessing/check_labels.py
```

It draws boxes for up to 50 randomly selected training images and writes them
to `preprocessing/detect_dataset/preview/`. Review the boxes against the
objects in the images. Correct bad boxes in their matching label text files;
if an image should not be used, remove both the image and its matching label.
Also inspect `skipped.txt` to see which source images YOLO-World could not
label.

These are **pseudo-labels**, not human-verified ground truth. YOLO-World can
miss an object, select a background object, or draw an inaccurate box. The
preview script displays only a sample of the training split; it does not
review validation or test labels. Review those splits as well before relying
on their scores.

## 5. Train the detector

After labels have been checked, run from the repository root:

```bash
.venv/bin/python train/train_detector.py
```

The default run performs three experiments, all at 480-pixel image size,
batch size 16, and seed 0:

1. **Transfer stage 1:** starts from `yolov8n.pt`, freezes the first ten
   layers, and trains for 15 epochs.
2. **Transfer stage 2:** starts from stage 1's best checkpoint, unfreezes the
   model, lowers the learning rate, and trains for 35 epochs.
3. **Scratch baseline:** starts from the `yolov8n.yaml` architecture with
   random weights and trains for 50 epochs.

Set `SMOKE=1` for a one-epoch-per-experiment pipeline check, or `DEVICE=cpu`
to force CPU execution. For example:

```bash
SMOKE=1 .venv/bin/python train/train_detector.py
DEVICE=cpu .venv/bin/python train/train_detector.py
```

Smoke runs have different run names and should not be treated as the full
experiment results.

Training run folders and logs are written under `train/runs/detect/`. The
script chooses between the two transfer stages using validation mAP50-95,
copies the chosen checkpoint to `train/final_model/best.pt`, then evaluates
the selected transfer model and scratch baseline on the configured test
split.

The current eight-class experiments use `det8_...` run names so they are not
confused with run folders created from the previous class configuration.
Older run folders and weights are retained as history; do not evaluate an old
checkpoint against the changed dataset. `evaluate.py` checks that checkpoint
class names match the current test folders and asks for retraining if they do
not.

## 6. Evaluation and generated results

After a full detector training run, optional analysis commands are:

```bash
.venv/bin/python train/compare_runs.py
.venv/bin/python train/evaluate.py
```

`compare_runs.py` plots validation mAP50 across the transfer and scratch
runs, writing `train/training_comparison.png` and
`train/comparison_table.txt`.

`evaluate.py` runs the selected detector on the original class-folder test
images in `data/office_items_dataset/test/`. It uses the highest-confidence
detected class as the image prediction and counts a missing box as `none`.
It writes `train/metrics.txt`, `train/misclassified.txt`, and
`train/confusion_matrix.png`. This is image-level classification-style
evaluation derived from detector predictions; it does not measure box
localization quality.

## Important interpretation and limitations

- Detector mAP is measured against the generated detection labels. Unless
  those labels are independently corrected, the score is agreement with the
  pseudo-labels, not verified real-world detection accuracy.
- The class-folder test data supplies an image-level expected class for
  `evaluate.py`; it does not supply a ground-truth bounding box.
- Do not use test metrics to select checkpoints or tune confidence,
  preprocessing, prompts, or other settings. Make those decisions using
  training/validation data, then evaluate the test split once at the end.
- The curation script's enhancement and train-only augmentation do not create
  new independent photographs. The reports distinguish augmented training
  files from real images.
- `train/detect_dataset/` is not the current detector data location. The
  detector YAML and generated image/label pairs live under
  `preprocessing/detect_dataset/`.

## Troubleshooting

**`Detector data.yaml is missing`:** run
`preprocessing/autolabel.py`. Confirm `data/office_items_dataset/{train,val,test}`
and matching class subfolders exist.

**A class/split is rejected by the labeler:** confirm each split has the same
class folders, and that every class/split folder contains supported images.

**An image appears in `skipped.txt`:** YOLO-World returned no box at the
configured confidence. Inspect the image and decide whether to annotate it
manually or leave it out.

**Training reports poor metrics:** first inspect/correct the generated boxes
and check class coverage in every split. Poor pseudo-label agreement may
reflect incorrect pseudo-labels rather than only a model-training problem.
