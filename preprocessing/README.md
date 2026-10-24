# Object-detection preprocessing

The normal workflow consumes the already curated classification splits in
`../data/office_items_dataset/`. From the repository root, run:

```bash
python preprocessing/autolabel.py
python preprocessing/check_labels.py
python train/train_detector.py
```

`autolabel.py` uses YOLO-World to produce pseudo-boxes, writing images,
labels, and `data.yaml` to `preprocessing/detect_dataset/`. It downloads
`yolov8s-worldv2.pt` on first use. Inspect the previews and correct bad boxes
before training; the generated labels are not guaranteed ground truth.

To curate a new raw-image collection instead, put one-object photos in
`preprocessing/raw_dataset/<class_name>/` and run
`python prepare_office_dataset.py` from the repository root. It produces the
class-folder splits in `data/office_items_dataset/`, which then become the
input to `autolabel.py`. Use `--overwrite` only when you intend to back up and
replace the existing curated dataset.

Keep the test split for final evaluation only. Note that evaluations against
automatically generated labels measure agreement with the pseudo-labeler,
not independently verified detection accuracy.
