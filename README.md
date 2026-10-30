# Office Item Classification and Detection

This coursework project builds two computer-vision pipelines for eight kinds of
office items:

1. **Image classification** with YOLOv8n-cls. The model predicts one class for
   an image.
2. **Object detection** with YOLOv8n. The model predicts object classes and
   bounding boxes. Its training labels are initialized with YOLO-World
   pseudo-labels and can be corrected by hand before training.

The classification dataset is organized by class folders. The detection
pipeline derives images and YOLO-format bounding-box labels from those same
folders. The repository includes data-preparation and label-audit utilities,
training and evaluation scripts, and the results from the recorded classifier
experiments.

## Classes

Class names are ordered alphabetically, which is also the class-ID order used
by the detector:

| ID | Class |
|---:|---|
| 0 | `extension_box` |
| 1 | `flash_drive` |
| 2 | `mug` |
| 3 | `notebook` |
| 4 | `pen` |
| 5 | `phone` |
| 6 | `usb_cord` |
| 7 | `water_bottle` |

[`labels.txt`](labels.txt) lists the class names in order. The classifier
training script also maintains [`train/class_names.txt`](train/class_names.txt)
as a numbered class map.

## Repository layout

```text
.
├── README.md
├── PIPELINE.md
├── requirements.txt
├── labels.txt
├── prepare_office_dataset.py
├── data/
│   ├── office_items_dataset/       # curated classification train/val/test
│   └── office_data_set/reports/    # reports retained from an earlier curation
├── preprocessing/
│   ├── README.md
│   ├── audit_dataset.py
│   ├── autolabel.py
│   ├── auto_label_missing.py
│   ├── check_labels.py
│   └── import_labels.py
├── train/
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── train_detector.py
│   ├── evaluate.py
│   ├── compare_runs.py
│   └── classifier result reports and plots
└── pretrained model checkpoints
```

Training runs, intermediate detector data, caches, and `.pt` checkpoints are
generated or stored locally and are excluded from version control by
`.gitignore`. In particular, a locally generated `preprocessing/detect_dataset/`
is not included as a source dataset; generate it from the curated classification
images before running the detector workflow.

## Setup

Use Python 3.10 or newer and install the packages listed in
[`requirements.txt`](requirements.txt). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The training scripts use Ultralytics, which downloads pretrained checkpoints
when needed; this requires network access on first use. The listed dependencies
include Ultralytics, PyTorch's runtime dependencies, OpenCV, Pillow,
Matplotlib, SciPy, and scikit-learn.

On Windows, activate the environment with ⁠ .venv\Scripts\activate ⁠ instead. On macOS, activate it with ⁠source .venv/bin/activate ⁠.

## Run the local detection UI

The interface supports a live browser camera stream and single-image detection.
All inference runs on the machine hosting Streamlit; frames and uploaded images
are not sent to a cloud inference service. For a Raspberry Pi demonstration,
run the app on the Pi and open its Streamlit address from a browser that can
reach it. Camera access requires localhost or HTTPS in most browsers.

The detector checkpoint is intentionally excluded from Git. Copy the trained
detector to `train/final_model/best.pt`, or point `OFFICE_DETECTOR_MODEL` at its
path. (The classifier checkpoint does not produce boxes and cannot be used for
this UI.) Then install requirements and launch from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run ui/app.py
```

To use a checkpoint elsewhere:

```bash
OFFICE_DETECTOR_MODEL=/path/to/best.pt streamlit run ui/app.py
```

In **Live camera**, press **START** and allow camera access. In **Image file**,
choose a JPG or PNG to view the annotated detections and download the result.
The sidebar controls the confidence threshold and inference image size. FPS is
measured from processed camera frames and depends on hardware and the selected
checkpoint; the UI does not claim the >5 FPS target until measured on the target
Raspberry Pi.

| UI parameter | Default | Purpose |
|---|---:|---|
| Model checkpoint | `train/final_model/best.pt` | Trained YOLO detector weights |
| Confidence threshold | 0.35 | Hide detections below this confidence |
| Inference size | 320 px | YOLO input size; lower values can improve edge speed |

⁠*Tip:* You can quickly stop the Streamlit local server at any time by pressing ⁠ Ctrl + C ⁠ in your terminal.

### UI troubleshooting

- **Checkpoint not found:** place the detector weights at the default path or
  set `OFFICE_DETECTOR_MODEL` to the checkpoint location.
- **Camera does not start:** allow browser camera permission; use localhost or
  HTTPS, and confirm the Pi camera is available to the host OS.
- **Low frame rate:** try 224 or 256 input size and check CPU load. Actual
  throughput depends on the detector export and Raspberry Pi configuration.
- **No boxes appear:** lower the confidence threshold and confirm you loaded the
  detector checkpoint, not the classifier checkpoint.
- **PyArrow reports that NumPy 2 is required:** reinstall the pinned compatible
  version with `python -m pip install -r requirements.txt` from the project
  environment.

Commands below assume the environment is activated and are run from the
repository root. Alternatively, use `.venv/bin/python` in place of `python` on
macOS/Linux.

## Dataset structure

### Curated classification dataset

The classifier reads
[`data/office_items_dataset/`](data/office_items_dataset/), which should have
the following Ultralytics-compatible layout:

```text
data/office_items_dataset/
├── train/
│   ├── extension_box/
│   ├── flash_drive/
│   └── ... one folder per class
├── val/
│   └── ... the same class folders
└── test/
    └── ... the same class folders
```

Folder names are the ground-truth class labels for classification. The current
recorded experiment used 560 training images and 119 held-out test images;
class counts are not perfectly balanced (the test split contains 14
`water_bottle` images and 15 images for each other class).

### Optional dataset curation

[`prepare_office_dataset.py`](prepare_office_dataset.py) curates raw images
from folders named by class. It checks image integrity and quality, tracks
duplicates, creates train/validation/test splits, and writes reports and
contact sheets under the output directory (by default,
`data/office_items_dataset/reports/` and `data/office_items_dataset/figures/`).
The repository also retains report files from an earlier curation under
`data/office_data_set/reports/`. When enough real,
distinct images are available, its documented target is 100 images per class
(70 train, 15 validation, 15 test); it augments training images only and
records shortfalls rather than silently treating synthetic variants as new
real photographs.

To curate another collection, put images in
`preprocessing/raw_dataset/<class_name>/` and run:

```bash
python prepare_office_dataset.py --raw preprocessing/raw_dataset
```

Review the generated reports and contact sheets before using the resulting
dataset. The raw collection is not modified. See
[`PIPELINE.md`](PIPELINE.md) and [`preprocessing/README.md`](preprocessing/README.md)
for additional operational notes. Do not replace the existing curated dataset
unless intended; the preparation script supports `--overwrite` for an
intentional replacement.

## Classification workflow

### Train

```bash
python train/train_classifier.py
```

[`train/train_classifier.py`](train/train_classifier.py) runs three
30-epoch-budget experiments with YOLOv8n-cls:

1. **Frozen-feature stage:** starts with `yolov8n-cls.pt`, freezes the
   backbone, and trains the classification head for 10 epochs.
2. **Fine-tuning stage:** starts from stage one's best checkpoint and trains
   the full network for 20 epochs at a lower learning rate.
3. **Scratch baseline:** trains the same architecture from random
   initialization for 30 epochs.

The better of the two transfer-learning stages is selected using validation
accuracy, not test performance. Its checkpoint is saved as
`train/final_model/classifier_best.pt`, and the class map is written to
`train/class_names.txt`. Training runs are written under
`train/runs/classify/`.

### Evaluate

After training and choosing the final model, evaluate the held-out test split
once:

```bash
python train/evaluate_classifier.py
```

[`train/evaluate_classifier.py`](train/evaluate_classifier.py) evaluates the
selected transfer model and the scratch baseline. It checks that checkpoint
class names match the test-folder classes and writes accuracy, macro-F1,
per-class metrics, a confusion matrix, and a list of incorrect predictions.
Do not use test results to tune or choose a model; use the validation split
for model selection.

### Recorded classifier results

| Model | Test images | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|---:|
| Pretrained / fine-tuned YOLOv8n-cls | 119 | 0.983 | 0.983 | 2 |
| YOLOv8n-cls trained from scratch | 119 | 0.647 | 0.643 | 42 |

The fine-tuned model's two recorded errors were a phone predicted as
`usb_cord` and a `water_bottle` image predicted as `pen`. The latter photo
contains a small bottle in a busy scene, illustrating that the folder label
does not necessarily describe most of the image. The scratch baseline is
included as a comparison, not as the selected model.

Results and supporting outputs:

- [`train/classifier_metrics.txt`](train/classifier_metrics.txt) and
  [`train/classifier_confusion_matrix.png`](train/classifier_confusion_matrix.png):
  fine-tuned model results.
- [`train/classifier_misclassified.txt`](train/classifier_misclassified.txt):
  incorrect fine-tuned predictions, including the true class, predicted class,
  and confidence.
- [`train/classifier_scratch_metrics.txt`](train/classifier_scratch_metrics.txt),
  [`train/classifier_scratch_confusion_matrix.png`](train/classifier_scratch_confusion_matrix.png),
  and [`train/classifier_scratch_misclassified.txt`](train/classifier_scratch_misclassified.txt):
  corresponding scratch-baseline results.
- [`draft.md`](draft.md): additional interpretation of the errors and
  limitations.

## Object-detection workflow

Detection is optional and uses images from the same classification splits.
It first predicts one bounding box per image using YOLO-World, then allows
manual corrections before training a detector. Generated boxes are
pseudo-labels, not ground truth.

### Generate and inspect labels

```bash
python preprocessing/autolabel.py
python preprocessing/check_labels.py
```

[`preprocessing/autolabel.py`](preprocessing/autolabel.py) uses
`yolov8s-worldv2.pt` to predict a box for the expected class in each image.
It creates `preprocessing/detect_dataset/` with `images/`, `labels/`, and
`data.yaml`; images without a predicted box are listed in `skipped.txt`.
If an output dataset already exists, the script moves it to a timestamped
backup before generating a new one. This preserves the old files but means
manual changes in the old dataset are not automatically carried into the new
output.

[`preprocessing/check_labels.py`](preprocessing/check_labels.py) renders a
sample of the test split's boxes into `preprocessing/detect_dataset/preview/`.
Review these previews and fix inaccurate boxes before training.

### Correct or complete labels

Export corrected YOLO labels from makesense.ai, then first preview the changes:

```bash
python preprocessing/import_labels.py path/to/export_folder_or_zip --dry-run
```

If the report is correct, import them by omitting `--dry-run`:

```bash
python preprocessing/import_labels.py path/to/export_folder_or_zip
```

[`preprocessing/import_labels.py`](preprocessing/import_labels.py) accepts a
folder or ZIP export. It can replace labels for images already in the detector
dataset, or add labels and matching images for source images that were
previously skipped. It validates YOLO label structure, class IDs, normalized
coordinates, and class-folder consistency; invalid or ambiguous files are
reported and skipped. Use dry-run first and inspect the result.

If autolabeling skipped images, add predictions for only the missing images
without replacing existing label files:

```bash
python preprocessing/auto_label_missing.py
```

[`preprocessing/auto_label_missing.py`](preprocessing/auto_label_missing.py)
uses a lower confidence threshold than the initial autolabeling pass to find
additional boxes. This can increase false positives; manually review all new
boxes. It writes images that still have no predicted box to
`skipped_after_retry.txt`.

Finally, audit the dataset and label files:

```bash
python preprocessing/audit_dataset.py
```

[`preprocessing/audit_dataset.py`](preprocessing/audit_dataset.py) compares
source images with generated detector images, checks image/label pairing and
YOLO label validity, counts boxes by class, and warns about classes with few
training or evaluation boxes. It writes the current missing-image list to
`preprocessing/detect_dataset/missing_boxes.txt`. This consolidated audit
replaces the former separate label-count and missing-image checks.

### Train and evaluate the detector

```bash
python train/train_detector.py
```

[`train/train_detector.py`](train/train_detector.py) runs a 15-epoch frozen
backbone stage, a 35-epoch full fine-tuning stage, and a 50-epoch scratch
baseline. It chooses between the transfer stages using validation mAP50-95,
then evaluates the chosen transfer model and scratch model on the test split.
The test split should only be used after training and model selection are
complete. A brief pipeline check is available with `SMOKE=1`; it reduces each
stage to one epoch. Set `DEVICE=cpu` to force CPU execution.

The detector's recorded held-out test results were:

| Metric | Score |
|---|---:|
| Precision | 0.879 |
| Recall | 0.896 |
| mAP50 | 0.932 |
| mAP50-95 | 0.696 |

The recorded test split contained 119 images and 128 boxes. Water bottle
(mAP50 0.802) and pen (0.862) were the weakest classes; extension box, flash
drive, notebook, and phone each reached 0.995 mAP50. The validation mAP50 was
0.753 in the recorded run.

These detection metrics measure agreement with the available boxes, which
began as YOLO-World pseudo-labels and were corrected by hand. They are not an
independent measure of real-world detection accuracy. Each class has only
14-21 test instances, so per-class results are indicative rather than precise.

Other detection utilities:

- [`train/evaluate.py`](train/evaluate.py) evaluates a detector as an
  image-level classifier by assigning each test image the class of its
  highest-confidence detection (or `none` if there is no detection). It writes
  image-level accuracy, macro-F1, a confusion matrix, and misclassified
  images.
- [`train/compare_runs.py`](train/compare_runs.py) compares validation mAP50
  curves from the detector's three runs and writes
  `train/training_comparison.png` and `train/comparison_table.txt`.

## Code reference

| File | Purpose |
|---|---|
| [`prepare_office_dataset.py`](prepare_office_dataset.py) | Validates and curates raw class-folder images, identifies duplicates and quality issues, creates splits, augments training data where appropriate, and generates curation reports. |
| [`preprocessing/autolabel.py`](preprocessing/autolabel.py) | Generates the initial YOLO-World detector dataset and `data.yaml`. |
| [`preprocessing/auto_label_missing.py`](preprocessing/auto_label_missing.py) | Attempts to pseudo-label source images missing from the detector dataset without overwriting existing image/label pairs. |
| [`preprocessing/import_labels.py`](preprocessing/import_labels.py) | Validates and imports makesense.ai YOLO label exports from a folder or ZIP; supports a no-write dry run. |
| [`preprocessing/check_labels.py`](preprocessing/check_labels.py) | Creates preview images with drawn bounding boxes for manual review. |
| [`preprocessing/audit_dataset.py`](preprocessing/audit_dataset.py) | Audits source coverage, detector labels, image/label pairing, class counts, and missing boxes. |
| [`train/train_classifier.py`](train/train_classifier.py) | Trains pretrained, fine-tuned, and scratch image-classification experiments; selects the transfer checkpoint by validation accuracy. |
| [`train/evaluate_classifier.py`](train/evaluate_classifier.py) | Evaluates the selected classifier and scratch baseline on the test split and saves metrics, errors, and confusion matrices. |
| [`train/train_detector.py`](train/train_detector.py) | Trains frozen, fine-tuned, and scratch object detectors; selects by validation mAP50-95 and reports test metrics. |
| [`train/evaluate.py`](train/evaluate.py) | Reports image-level class predictions derived from the detector's highest-confidence box. |
| [`train/compare_runs.py`](train/compare_runs.py) | Plots and tabulates detector validation mAP50 by epoch. |

## Notable workflow changes

- Dataset auditing is consolidated in `preprocessing/audit_dataset.py`; the
  older standalone label-count, verification, and missing-image scripts have
  been removed.
- `preprocessing/import_labels.py` adds a guarded path for bringing corrected
  makesense.ai labels into the generated dataset, including a dry-run option
  and support for labels on images that were initially skipped.
- The missing-image retry is a separate operation from the initial
  autolabeling pass, so it can add boxes without replacing existing
  corrections. The label-preview script also validates YOLO rows before
  drawing them.
- `labels.txt` now uses one class name per line. Classifier-generated IDs are
  recorded separately in `train/class_names.txt`.
- Classifier result reports, confusion matrices, and the written error
  analysis document the recorded pretrained-versus-scratch comparison. The
  exact results and their limitations are summarized above.

## Current limitations and interpretation

- The classifier's 119-image test set is small. With 119 images, one prediction
  changes overall accuracy by about 0.84 percentage points; reported metrics
  are useful comparisons for this experiment, not precise estimates of
  production performance.
- Some images contain multiple objects, clutter, or a small target. The
  classification folder label describes the assigned class, but may not
  describe the dominant content of every photograph.
- Performance on scenes containing several office items has not been
  independently measured.
- Detector scores depend on the quality and completeness of the manually
  corrected pseudo-labels. Validation/test labels must be reviewed carefully
  before treating mAP as meaningful.
- Keep the test split out of model selection and threshold tuning. Use
  validation results during development and reserve test evaluation for the
  final model.

## Further documentation

- [`PIPELINE.md`](PIPELINE.md) describes the end-to-end dataset preparation,
  detector-labeling, training, and evaluation flow.
- [`preprocessing/README.md`](preprocessing/README.md) focuses on preparing
  and auditing the detection dataset.
- [`draft.md`](draft.md) contains the detailed recorded classifier error
  analysis and detector evaluation caveats.
