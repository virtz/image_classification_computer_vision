# Project report: code overview and results

## Project overview

The project has two model workflows. The classifier predicts one class for an
image using its class-folder label. The optional detector predicts object
classes and bounding boxes. Detector labels were initialized with YOLO-World
pseudo-labels and can be manually corrected. The results below are from the
recorded experiments; configuration changes alone are not evidence of a new
or improved run.

## Source-code overview

| Area | Files | What they do |
|---|---|---|
| Dataset preparation | [`prepare_office_dataset.py`](prepare_office_dataset.py) | Validates and ranks raw images, filters duplicates, creates train/validation/test splits, augments training images only, and writes audit reports. |
| Detector labels | [`preprocessing/autolabel.py`](preprocessing/autolabel.py), [`preprocessing/auto_label_missing.py`](preprocessing/auto_label_missing.py) | Creates initial YOLO-World pseudo-boxes and attempts to add boxes for images still missing them. |
| Label QA and import | [`preprocessing/check_labels.py`](preprocessing/check_labels.py), [`preprocessing/import_labels.py`](preprocessing/import_labels.py), [`preprocessing/audit_dataset.py`](preprocessing/audit_dataset.py) | Renders boxes for review, validates/imports manual YOLO labels, and checks dataset coverage and label consistency. |
| Classifier | [`train/train_classifier.py`](train/train_classifier.py), [`train/evaluate_classifier.py`](train/evaluate_classifier.py) | Trains frozen-feature and fine-tuning stages plus a scratch baseline; selects between transfer stages by validation accuracy and reports test metrics. |
| Detector | [`train/train_detector.py`](train/train_detector.py), [`train/evaluate.py`](train/evaluate.py), [`train/compare_runs.py`](train/compare_runs.py) | Trains transfer and scratch detectors, reports detector test metrics, derives an image-level summary from detections, and plots validation mAP50 curves. |
| User interface | [`ui/app.py`](ui/app.py) | Provides local live-camera and image-upload detection using a trained detector checkpoint. |

## Dataset card

The image data used for training and evaluation lives in
[`data/office_items_dataset/`](data/office_items_dataset/). The repository also
contains a local backup copy on the Desktop for safety.

### Dataset purpose and scope

The dataset supports two tasks: image classification of office objects and
optional object detection with bounding boxes. It contains eight classes:
`extension_box`, `flash_drive`, `mug`, `notebook`, `pen`, `phone`,
`usb_cord`, and `water_bottle`.

### Split sizes and class counts

| Split | Images |
|---|---:|
| Train | 560 |
| Validation | 120 |
| Test | 119 |

Per-class counts are balanced in training and validation, while the test split
contains 14 `water_bottle` images and 15 images for each other class.

| Class | Train | Val | Test |
|---|---:|---:|---:|
| extension_box | 70 | 15 | 15 |
| flash_drive | 70 | 15 | 15 |
| mug | 70 | 15 | 15 |
| notebook | 70 | 15 | 15 |
| pen | 70 | 15 | 15 |
| phone | 70 | 15 | 15 |
| usb_cord | 70 | 15 | 15 |
| water_bottle | 70 | 15 | 14 |

### Data collection and known limitations

- Images are organized by class-folder labels and intended for local coursework
  and model-development experiments.
- The dataset is small by modern deep-learning standards and is best treated as
  an indicative benchmark rather than a production-grade benchmark dataset.
- Some classes are visually similar in cluttered scenes, and folder labels are
  used as the ground truth for classification even when a single image contains
  multiple objects or non-target clutter.
- The detector labels were initialized from YOLO-World pseudo-labels and then
  corrected by hand; they are useful for local experimentation but are not an
  independent ground-truth source.

### Data access and backup

The data used for the recorded experiments is already present in the repo under
[`data/office_items_dataset/`](data/office_items_dataset/). A full project backup
was created on the Desktop for safe archival storage.

## Recorded classifier results

The classifier used 560 training images across eight classes and was evaluated
once on 119 held-out test images (15 per class, except 14 water-bottle images).

| Model | Accuracy | Macro-F1 | Incorrect predictions |
|---|---:|---:|---:|
| Pretrained and fine-tuned YOLOv8n-cls | 0.983 | 0.983 | 2 |
| YOLOv8n-cls trained from scratch | 0.647 | 0.643 | 42 |

The recorded pretrained model scored about 34 percentage points higher in
accuracy than the scratch baseline. The test set is small, so one prediction
changes overall accuracy by about 0.8 percentage points; the scores should be
treated as indicative, not a precise estimate of future performance.

### Classifier error analysis

- `phone_013` was predicted as `usb_cord` with confidence 0.71. The phone is
  the only item in the image; the cause of the error is unclear.
- `water_bottle_007` was predicted as `pen` with confidence 0.75. The bottle
  is small in a busy scene with two children, and the folder label does not
  describe most of the image.

Both errors had moderate-to-high confidence, so increasing a confidence
threshold alone would not have removed them. Folder labels also simplify
images containing multiple objects; performance on cluttered scenes with
several office items has not been measured.

### Confusion matrix and test-set summary

The classifier evaluation script writes the confusion matrix to
[`train/classifier_confusion_matrix.png`](train/classifier_confusion_matrix.png)
and the text metrics to [`train/classifier_metrics.txt`](train/classifier_metrics.txt).
This held-out test split provides the final accuracy and macro-F1 values used in
this report.

## Recorded detector results

The detector test split contained 119 images and 128 boxes.

| Metric | Score |
|---|---:|
| Precision | 0.879 |
| Recall | 0.896 |
| mAP50 | 0.932 |
| mAP50-95 | 0.696 |

The validation mAP50 was 0.753. Test mAP50 was weakest for water bottle
(0.802) and pen (0.862); extension box, flash drive, notebook, and phone each
scored 0.995. These metrics measure agreement with the available boxes,
which began as YOLO-World pseudo-labels and were corrected by hand; they do
not independently establish real-world detection accuracy. Each class had
only 14–21 test instances, so per-class values are uncertain.

## Evaluation and limitations

Classifier evaluation reports accuracy, macro-F1, per-class precision/recall/F1,
a confusion matrix, and misclassified examples. Detector training selects a
transfer stage using validation mAP50-95; final detector evaluation reports
precision, recall, mAP50, and mAP50-95 on the held-out test split. The separate
`train/evaluate.py` script reduces detector output to its highest-confidence
box per image (or `none`) for a classification-style summary; this is not a
bounding-box localization metric.

The test set is reserved for final evaluation, not model selection. Detector
scores depend on label quality and completeness, and pseudo-labels are not
guaranteed ground truth. Additional data and independent, carefully reviewed
annotations would be needed to make stronger claims about generalization.
