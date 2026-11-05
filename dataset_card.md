# Dataset Card: Workstation & Office Items Dataset

## 1. Overview & Summary
This dataset card documents our dataset of office items, which we collected and annotated for object detection and classification using YOLOv8. To build our dataset, we gathered high-resolution workstation and desktop item images online from Google Images across various environments, angles, and lighting setups to test how well edge computer vision models perform on real desktop items.

- Main Goal: Real-time detection and multi-class classification of standard office desktop items.
- Task Type: Multi-Class Bounding-Box Object Detection & Image Classification.
- Dataset Path: data/office_items_dataset/
- Target Setup: Personal computer desks, shared study tables, and online web images of office workspaces.
- Data Origin: Web-sourced collection gathered online via Google Images.

## 2. Classes & Data Format
Our dataset includes 8 target classes covering everyday office gear:

1. extension_box
2. flash_drive
3. mug
4. notebook
5. pen
6. phone
7. usb_cord
8. water_bottle

### Project Organization
- Bounding Box Format: Standard YOLO format ([class_id x_center y_center width height], normalized between 0 and 1).
- Folder Structure:
  - preprocessing/: Contains scripts we used for auditing the dataset (audit_dataset.py), auto-labeling (autolabel.py), and checking label accuracy (check_labels.py).
  - train/: Houses our model training scripts (train_detector.py, train_classifier.py) and evaluation pipeline (evaluate.py).
  - ui/: Streamlit frontend (app.py) for running live model predictions.

## 3. Dataset Splits & Test Set Distribution
The current dataset contains 800 images across eight classes. The test set is
held out from training and validation.

| Split | Images |
| :--- | ---: |
| Train | 560 |
| Validation | 120 |
| Test | 120 |

Each class has 70 training images, 15 validation images, and 15 test images.
The balanced test split therefore contains 12.5% of its images from each class.

| Class Name | Train | Validation | Test |
| :--- | ---: | ---: | ---: |
| Extension Box | 70 | 15 | 15 |
| Flash Drive | 70 | 15 | 15 |
| Mug | 70 | 15 | 15 |
| Notebook | 70 | 15 | 15 |
| Pen | 70 | 15 | 15 |
| Phone | 70 | 15 | 15 |
| USB Cord | 70 | 15 | 15 |
| Water Bottle | 70 | 15 | 15 |
| **Total** | **560** | **120** | **120** |

## 4. How We Evaluated the Model
The classifier evaluation uses the class-folder labels in
`data/office_items_dataset/test`. The detector evaluation script
(`train/evaluate.py`) also summarizes the highest-confidence detection per
image (or `none` when no box is detected). This image-level summary is not a
bounding-box localization metric.

1. Prediction Selection: We set a confidence threshold of CONF = 0.25. The bounding box with the highest confidence score was picked as the final class guess for each test image.
2. Missing Detections: If the model didn't pick up any box above 0.25 confidence, we flagged the image as "none".
3. Exported Files: The detector image-level evaluation writes
   `train/metrics.txt`, `train/misclassified.txt`, and
   `train/confusion_matrix.png`. The classifier evaluation writes its own
   metrics and confusion matrices under `train/`.

## 5. Model Benchmarks
Both classifier and detector results below use the current 120-image test
split. The classifier results are:

| Classifier | Accuracy | Macro-F1 | Incorrect predictions |
| :--- | ---: | ---: | ---: |
| Pretrained and fine-tuned YOLOv8n-cls | 0.983 | 0.983 | 2 |
| YOLOv8n-cls trained from scratch | 0.642 | 0.637 | 43 |

The detector's bounding-box evaluation covered 120 test images and 129
annotated boxes:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.927 | 0.873 | 0.916 | 0.697 |

As a separate image-level summary, using the highest-confidence detection per
image (or `none` when no box is detected), the detector achieved 0.933
accuracy and 0.844 macro-F1, with 8 incorrect predictions. These detector
metrics measure agreement with the available labels, which began as
YOLO-World pseudo-labels and were manually corrected; they are not an
independent ground-truth assessment.

## 6. Challenges & Ideas for Improvement
1. Small & Thin Items: Thin objects like pens and usb_cords scored lower on recall (0.47 and 0.53) because they easily blend into messy desk backgrounds or get partially covered.
2. Web Image Variance: Differing image resolutions, watermarks, and lighting conditions across web-sourced Google images caused occasional misclassifications for reflective items like water bottles and mugs.
3. Next Steps: We can improve performance by sourcing cleaner images of thin accessories and applying stronger data augmentations like random cropping and copy-paste blending during preprocessing.