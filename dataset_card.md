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

## 3. Test Set Breakdown & Class Percentages
We evaluated our main pipeline on a held-out test split of 118 total item instances across all 8 classes.

### Test Dataset Distribution (118 Total Instances)

| Class Name | Instance Count | Class Share (%) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Extension Box | 15 | 12.71% | 1.00 | 1.00 | 1.00 |
| Flash Drive | 14 | 11.86% | 0.88 | 1.00 | 0.93 |
| Mug | 15 | 12.71% | 0.70 | 0.93 | 0.80 |
| Notebook | 15 | 12.71% | 1.00 | 0.67 | 0.80 |
| Pen | 15 | 12.71% | 1.00 | 0.47 | 0.64 |
| Phone | 15 | 12.71% | 1.00 | 0.93 | 0.97 |
| USB Cord | 15 | 12.71% | 1.00 | 0.53 | 0.70 |
| Water Bottle | 14 | 11.86% | 0.73 | 0.79 | 0.76 |
| Total / Overall | 118 | 100.00% | 0.81 (Macro Avg) | 0.70 (Macro Avg) | 0.73 (Macro Avg) |

Note: We calculated each percentage as (Class Instance Count / 118) * 100%. Our model achieved an overall test accuracy of 78.8% (0.788) and a Macro-F1 score of 0.732.

## 4. How We Evaluated the Model
Following our script in train/evaluate.py, we tested predictions directly against the test set folder layout (TEST = data/office_items_dataset/test):

1. Prediction Selection: We set a confidence threshold of CONF = 0.25. The bounding box with the highest confidence score was picked as the final class guess for each test image.
2. Missing Detections: If the model didn't pick up any box above 0.25 confidence, we flagged the image as "none".
3. Exported Files: Running evaluation automatically outputs our metrics (train/metrics.txt), misclassified image list (train/misclassified.txt), and plot (train/confusion_matrix.png).

## 5. Model Benchmarks
Here is how our fine-tuned transfer model compared against training from scratch:

- Fine-Tuned Model (Stage 2 Transfer Learning):
  - Test mAP50: 0.903
  - Test mAP50-95: 0.783
  - Overall Test Accuracy: 78.8% (93 correctly predicted images out of 118, 25 errors).
- Scratch Baseline Model:
  - Test mAP50: 0.521
  - Test mAP50-95: 0.334

## 6. Challenges & Ideas for Improvement
1. Small & Thin Items: Thin objects like pens and usb_cords scored lower on recall (0.47 and 0.53) because they easily blend into messy desk backgrounds or get partially covered.
2. Web Image Variance: Differing image resolutions, watermarks, and lighting conditions across web-sourced Google images caused occasional misclassifications for reflective items like water bottles and mugs.
3. Next Steps: We can improve performance by sourcing cleaner images of thin accessories and applying stronger data augmentations like random cropping and copy-paste blending during preprocessing.