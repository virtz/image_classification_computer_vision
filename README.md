# Office Item Classification and Detection

This project trains an image classifier for office items and an optional YOLO object detector. It also includes a local Streamlit app for running the detector on a live camera or uploaded image.

## 1) Install

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If needed, the first run will download pretrained YOLO weights.

## 2) Dataset

The classification dataset should look like this:

```text
data/office_items_dataset/
├── train/
│   ├── extension_box/
│   ├── flash_drive/
│   ├── mug/
│   ├── notebook/
│   ├── pen/
│   ├── phone/
│   ├── usb_cord/
│   └── water_bottle/
├── val/
└── test/
```

If you are starting from raw images, prepare the dataset first:

```bash
python prepare_office_dataset.py --raw preprocessing/raw_dataset
```

## 3) Train the classifier

```bash
python train/train_classifier.py
```

This trains a pretrained classifier and a scratch baseline. The best classifier is saved to:

```text
train/final_model/classifier_best.pt
```

## 4) Evaluate the classifier

```bash
python train/evaluate_classifier.py
```

Expected outputs:

```text
train/classifier_metrics.txt
train/classifier_misclassified.txt
train/classifier_confusion_matrix.png
```

These files contain accuracy, macro-F1, confusion matrix, and misclassified examples.

## 5) Train the detector (optional)

Generate detector labels and review them:

```bash
python preprocessing/autolabel.py
python preprocessing/check_labels.py
```

Then train the detector:

```bash
python train/train_detector.py
```

This runs the detector experiments and saves the final model to:

```text
train/final_model/best.pt
```

Use CPU only if needed:

```bash
DEVICE=cpu python train/train_detector.py
```

## 6) Run the detection UI

```bash
streamlit run ui/app.py
```

Optional custom model path:

```bash
export OFFICE_DETECTOR_MODEL="/path/to/train/final_model/best.pt"
streamlit run ui/app.py
```

Windows PowerShell:

```powershell
$env:OFFICE_DETECTOR_MODEL = "C:\path\to\train\final_model\best.pt"
streamlit run ui/app.py
```

The app supports live camera input and image upload. It shows detected objects, confidence scores, and bounding boxes.

## 7) Expected outputs

After a successful run, you should have:

```text
train/final_model/classifier_best.pt
train/final_model/best.pt
train/classifier_metrics.txt
train/classifier_confusion_matrix.png
train/runs/
```

The `train/runs/` folder contains logs and checkpoints for training runs.

## 8) Troubleshooting

### Dataset not found

```bash
ls data/
ls data/office_items_dataset/
```

If missing, run:

```bash
python prepare_office_dataset.py --raw preprocessing/raw_dataset
```

### Model checkpoint not found

Check that the files exist:

```bash
ls train/final_model/
```

If missing, rerun the training scripts.

### Camera or UI problems

- Use localhost or HTTPS in the browser
- Allow camera permission
- Confirm the model file exists
- Lower the confidence threshold in the sidebar

### Dependency problems

```bash
python -m pip install -r requirements.txt
```

## Project references

See:

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md)
- [`PIPELINE.md`](PIPELINE.md)

These files contain the recorded results, model summary, and error analysis.
