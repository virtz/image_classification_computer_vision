import os
from pathlib import Path
from ultralytics import YOLO

# Dynamically resolve project base path relative to this script
BASE_DIR = Path(__file__).resolve().parent
DATA = str(BASE_DIR / 'code' / 'data' / 'office_data_set')

# Stage 1: Feature extraction. Pretrained backbone frozen, only the new head trains.
model_stage1 = YOLO('yolov8n-cls.pt')
results_stage1 = model_stage1.train(
    data=DATA,
    epochs=10,
    batch=64,
    name='office_stage1_frozen',
    freeze=9,
    imgsz=224,
    lr0=1e-3,
    seed=0,
    optimizer='AdamW',
    pretrained=True,
    exist_ok=True
)

# Stage 2: Fine-tuning using the best weights from Stage 1
best_weights_stage1 = str(Path(results_stage1.save_dir) / 'weights' / 'best.pt')
model_stage2 = YOLO(best_weights_stage1)
model_stage2.train(
    data=DATA,
    epochs=20,
    batch=64,
    name='office_stage2_fine_tune',
    freeze=0,
    imgsz=224,
    lr0=1e-4,
    seed=0,
    optimizer='AdamW',
    exist_ok=True
)

# Stage 3: Train baseline model from scratch
scratch = YOLO('yolov8n-cls.yaml')
scratch.train(
    data=DATA,
    pretrained=False,
    epochs=30,
    imgsz=224,
    batch=64,
    optimizer='AdamW',
    lr0=1e-3,
    seed=0,
    name='office_scratch_baseline',
    exist_ok=True
)