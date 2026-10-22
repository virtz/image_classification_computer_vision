
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

# Stage 1:feature extraction. Pretrained backbone frozen, only the new head trains.
DATA = '/Users/macbook/Documents/GitHub/image_classification/code/data/office_data_set'
model.train(data=DATA,
            epochs=10, batch=64,
            name='office_stage1_frozen',
            freeze=9,
            imgsz=224,

            lr0=1e-3,
            seed=0,
            optimizer='AdamW',
            pretrained=True,



            )
model = YOLO('runs/classify/office_stage1_frozen/weights/best.pt')
model.train(data=DATA,
            epochs=20, batch=64,
            name='office_stage2_fine_tune',
            freeze=0,
            imgsz=224,
            lr0=1e-4,
            seed=0,

            # seed=0,
            optimizer='AdamW',

            )
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
)