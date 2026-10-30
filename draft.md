
 ## Results

The YOLOv8n classifier was trained on 560 images across eight classes, starting from pretrained weights, and evaluated once on 119 held-out test images (15 per class, 14 for water bottle). It reached an accuracy of 0.983 and a macro-F1 of 0.983, with two errors. The same architecture trained from random weights for the same 30 epochs reached 0.647 accuracy and 0.643 macro-F1, so pretraining improved accuracy by about 34 percentage points. With so few test images per class, each image is worth about 0.8 percentage points, so these figures are indicative and not a precise accuracy estimate.

## Error analysis

A phone photo (`phone_013`) was predicted as `usb_cord` with confidence 0.71. It shows a Samsung phone with its screen on, and it is the only item in the image; the cause is not clear. A water bottle photo (`water_bottle_007`) was predicted as `pen` with confidence 0.75. The bottle is a small object tossed in the air in a busy scene with two children playing, so the folder label "water_bottle" does not describe most of what is in the image. Both errors were made with moderate to high confidence, so a confidence threshold alone would not remove them.

The class label comes from the folder name, and some photos contain more than one object. Performance on a cluttered table with several items has not been measured.

I trained a YOLOv8n detector on 560 training images, starting from pretrained weights, and evaluated it once on a held-out test split of 119 images (128 boxes, eight classes). It reached a precision of 0.879, a recall of 0.896, a mAP50 of 0.932 and a mAP50-95 of 0.696. The weakest classes were water bottle (mAP50 0.802) and pen (0.862); extension box, flash drive, notebook and phone were all at 0.995. On the validation split the same model reached a mAP50 of 0.753, with low recall on pen, notebook and usb cord, which likely reflects denser box labelling in the validation photos. Boxes began as YOLO-World pseudo-labels and were corrected by hand in makesense.ai, so these scores measure agreement with those boxes and not independent real-world accuracy. Each class has 14 to 21 test instances, so per-class figures are indicative.