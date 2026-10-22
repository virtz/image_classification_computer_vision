# Custom Image Classification using YOLOv8n-cls

This repository contains the coursework project for developing a custom image classification model using **YOLOv8n-cls**. The project leverages **Transfer Learning** to adapt a high-performance deep learning model trained on a massive benchmark dataset to a specific custom classification task.

# Transfer Learning Methodology
Instead of training a deep convolutional neural network from scratch—which requires millions of images and significant computational power—this project utilizes **Transfer Learning**.

### 1. Pre-trained Base (Source Knowledge)
We initiate the model using `yolov8n-cls.pt`. These weights are pre-trained on the **ImageNet** dataset, which contains over 14 million images across 1,000 distinct categories. Through this pre-training, the model has already learned to extract fundamental visual features such as:
* Edges, curves, and gradients in its early layers.
* Textures, material patterns, and geometric shapes in its middle layers.

### 2. Fine-Tuning (Knowledge Transfer)
To adapt this baseline intelligence to our custom task, we replace the final classification layer (the head) of the network to match the number of classes in our custom dataset. During training, the model uses its existing feature-extraction knowledge and refines its internal weights to map those features to our specific target categories.
