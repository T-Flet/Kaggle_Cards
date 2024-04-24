---
title: Card Image Classifier Comparison
colorFrom: yellow
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---


# Card Image Classifier Comparison
[Kaggle page](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)

Data: 53 classes 7624 train, 265 test, 265 validation images 224 X 224 X 3.
The train, test and validation directories are partitioned into 53 sub directories, one for each of the 53 types of cards. The dataset also includes a csv file which can be used to load the datasets.


## Plan:

* Fine-tune a pretrained image classification model
    * Compare a few small efficient ones
    * Use old PyTorch utility functions
    * If enough time, redo in Lightning and write utility functions for it
* Extract outputs from the feature extraction layers and try completing the task with gradient boosting
* Compare results


## Comparison of best models of each type

The following shows the training and performance benefits of gradient boosting over NN classification layers.
Compare these models in [their huggingface space](https://huggingface.co/spaces/T-Flet/Kaggle-Cards) (or clone the repo and python app.py).

| **Model** | **Retrained Portion** | **Epochs** | **Time** | **test_loss** | **test_F1** |
| ----- | ----- | ----- | ----- | ----- | ----- |
| **RexNet 1.0** | Full Retrain | val_loss early stop at 5 (equalling 2)| 16:32 | 0.0996 | 1 |
| **RexNet 1.5 features -> LightGBM** | No retraining; feature extraction -> GB | 100 bagging gbdt iterations (OpenCL, not CUDA) | 3:37 |  | 0.5433 |
| **RexNet 1.5** | Classification | 10 | 8:00 | 1.5839 | 0.4884 |


