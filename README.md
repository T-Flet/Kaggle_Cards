# Cards Image Dataset-Classification
[Kaggle page](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)

The data is in E:\Data_and_Models\Kaggle_Cards (53 classes 7624 train, 265 test, 265 validation images 224 X 224 X 3).
The train, test and validation directories are partitioned into 53 sub directories, one for each of the 53 types of cards. The dataset also includes a csv file which can be used to load the datasets.


## Plan:

* Fine-tune a pretrained image classification model
    * Compare a few small efficient ones
    * Use old PyTorch utility functions
    * If enough time, redo in Lightning and write utility functions for it
* Extract outputs from the feature extraction layers and try completing the task with gradient boosting
* Compare results


