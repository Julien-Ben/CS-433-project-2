# Road Segmentation - ML Project 2 

<!-- ## On the empirical comparison between a data-augmented U-Net and a patch-wise CNN -->

## Introduction

<table>
    <tr>
        <td>
            <img src="/assets/readme_img_1.png" width="300" height="300" />
        </td>
        <td>
            <img src="/assets/readme_img_2.png" width="300" height="300" />
        </td>
    </tr>
</table>

In the context of the [EPFL Road Segmentation AICrowd challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation), our goal is to create a machine learning model that labels every 16x16 patches as either `road` or `background` on satellite images from GoogleMaps. The dataset is composed of 100 training images along with their respective grountruths and the 50-image test set whose predictions are to be submitted on AICrowd. 
For that, we trained a U-Net model that predicts each pixel's class as well as a Convolutional Neural Network trained to output a 2D-grid of labels: one for each patch of the image. In the end the U-Net, trained using heavy data augmentation, was more performant and is used for our final AICrowd submission ([#109366](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/109366)), in which we reached 90% of F1 score and 94.5% of accuracy.

### Team members
* Julien Benhaim
* Hugo Lepeytre
* Julien Vignoud 

## Table of contents

1. [Submission reproduction](#submission-reproduction)
2. [Folder architecture](#folder-architecture)

## Submission reproduction

First install the required libraris and packages used in this project:
```
pip install -r requirements.txt
```
The setup is done and all is left is to create the predictions on the test set:
```
python run.py
```
It will load the model, run predictions on the test set and create a `.csv` containing a label for each patch.
The submission file is created in the top folder as `submission.csv`. Submitting this file on AICrowd yields an F1-score of 90%, identical to our submission [#109366](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/109366).

## Folder architecture

This folder contains several python modules, categorized as follows :
```
project
│   README.md
│   requirements.txt
|   run.py    
│
└───data
│   └───training
│   └───test_set_images
|   └───predictions
│   └───generated
|       └───flip
│       └───rotation
│       │   ...   
│   
└───helpers
|       colab.py
|       constants.py
|       file_manipulation.py
│        ...
|
└───model_save
|       final_model
|
└───notebooks
│       create_submission.ipynb
│       data_augmentation.ipynb
|       ...
|
└───assets
│       ...
```

Here are the main subfolders description:
<details open>
    <summary>data</summary>
    <br/>
    ajsdkskadjakd
</details>

<details>
    <summary>helpers</summary>
    <br>
    asdasda
</details>
<details>
    <summary>model_save</summary>
    <br>
    asdas
</details>