# Phase detection in hernia surgery

## Abstract

State-of-the-art CNN based architectures have
been developed to automatically recognize surgical phases in a
supervised learning fashion. This repo contains our work (team
HAAV) on a dataset of 193 hernia surgery videos whose frames
all belong to one of 14 phases treated as labels. We exploited pretrained CNN architectures such
as ResNet and MobileNet to serve as backbone to our models.
We completed our architectures with LSTM (MobileNetLSTM)
or linear layers (MobileNetFC) to model temporal correlation
between frames and eventually added stage of the operation as
feature (MobileNetStage). In order to ensure higher performance,
we processed the label predictions of our models with a smoothing
function. Our best model achieved an accuracy of 80.0%
on the test set.

## Preprocessing

```
python videos-to-images.py # convert videos to frames
python label.ipynb # create label dataframe
python split.ipynb # split into training and validation, oversample, ...
```

## Organization of this directory

```./
├── dfs
│   |── labels.pkl
│   |── master_training_no_temp.pkl
│   |── master_training_temp.pkl
│   |── testing.pkl
│   |── training_no_temp_os.pkl
│   |── training_no_temp.pkl
│   |── training_temp.pkl
│   |── validation_no_temp_os.pkl
│   |── validation_no_temp.pkl
│   └── validation_temp.pkl
├── images
├── predictions
│   |── BEST.csv
│   └── other predictions
├── report
│   |── Assignement2_av3023_ha2605.pdf
│   └── models.png
├── surgery.videos.hernia
├── weights
├── label.ipynb
├── models.py
├── README.md
├── split.ipynb
├── train-predict.py
├── train-predict.sh
├── train-validate.py
├── train-validate.sh
├── utils.py
├── video.phase.trainingData.clean.StudentVersion.csv
└── videos-to-images.py
```

## Models

![models](./report/models.png)

## Results

| Model            | Validation acc.      | Macro F1 score      |
| ---------------- | -------------------- | ------------------- |
| MobileNet        | 0.8037               | 0.5545              |
| MobileNetStage   | 0.7988               | 0.5239              |
| MobileNetFC      | 0.7002               | 0.3124              |
| MobileNetLSTM    | 0.7011               | 0.3265              |
