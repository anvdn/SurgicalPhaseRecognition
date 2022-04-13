# Phase detection in hernia surgery

*Authors :* Antonin Vidon and Hugo Artigas

State-of-the-art CNN based architectures have
been developed to automatically recognize surgical phases in a
supervised learning fashion. This repo contains our work (team
HAAV) on a dataset of 193 hernia surgery videos whose frames
all belonged to one of 14 phases treated as labels. We exploited pretrained CNN architectures such
as ResNet and MobileNet to serve as backbone of our models.
We completed our architectures with LSTM (MobileNetLSTM)
or linear layers (MobileNetFC) to model temporal correlation
between frames and eventually added stage of the operation as
feature (MobileNetStage). In order to ensure higher performance,
we processed the label predictions of our models with a smoothing
function. Our best model achieved an accuracy of 80.0%
on the test set.

## Preprocessing

```
python videos-to-images.py # converts videos to frames
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
│   |── BEST.csv # best prediction on test set, reaching 80.0% accuracy
│   └── other predictions
├── report
│   |── Assignement2_av3023_ha2605.pdf
│   └── models.png
├── surgery.videos.hernia
├── weights # all weights
├── label.ipynb # create label df
├── models.py # all models
├── README.md
├── split.ipynb # split data into training, validation and testing set with/without oversampling and temporal organization
├── train-predict.py # script to train and make predictions on test set
├── train-predict.sh # template to train and make predictions on test set
├── train-validate.py # script to train and validate model with argparse
├── train-validate.sh # template to train and validate model with argparse
├── utils.py # all useful functions for preprocessing, training, predicting, ...
├── video.phase.trainingData.clean.StudentVersion.csv # initial labeling data
└── videos-to-images.py # script to convert and save videos to images
```

## Models

![models](./report/models.png)

## Results

| Model            | Validation acc.      | Macro F1 score      |
| ---------------- | -------------------- | ------------------- |
| MobileNet        | 0.8037               | 0.5545              |
| MobileNetStage   | 0.7988               | 0.5239              |
| MobileNetLSTM    | 0.7011               | 0.3265              |
| MobileNetFC      | 0.7002               | 0.3124              |
