from barbar import Bar
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

# memorize all paths
videos_path = os.path.join(os.getcwd(), 'surgery.videos.hernia')
csv_path = os.path.join(os.getcwd(), 'video.phase.trainingData.clean.StudentVersion.csv')
dfs_path = os.path.join(os.getcwd(), 'dfs')
labels_path = dfs_path + '/labels.pkl'
images_path = os.path.join(os.getcwd(), 'images')
weights_path = os.path.join(os.getcwd(), 'weights')
predictions_path = os.path.join(os.getcwd(), 'predictions')
kaggle_template_path = os.path.join(os.getcwd(), 'predictions')

# memorize number of classes
num_classes = pd.read_pickle(labels_path).label.unique().size

# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_frames(videoname):
    """
    Description
    -------------
    Count number of frames in video

    Parameters
    -------------
    videoname   : name of the video file without the extension (.mp4 or .mov)

    Returns
    -------------
    number of frames in the video
    """
    if os.path.exists(videos_path + '/' + videoname + '.mp4'):
        video = cv2.VideoCapture(videos_path + '/' + videoname + '.mp4')
    elif os.path.exists(videos_path + '/' + videoname + '.mov'):
        video = cv2.VideoCapture(videos_path + '/' + videoname + '.mov')
    else:
        return 'no video of this name'
    totalframecount= int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return totalframecount

def save_frames(videoname, resize = (224,224)):
    """
    Description
    -------------
    Resize and save frames of a video

    Parameters
    -------------
    videoname   : name of the video file without the extension (.mp4 or .mov)
    resize      : tuple, shape of the resized frame

    Returns
    -------------
    tensor of dimension (#frames,channels,width,height)
    """
    frames = []
    if os.path.exists(videos_path + '/' + videoname + '.mp4'):
        video = cv2.VideoCapture(videos_path + '/' + videoname + '.mp4')
    elif os.path.exists(videos_path + '/' + videoname + '.mov'):
        video = cv2.VideoCapture(videos_path + '/' + videoname + '.mov')
    else:
        return 'no video of this name'
    # select location to save all frames
    frames_path = images_path + '/' + videoname
    # create folder to store frames if it does not already exist
    if not os.path.exists(frames_path): os.makedirs(frames_path)
    # read first frame
    success, image = video.read()
    # count frames of video
    number_frames = count_frames(videoname=videoname)
    for num_frame in range(number_frames):
        # resize frame
        image = cv2.resize(image, resize, interpolation = cv2.INTER_AREA)
        # save the frame
        if not cv2.imwrite(frames_path + '/' + str(num_frame) + '.jpg', image):
            return 'could not save frame'
        # function extract frames
        success, image = video.read()
    video.release()
    # print videoname
    print(videoname)

def transforms(resize, mean, std):
    """
    Description
    -------------
    Preprocess image screen before feeding it to a neural network.
    
    Parameters
    -------------
    resize : tuple, shape of the resized frame
    mean   : mean values to normalize the frame (ex = [0.485, 0.456, 0.406])
    std    : standard deviation values to normalize the frame (ex = [0.229, 0.224, 0.225])
    
    Returns
    -------------
    torchvision.transforms.transforms.Compose object, the composed transformations.
    """
    if mean and std:
        return T.Compose([T.ToPILImage(),
                T.Resize(resize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)])
    
    else:
        return T.Compose([T.ToPILImage(),
                T.Resize(resize),
                T.ToTensor()])


def get_frames(videoname, resize, mean, std):
    """
    Description
    -------------
    Resize and stack frames of a video

    Parameters
    -------------
    videoname   : name of the video file without the extension (.mp4 or .mov)
    resize      : tuple, shape of the resized frame
    mean        : mean values to normalize the frame (ex = [0.485, 0.456, 0.406])
    std         : standard deviation values to normalize the frame (ex = [0.229, 0.224, 0.225])

    Returns
    -------------
    tensor of dimension (#frames,channels,width,height)
    """
    frames = []
    if os.path.exists(videos_path + '/' + videoname + '.mp4'):
        video = cv2.VideoCapture(videos_path + '/' + videoname + '.mp4')
    elif os.path.exists(videos_path + '/' + videoname + '.mov'):
        video = cv2.VideoCapture(videos_path + '/' + videoname + '.mov')
    else:
        return 'no video of this name'
    # read first frame
    success, image = video.read()
    # count frames of video
    number_frames = count_frames(videoname=videoname)
    for _ in range(number_frames):
        # brg -> rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # send tensor image to device
        image = transforms(resize, mean, std)(image)
        # save the frame to the list of frames
        frames.append(image)
        # function extract frames
        success, image = video.read()
    video.release()
    # stack frames
    frames = torch.stack(frames, dim=0).to(device)
    return frames

def get_labels(videoname):
    """
    Description
    -------------
    Produces tensor of labels for each frame of a video

    Parameters
    -------------
    videoname   : name of the video file without the extension (.mp4 or .mov)

    Returns
    -------------
    tensor of dimension (#frames)
    """
    if not os.path.exists(labels_path): return 'no labels stored'
    # recover all labels
    all_labels = pd.read_pickle(labels_path)
    # recover labels of the video
    labels = torch.tensor(all_labels.loc[all_labels['videoname'] == videoname]['label'].tolist())
    labels = F.one_hot(labels, num_classes=num_classes).to(device)
    return labels

def get_train_test_video_names(videos_path = videos_path, labels_path = labels_path):
    """
    Description
    -------------
    List names of train and test videos

    Parameters
    -------------
    videos_path    : path to directory of videos
    labels_path    : path to labels file (created in the labeling notebook)

    Returns
    -------------
    (train_names, test_names) , each of the tuple elements is a list of strings
    """
    video_names = {'train': [], 'test': []}

    # list all videos
    all_video_files = os.listdir(videos_path) 

    # get list of train video names
    with open(labels_path, 'rb') as labels_file:
        labels = pickle.load(labels_file)
    train_video_names = labels.videoname.unique()
    
    for video_file in all_video_files:
        if video_file[:-4] in train_video_names:
            video_names['train'].append(video_file[:-4])
        else:
            video_names['test'].append(video_file[:-4])

    # sort list of names
    video_names['train'].sort()
    video_names['test'].sort()

    return video_names

class HernitiaDataset(Dataset):
    """Hernitia dataset defined by annotation df."""

    def __init__(self, annotation_path, transform = None, test_mode = False):
        """
        Description
        -------------
        Creates dataset class for the training set.

        Parameters
        -------------
        annotation_path   : path to annotation df
        transform         : transforms to be applied to the frame (eg. data augmentation)
        test_mode         : boolean, if true there are no label in the annotation df and in the output of __getitem__

        Returns
        -------------
        Torch Dataset for the training set
        """
        self.annotation = pd.read_pickle(annotation_path)
        self.transform = transform
        self.test_mode = test_mode
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # recover frame info
        videoname = self.annotation.iloc[index]['videoname']
        frame_idx = self.annotation.iloc[index]['frame']
        if not self.test_mode: label = self.annotation.iloc[index]['label']
        # load frame
        frame_path = images_path + '/' + videoname + '/' + str(frame_idx) + '.jpg'
        frame = cv2.imread(frame_path)
        # transform
        if self.transform:
            frame = self.transform(frame)
        if self.test_mode: return frame
        else: return frame, label

def save_testing_df(dfs_path = dfs_path):
    """
    Description
    -------------
    Creates and saves testing dataframe ('videoname', 'frame') for kaggle prediction.

    Parameters
    -------------
    dfs_path      : path where all dataframes for training are stored
    """
    if os.path.exists(dfs_path + '/testing.pkl'): return 'testing dataframe already in storage'

    # list names of all videos in the test set
    surgeon1_test_videonames = ['RALIHR_surgeon01_fps01_' + str(x).zfill(4) for x in range(71,126)]
    surgeon2_test_videonames = ['RALIHR_surgeon02_fps01_' + str(x).zfill(4) for x in range(1,5)]
    surgeon3_test_videonames = ['RALIHR_surgeon03_fps01_0001']
    all_test_videonames = surgeon1_test_videonames + surgeon2_test_videonames + surgeon3_test_videonames

    # generate df with all frames of these videos
    videonames = []
    frames = []
    Ids = [] # id list for kaggle prediction
    for videoname in all_test_videonames:
        video_id = re.sub('[^0-9]', '',  videoname)
        video_id = video_id[0:2].zfill(3) + '-' + video_id[-4:] + '-'
        for frame in range(count_frames(videoname)):
            videonames.append(videoname)
            frames.append(frame)
            Ids.append(video_id + str(frame + 1).zfill(5))
    testing_df = pd.DataFrame({'videoname' : videonames, 'frame' : frames, 'Id': Ids})

    # save df
    testing_df.to_pickle(dfs_path + '/testing.pkl')

def predict_kaggle(model, model_name, transform, weights_path = weights_path, predictions_path = predictions_path, 
                kaggle_template_path = kaggle_template_path, batch_size = 128, predictions_name = 'kaggle_prediction'):
    """
    Description
    -------------
    Makes and save predictions on the testing set.

    Parameters
    -------------
    model               : model
    model_name          : name of the model from which to load the weights within weights/
    transform           : transforms to be applied to the frame (eg. data augmentation)
    weights_path        : path to model weights
    predictions_path    : path to make predictions to
    kaggle_template_path: path to the kaggle template for submissions
    batch_size          : batch size to use to make predictions
    predictions_name    : name of the csv file to which the predictions are saved
    """

    if os.path.exists(predictions_path + '/' + predictions_name + '.csv'): return 'predictions already exist under this file name'

    # create dictionary for all phases
    phase_to_label = {
    'adhesiolysis' : 0,
    'blurry' : 1,
    'catheter insertion' : 2,
    'mesh placement' : 3,
    'mesh positioning' : 4,
    'out of body' : 5,
    'peritoneal closure' : 6,
    'peritoneal scoring' : 7,
    'positioning suture' : 8,
    'preperioneal dissection' : 9,
    'primary hernia repair' : 10,
    'reduction of hernia' : 11,
    'stationary idle' : 12,
    'transitionary idle' : 13,
    }

    label_to_phase = {v: k for k, v in phase_to_label.items()}

    # load kaggle template
    kaggle_template_df = pd.read_csv(predictions_path + '/kaggle_template.csv')

    model.load_state_dict(torch.load(weights_path + '/' + model_name + '.pkl'))
    model.eval()

    # create pytorch dataset
    testing_dataset = HernitiaDataset(dfs_path + '/testing.pkl', transform, test_mode=True)

    # instantiate data loader
    testing_dataloader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)

    # load testing df
    testing_df = testing_dataset.annotation

    Id = testing_df['Id'].tolist()
    Predicted = []

    # iterate over testing data to make predictions
    for batch_idx, inputs in enumerate(Bar(testing_dataloader)):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        Predicted += preds.tolist()
    
    # save predictions
    Predicted = [label_to_phase[prediction] for prediction in Predicted]
    predictions_df = pd.DataFrame({'Id' : Id, 'Predicted' : Predicted})
    predictions_df = pd.merge(predictions_df, kaggle_template_df['Id'], how='inner', on=['Id'])[['Id', 'Predicted']]
    predictions_df.to_csv(predictions_path + '/' + predictions_name + '.csv', index = False)

def train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs):
    """
    Description
    -------------
    Train model, saves and returns one with best validation accuracy

    Parameters
    -------------
    model               : model
    model_name          : name of the model which will be the name of the saved weights file within weights/
    dataloaders         : dictionary of dataloaders (keys are 'training' and 'validation')
    criterion           : criterion
    optimizer           : optimizer
    scheduler           : scheduler
    num_epochs          : number of epochs
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in Bar(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'training':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save best model weights
    torch.save(model.state_dict(), weights_path + '/' + model_name + '.pkl')