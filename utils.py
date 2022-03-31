import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

# memorize all paths
videos_path = os.path.join(os.getcwd(), 'surgery.videos.hernitia')
csv_path = os.path.join(os.getcwd(), 'video.phase.trainingData.clean.StudentVersion.csv')
labels_path = os.path.join(os.getcwd(), 'labels/labels.pkl')

# memorize number of classes
num_classes = pd.read_pickle(labels_path).label.unique().size

# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    labels = torch.tensor(all_labels.loc[all_labels['videoName'] == videoname]['label'].tolist())
    labels = F.one_hot(labels, num_classes=num_classes).to(device)
    return labels

def get_train_test_video_names(videos_path = videos_path, labels_path = labels_path):
    """
    Description
    -------------
    List of names of train and test videos

    Parameters
    -------------
    videos_path    : path to directory of videos
    labels_path    : path to labels file (created in the labeling notebook)

    Returns
    -------------
    (train_names, test_names) , each of the tuple elements is a list of strings
    """
    train_names, test_names = [], []

    # list all videos
    all_video_files = os.listdir(videos_path) 

    # get list of train video names
    with open(labels_path, 'rb') as labels_file:
        labels = pickle.load(labels_file)
    train_video_names = labels.videoName.unique()
    
    for video_file in all_video_files:
        if video_file[:-4] in train_video_names:
            train_names.append(video_file[:-4])
        else:
            test_names.append(video_file[:-4])

    # sort list of names
    train_names.sort()
    test_names.sort()

    return train_names, test_names

class HernitiaDataset(Dataset):
    """Hernitia dataset composed of a specified list of videonames."""

    def __init__(self, videonames, resize = (224,224), mean = None, std = None):
        """
        Description
        -------------
        Creates dataset class for the training set.

        Parameters
        -------------
        videonames     : list of names of the video to include in the dataset
        resize         : tuple, shape of the resized frame (default = (224,224))
        mean           : mean values to normalize the frame (ex = [0.485, 0.456, 0.406])
        std            : standard deviation values to normalize the frame (ex = [0.229, 0.224, 0.225])

        Returns
        -------------
        Torch Dataset for the training set
        """
        self.videonames = videonames
        self.resize = resize
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

    def __len__(self):
        return len(self.videonames)

    def __getitem__(self, index):
        videoname = self.videonames[index]
        frames = get_frames(videoname, self.resize, self.mean, self.std)
        labels = get_labels(videoname)
        return frames, labels