import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

# memorize all paths
videos_path = os.path.join(os.getcwd(), 'surgery.videos.hernitia')
csv_path = os.path.join(os.getcwd(), 'video.phase.trainingData.clean.StudentVersion.csv')
dfs_path = os.path.join(os.getcwd(), 'dfs')
labels_path = dfs_path + '/labels.pkl'
images_path = os.path.join(os.getcwd(), 'images')
weights_path = os.path.join(os.getcwd(), 'weights')
predictions_path = os.path.join(os.getcwd(), 'predictions')

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
    List of names of train and test videos

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
            Ids.append(video_id + str(frame).zfill(5))
    testing_df = pd.DataFrame({'videoname' : videonames, 'frame' : frames, 'Id': Ids})

    # save df
    testing_df.to_pickle(dfs_path + '/testing.pkl')