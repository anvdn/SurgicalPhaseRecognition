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
if os.path.exists(labels_path): num_classes = pd.read_pickle(labels_path).label.unique().size

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
        return 'no video under this name'
    totalframecount= int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return totalframecount

def save_frames(videoname, resize = (224,224)):
    """
    Description
    -------------
    Resize and save jpg frames of a video

    Parameters
    -------------
    videoname   : name of the video file without the extension (.mp4 or .mov)
    resize      : tuple, shape of the resized frame e.g. (224,224)
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
        # extract next frame
        success, image = video.read()
    video.release()
    # print videoname
    print(videoname)

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
    (train_names, test_names) , each of the tuple elements is a list of strings corresponding to video names
    """
    video_names = {'train': [], 'test': []}

    # list all videos
    all_video_files = os.listdir(videos_path) 

    # get list of train video names
    with open(labels_path, 'rb') as labels_file:
        labels = pickle.load(labels_file)
    train_video_names = labels.videoname.unique()
    
    for video_file in all_video_files:
        # we omit the 4 last characters of the file names corresponding to the extension
        if video_file[:-4] in train_video_names:
            video_names['train'].append(video_file[:-4])
        else:
            video_names['test'].append(video_file[:-4])

    # sort list of names
    video_names['train'].sort()
    video_names['test'].sort()

    return video_names

class HerniaDataset(Dataset):
    """Hernia dataset defined by annotation df."""

    def __init__(self, annotation_path, is_stage_feature = False, num_stages = 20, transform = None, 
                    test_mode = False, frame_idx_black_pad = 10000):
        """
        Description
        -------------
        Creates dataset class for the training set.

        Parameters
        -------------
        annotation_path         : path to annotation df
        is_stage_feature        : boolean, whether to add one hot encoded stage of the operation as extra feature (e.g. frame is between 10 and 15% of the operation)
        num_stages              : int, number of stages for the operation to consider (e.g. if 20 stages, the operation duration is divided into 20 intervals)
        transform               : transforms to be applied to the frame (eg. data augmentation)
        test_mode               : boolean, if true there are no label in the annotation df and in the output of __getitem__
        frame_idx_black_pad     : int, frame index in dataframes for which there is no image (padding),the frame output should be black

        Returns
        -------------
        Torch Dataset for the training set
        """
        self.annotation = pd.read_pickle(annotation_path)
        self.is_stage_feature = is_stage_feature
        self.num_stages = num_stages
        self.transform = transform
        self.test_mode = test_mode
        self.num_classes = num_classes
        self.frame_idx_black_pad = frame_idx_black_pad

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # recover frame info
        videoname = self.annotation.iloc[index]['videoname']
        frame_idx = self.annotation.iloc[index]['frame']
        video_num_frames = self.annotation.iloc[index]['video_num_frames']

        # recover label of frame if not in test mode
        if not self.test_mode: label = self.annotation.iloc[index]['label']

        # recover frame, or create empty black frame if it does not exist
        if frame_idx == self.frame_idx_black_pad:
                # frame should be black
                frame = torch.zeros((3,224,224))
        else:
            # load frame
            frame_path = images_path + '/' + videoname + '/' + str(frame_idx) + '.jpg'
            frame = cv2.imread(frame_path)
        # transform
            if self.transform:
                frame = self.transform(frame)

        # treat cases if or if not we should add the one hot encoded stage as extra feature
        if self.is_stage_feature:
            one_hot_vec = np.zeros((self.num_stages), dtype = np.float32)
            # add 1 value at index corresponding to stage if it is a true frame
            if frame_idx != self.frame_idx_black_pad: 
                index_one = int(self.num_stages * frame_idx / video_num_frames)
                one_hot_vec[index_one] = 1
            # return features, and label if not in test mode
            if self.test_mode: return frame, one_hot_vec
            else: return (frame, one_hot_vec), label

        else:
            # return features, and label if not in test mode
            if self.test_mode: return frame
            else: return frame, label

def predict_kaggle(model, model_name, is_stage_feature, num_stages, transform = None, weights_path = weights_path, 
        predictions_path = predictions_path, kaggle_template_path = kaggle_template_path, batch_size = 64, predictions_name = 'kaggle_prediction'):
    """
    Description
    -------------
    Makes and save predictions on the testing set.

    Parameters
    -------------
    model                   : model
    model_name              : name of the model from which to load the weights within weights/
    is_stage_feature        : boolean, whether to add one hot encoded stage of the operation as extra feature (e.g. frame is between 10 and 15% of the operation)
    num_stages              : int, number of stages for the operation to consider (e.g. if 20 stages, the operation duration is divided into 20 intervals)
    transform               : transforms to be applied to the frame (eg. data augmentation)
    weights_path            : path to model weights (folder)
    predictions_path        : path to make predictions to
    kaggle_template_path    : path to the kaggle template for submissions
    batch_size              : batch size to use to make predictions
    predictions_name        : name of the csv file to which the predictions are saved
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
    testing_dataset = HerniaDataset(dfs_path + '/testing.pkl', is_stage_feature, num_stages, transform, test_mode=True)

    # instantiate data loader
    testing_dataloader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)

    # load testing df
    testing_df = testing_dataset.annotation

    Id = testing_df['Id'].tolist()
    Predicted = []

    # iterate over testing data to make predictions
    for inputs in Bar(testing_dataloader):
        if isinstance(inputs , list):
            inputs = [input.to(device) for input in inputs]
        else:
            inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        Predicted += preds.tolist()
        
    
    # build predictions
    Predicted = [label_to_phase[prediction] for prediction in Predicted]
    predictions_df = pd.DataFrame({'Id' : Id, 'Predicted' : Predicted})
    # only keep rows in the template
    predictions_df = pd.merge(predictions_df, kaggle_template_df['Id'], how='inner', on=['Id'])[['Id', 'Predicted']]
    # sort rows of predictions in logical order
    predictions_df['sort'] = (predictions_df['Id'].str[:-11] + predictions_df['Id'].str[-10:-6] + predictions_df['Id'].str[-5:]).astype(int)
    predictions_df.sort_values(['sort'],inplace=True, ascending=True)
    predictions_df = predictions_df.drop(['sort'], axis=1)
    # save predictions to csv
    predictions_df.to_csv(predictions_path + '/' + predictions_name + '.csv', index = False)

def train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs, validation = True):
    """
    Description
    -------------
    Train model and saves the one with best validation accuracy if we are using validation or the one of the last epoch otherwise

    Parameters
    -------------
    model               : model
    model_name          : name of the model which will be the name of the saved weights file within weights/
    dataloaders         : dictionary of dataloaders (keys are 'training'/'validation' or just 'training' if we use the whole data)
    criterion           : criterion
    optimizer           : optimizer
    scheduler           : scheduler
    num_epochs          : number of epochs
    validation          : whether to use validation, also used as a criteria to save the best model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            # break if in validation phase and validation is set to False
            if phase == 'validation' and validation == False: break
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_true_examples = 0 # length of the dataset without the padded white images

            # iterate over data
            for inputs, labels in Bar(dataloaders[phase]):
                if isinstance(inputs , list):
                    inputs = [input.to(device) for input in inputs]
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)

                idx_true_images = (labels != -1)
                num_true_examples += idx_true_images.sum().item()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs[idx_true_images], labels[idx_true_images]) # don't take into account padded white images

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * idx_true_images.sum().item()
                running_corrects += torch.sum(preds[idx_true_images] == labels.data[idx_true_images])
            if phase == 'training':
                scheduler.step()

            epoch_loss = running_loss / num_true_examples
            epoch_acc = running_corrects.double() / num_true_examples

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # reload best model according to val acc if we are using validation data
    if validation == True: 
        print(f'Best val Acc: {best_acc:4f}')
        # load best model weights
        model.load_state_dict(best_model_wts)

    # save best model weights
    torch.save(model.state_dict(), weights_path + '/' + model_name + '.pkl')


def evaluate_model(model, dataloaders, criterion):
    """
    Description
    -------------
    Compute validation loss and accuracy

    Parameters
    -------------
    model               : model
    dataloaders         : dictionary of dataloaders with a key 'validation' for the validation data
    criterion           : criterion
    """
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    num_true_examples = 0 # length of the dataset without the padded white images

    # iterate over data
    for inputs, labels in Bar(dataloaders['validation']):
        if isinstance(inputs , list):
            inputs = [input.to(device) for input in inputs]
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        # compute indexes and number of true frames
        idx_true_images = (labels != -1)
        num_true_examples += idx_true_images.sum().item()

        # forward pass
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs[idx_true_images], labels[idx_true_images]) # don't take into account padded black images

        # compute running loss and correct predictions
        running_loss += loss.item() * idx_true_images.sum().item()
        running_corrects += torch.sum(preds[idx_true_images] == labels.data[idx_true_images])

    # compute epoch loss and accuracy
    epoch_loss = running_loss / num_true_examples
    epoch_acc = running_corrects.double() / num_true_examples

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


def smooth_predictions(predictions_name, window_size = 11):
    """
    Description
    -------------
    Smoothes labels of prediction.

    Parameters
    -------------
    predictions_name    : name of predictions within predictions/
    window_size         : size of window of surrouding frames/ labels to consider, the label is replaced if a label accounts for strictly more than half of them
    """
    # read original predictions file
    preds = pd.read_csv(predictions_path + '/' + predictions_name + '.csv')
    
    labels = preds['Predicted'].tolist()
    
    # compute number of labels to consider before/after the label considered
    half_window_size = window_size // 2

    for index in range(half_window_size, len(labels) - half_window_size):
        surrounding_labels = labels[index-half_window_size:index+half_window_size+1]
        # compute frequencies of labels inside the window
        freqs = dict()
        for label in surrounding_labels:
            freqs[label] = freqs.get(label, 0) + 1
        # replace if there is a majority label
        for key in freqs.keys():
            if freqs[key] > half_window_size:
                labels[index] = key
            
    preds['Predicted'] = labels
    # save the smoothed predictions
    preds.to_csv(predictions_path + '/' + predictions_name + '_smoothed.csv', index = False)
