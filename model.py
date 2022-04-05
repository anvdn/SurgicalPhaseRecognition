import torch
import torch.nn as nn
from torchvision import models


class HernitiaModel(nn.Module):
    """ Hernitia model made of pretrained bottleneck + lstm. """

    def __init__(self, model_name, num_classes, pretrained = True, num_layers_lstm = 2, hidden_size_lstm = 64,  skip_lstm = False):
        """
        Description
        -------------
        Initialize Hernitia model

        Parameters
        -------------
        model_name             : string, name of the model
        num_classes            : int, number of classes
        pretrained             : boolean, whether the bottleneck is pretrained
        num_layers_lstm        : int, number of layers in lstm
        hidden_size_lstm       : int, hidden size of lstm
        skip_lstm              : whether to skip the lstm (for cnn finetuning)
        """
        super(HernitiaModel, self).__init__()

        self.model_name = model_name
        self.num_layers_lstm = num_layers_lstm
        self.hidden_size_lstm = hidden_size_lstm
        self.skip_lstm = skip_lstm
        # build model
        self.bottleneck = models.mobilenet_v2(pretrained=pretrained)
        self.bottleneck.classifier[1] = nn.Linear(self.bottleneck.classifier[1].in_features, hidden_size_lstm)
        self.lstm = nn.LSTM(input_size=hidden_size_lstm, hidden_size=hidden_size_lstm, num_layers=num_layers_lstm, batch_first = True)
        self.fc_finetuning = nn.Linear(hidden_size_lstm, num_classes)
        self.fc_lstm = nn.Linear(hidden_size_lstm, num_classes)

    def freeze_bottleneck(self):
        """ Freeze all parameters of bottleneck. """
        for param in self.bottleneck.parameters():
            param.requires_grad = False
        for param in self.fc_finetuning.parameters():
            param.requires_grad = False

    def unfreeze_bottleneck(self):
        """ Unfreeze all parameters of bottleneck. """
        for param in self.bottleneck.parameters():
            param.requires_grad = True
        for param in self.fc_finetuning.parameters():
            param.requires_grad = True

    def freeze_lstm(self):
        """ Freeze all parameters of LSTM. """
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.fc_lstm.parameters():
            param.requires_grad = False

    def unfreeze_lstm(self):
        """ Unfreeze all parameters of LSTM. """
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.fc_lstm.parameters():
            param.requires_grad = True

    def forward(self, input):
        """
        Description
        -------------
        Forward pass

        Parameters
        -------------
        input                : tensor of shape (temporal length, c, w, h) if skip_lstm is True else (batch_size, c, w, h)
        """
        x = self.bottleneck(input) 
        if not self.skip_lstm:
            print(x.shape)
            print(x.shape)
            out, hidden = self.lstm(x[None, :], None) # None is because it expects a batch dim
            x = nn.functional.relu(x)
            x = self.fc_lstm(x)     
        else:
            x = self.fc_finetuning(x)
        return x