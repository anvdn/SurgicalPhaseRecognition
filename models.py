import torch.nn as nn
from torchvision import models


class HernitiaModel(nn.Module):
    """ Hernitia model made of pretrained bottleneck + lstm. """

    def __init__(self, model_name, num_classes, pretrained = True, skip_lstm = False):
        """
        Description
        -------------
        Initialize Hernitia model

        Parameters
        -------------
        model_name             : string, name of the model
        num_classes            : int, number of classes
        pretrained             : boolean, whether the bottleneck is pretrained
        skip_lstm              : whether to skip the lstm (for cnn pretraining)
        """
        super(HernitiaModel, self).__init__()

        self.model_name = model_name
        self.skip_lstm = skip_lstm
        # build model
        self.bottleneck = models.mobilenet_v2(pretrained=pretrained)
        self.bottleneck.classifier[1] = nn.Linear(self.bottleneck.classifier[1].in_features, num_classes)
        self.lstm = nn.LSTM(input_size=num_classes, hidden_size=64, num_layers=3)
        self.fc = nn.Linear(64, num_classes)

    def freeze_bottleneck(self):
        """ Freeze all parameters of bottleneck. """
        for param in self.bottleneck.parameters():
            param.requires_grad = False

    def unfreeze_bottleneck(self):
        """ Unfreeze all parameters of bottleneck. """
        for param in self.bottleneck.parameters():
            param.requires_grad = True

    def freeze_lstm(self):
        """ Freeze all parameters of LSTM. """
        for param in self.lstm.parameters():
            param.requires_grad = False

    def unfreeze_lstm(self):
        """ Unfreeze all parameters of LSTM. """
        for param in self.lstm.parameters():
            param.requires_grad = True

    def switch_mode(self):
        """ Switch from CNN mode to CNN-LSTM mode. """
        self.skip_lstm = not self.skip_lstm

    def forward(self, input):
        """
        Description
        -------------
        Forward pass

        Parameters
        -------------
        input                : tensor of shape (batch_size, temporal length, c, w, h) if skip_lstm is True else (batch_size, c, w, h)
        """
        hidden = None
        if not self.skip_lstm:
            for t in range(input.size(1)):
                x = self.bottleneck(input[:, t, :, :, :])  
                out, hidden = self.lstm(x.unsqueeze(0), hidden) 
            x = nn.functional.relu(x)        
            x = self.fc(out[-1, :, :])
        else:
            x = self.bottleneck(input) 
        return x