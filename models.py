import torch
import torch.nn as nn
from torchvision import models

class Identity(nn.Module):
    def __init__(self, out_features):
        super(Identity, self).__init__()
        self.out_features = out_features
        
    def forward(self, x):
        return x

class MobileNet(nn.Module):
    """ Pretrained mobilenet_v2. """

    def __init__(self, model_name, num_classes, pretrained):
        """
        Description
        -------------
        Initialize model

        Parameters
        -------------
        model_name             : string, name of the model
        num_classes            : int, number of classes
        pretrained             : boolean, whether the backbone is pretrained
        """
        super(MobileNet, self).__init__()

        self.model_name = model_name
        # build model
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, input):
        """
        Description
        -------------
        Forward pass

        Parameters
        -------------
        input                : tensor of shape (batch_size, c, w, h)
        """
        x = self.backbone(input) 
        
        return x


class MobileNetLSTM(nn.Module):
    """ Model made of pretrained backbone mobilenet_v2 + lstm. """

    def __init__(self, model_name, num_classes, pretrained, num_layers_lstm = 1, bidirectional = False, hidden_size_lstm = 32, skip_lstm = False):
        """
        Description
        -------------
        Initialize model

        Parameters
        -------------
        model_name             : string, name of the model
        num_classes            : int, number of classes
        pretrained             : boolean, whether the backbone is pretrained
        num_layers_lstm        : int, number of layers in lstm
        bidirectional          : boolean, whether to make the lstm bidirectional
        hidden_size_lstm       : int, hidden size of lstm
        skip_lstm              : whether to skip the lstm (for cnn finetuning)
        """
        super(MobileNetLSTM, self).__init__()

        self.model_name = model_name
        self.num_layers_lstm = num_layers_lstm
        self.hidden_size_lstm = hidden_size_lstm
        self.skip_lstm = skip_lstm
        # build model
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier[1] = Identity(self.backbone.classifier[1].in_features) # convert classifier to identity
        self.lstm = nn.LSTM(input_size=self.backbone.classifier[1].out_features, hidden_size=hidden_size_lstm, num_layers=num_layers_lstm, batch_first = True, bidirectional=bidirectional)
        self.fc_finetuning = nn.Linear(self.backbone.classifier[1].out_features, num_classes)
        self.fc_lstm = nn.Linear(hidden_size_lstm + bidirectional * hidden_size_lstm, num_classes) # output size is twice as large if lstm is bidirectional

    def freeze_backbone(self):
        """ Freeze all parameters of backbone. """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.fc_finetuning.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """ Unfreeze all parameters of backbone. """
        for param in self.backbone.parameters():
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
        x = self.backbone(input) 
        if not self.skip_lstm:
            out, hidden = self.lstm(x[None, :], None) # None is because it expects a batch dim
            x = nn.functional.relu(out[0])
            x = self.fc_lstm(x)     
        else:
            x = self.fc_finetuning(x)

        return x


class MobileNetFC(nn.Module):
    """ Model made of pretrained backbone mobilenet_v2 + fc for temporal treatment. """

    def __init__(self, model_name, num_classes, pretrained, skip_temp_fc):
        """
        Description
        -------------
        Initialize model

        Parameters
        -------------
        model_name             : string, name of the model
        num_classes            : int, number of classes
        pretrained             : boolean, whether the backbone is pretrained
        skip_temp_fc           : whether to skip the fc for temporal treatment (for cnn finetuning)
        """
        super(MobileNetFC, self).__init__()

        self.model_name = model_name
        self.skip_temp_fc = skip_temp_fc
        # build model
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier[1] = Identity(self.backbone.classifier[1].in_features)
        self.fc = nn.Linear(self.backbone.classifier[1].out_features, num_classes)
        self.temp_fc = nn.Linear(num_classes, num_classes)

    def freeze_backbone(self):
        """ Freeze all parameters of backbone. """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """ Unfreeze all parameters of backbone. """
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def freeze_fc(self):
        """ Freeze all parameters of backbone. """
        for param in self.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        """ Unfreeze all parameters of backbone. """
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_temp_fc(self):
        """ Freeze all parameters of temporal FC. """
        for param in self.temp_fc.parameters():
            param.requires_grad = False

    def unfreeze_temp_fc(self):
        """ Unfreeze all parameters of temporal FC. """
        for param in self.temp_fc.parameters():
            param.requires_grad = True

    def forward(self, input):
        """
        Description
        -------------
        Forward pass

        Parameters
        -------------
        input                : tensor of shape (temporal length, c, w, h) if skip_temp_fc is True else (batch_size, c, w, h)
        """
        x = self.backbone(input)
        x = self.fc(x)
        if not self.skip_temp_fc:
            # dim of x is (temporal length, num_classes)
            x = x[None, :]
            # add one batch size dimension
            x = self.temp_fc(x)  
            # remove batch dim
            x = x[0]


        return x

class MobileNetStage(nn.Module):
    """ Model made of backbone mobilenet_v2 + parallel fc for stage treatment. """

    def __init__(self, model_name, num_classes, pretrained, num_stages):
        """
        Description
        -------------
        Initialize Hernitia model

        Parameters
        -------------
        model_name             : string, name of the model
        num_classes            : int, number of classes
        pretrained             : boolean, whether the backbone is pretrained
        num_stages             : int, number of stages for the operation to consider (e.g. if 20 stages, the operation duration is divided into 20 intervals)
        """
        super(MobileNetStage, self).__init__()

        self.model_name = model_name
        # build model
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier[1] = Identity(self.backbone.classifier[1].in_features)
        self.fc11 = nn.Linear(self.backbone.classifier[1].out_features, 32)
        self.fc12 = nn.Linear(num_stages, 32)
        self.fc2 = nn.Linear(32 + 32, num_classes)

    def freeze_backbone(self):
        """ Freeze all parameters of backbone. """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """ Unfreeze all parameters of backbone. """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_fc11(self):
        """ Freeze all parameters of fc11. """
        for param in self.fc11.parameters():
            param.requires_grad = False

    def unfreeze_fc11(self):
        """ Unfreeze all parameters of fc11. """
        for param in self.fc11.parameters():
            param.requires_grad = True

    def freeze_fc12(self):
        """ Freeze all parameters of fc12. """
        for param in self.fc12.parameters():
            param.requires_grad = False

    def unfreeze_fc12(self):
        """ Unfreeze all parameters of fc12. """
        for param in self.fc12.parameters():
            param.requires_grad = True

    def freeze_fc2(self):
        """ Freeze all parameters of fc2. """
        for param in self.fc2.parameters():
            param.requires_grad = False

    def unfreeze_fc2(self):
        """ Unfreeze all parameters of fc2. """
        for param in self.fc2.parameters():
            param.requires_grad = True

    def forward(self, input):
        """
        Description
        -------------
        Forward pass

        Parameters
        -------------
        input                : tuple (tensor : (batch_size, c, w, h), tensor (batch_size, 50))
        """
        # frames treatment
        x11 = self.backbone(input[0])
        x11 = self.fc11(x11)
        x11 = torch.nn.ReLU()(x11)

        # ratio treatment
        x12 = self.fc12(input[1])
        x12 = torch.nn.ReLU()(x12)

        # concat
        x2 =  torch.cat([x11, x12], dim=1)

        # through last linear layer
        x = self.fc2(x2)

        return x

