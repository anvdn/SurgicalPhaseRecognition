"""
Implementation of ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_features, mid_features, enlargement = 1, big_stride = 2):
        super(ConvBlock, self).__init__()
        """
        in_features   : int, number of channels entering the block
        mid_features  : int, number of features produced by the first convolutional layer (two if there are 3 layers)
        enlargement   : int, multiplication factor from mid number of features to out number of features
        big_stride    : int, stride of the first convolutional layer of the block
        """

        self.num_layers = 3 if enlargement == 4 else 2
        out_features = enlargement * mid_features

        # set up kernel size and padding of first layer depending on number of layers (2 or 3)
        # padding is always equal to 1 whenever the kernel size is 3
        kernel_size = 3 if self.num_layers == 2 else 1
        padding = 1 if self.num_layers == 2 else 0

        # first convolution layer of the block, stride is big_stride
        self.conv1 = nn.Conv2d(in_channels = in_features, out_channels = mid_features, kernel_size = kernel_size, stride = big_stride, padding = padding, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = mid_features)
        
        if self.num_layers == 2:
            # second convolution layer of the block, stride is 1
            self.conv2 = nn.Conv2d(in_channels = mid_features, out_channels = out_features, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(num_features = out_features)

        elif self.num_layers == 3:
            # second convolution layer of the block, stride is 1
            self.conv2 = nn.Conv2d(in_channels = mid_features, out_channels = mid_features, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(num_features = mid_features)

            # third convolution layer of the block, stride is 1
            self.conv3 = nn.Conv2d(in_channels = mid_features, out_channels = out_features, kernel_size = 1, stride = 1, bias = False)
            self.bn3 = nn.BatchNorm2d(num_features = out_features)

        if big_stride == 1 and in_features == enlargement * mid_features:
            self.skip = nn.Sequential()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels = in_features, out_channels = out_features, kernel_size = 1, stride = big_stride, bias = False),
                nn.BatchNorm2d(num_features = out_features)
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        #print("dimension after first conv : ", y.size())
        if self.num_layers == 2:
            y = self.bn2(self.conv2(y))
            #print("dimension after second conv : ", y.size())
        elif self.num_layers == 3:
            y = F.relu(self.bn2(self.conv2(y)))
            #print("dimension after second conv : ", y.size())
            y = self.bn3(self.conv3(y))
            #print("dimension after third conv : ", y.size())
        #print("dimensions before add : ", y.size(), x.size())
        y += self.skip(x)
        out = F.relu(y)
        return out


class ResNet(nn.Module):

    def __init__(self, architecture, enlargement = 1, num_classes = 2):
        super(ResNet, self).__init__()
        """
        architecture  : list of integers, number of convolutional blocks for stage 2,3,4 and 5
        enlargement   : int, multiplication factor from mid number of features to out number of features
        num_classes   : int, number of classes for classification
        """
        # stage 1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = 64)
        
        self.in_features = 64

        self.stage2 = self.__create_layer(mid_features = 64, num_blocks = architecture[0], enlargement = enlargement, big_stride = 1)
        self.stage3 = self.__create_layer(mid_features = 128, num_blocks = architecture[1], enlargement = enlargement, big_stride = 2)
        self.stage4 = self.__create_layer(mid_features = 256, num_blocks = architecture[2], enlargement = enlargement, big_stride = 2)
        self.stage5 = self.__create_layer(mid_features = 512, num_blocks = architecture[3], enlargement = enlargement, big_stride = 2)
        self.linear = nn.Linear(in_features = enlargement * 512, out_features = num_classes)

    def __create_layer(self, mid_features, num_blocks, enlargement, big_stride):
        big_strides = [big_stride] + [1]*(num_blocks-1)
        layers = []
        for big_stride in big_strides:
            layers.append(ConvBlock(in_features = self.in_features, mid_features = mid_features, enlargement = enlargement, big_stride = big_stride))
            self.in_features = enlargement * mid_features
        return nn.Sequential(*layers)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        #print("dimension after first stage : ", y.size())
        y = self.stage2(y)
        #print("dimension after second stage : ", y.size())
        y = self.stage3(y)
        #print("dimension after third stage : ", y.size())
        y = self.stage4(y)
        #print("dimension after fourth stage : ", y.size())
        y = self.stage5(y)
        #print("dimension after fifth stage : ", y.size())
        y = F.avg_pool2d(y, kernel_size = 4)
        #print("dimension after avg pool : ", y.size())
        y = y.view(y.size(0), -1)
        out = self.linear(y)
        #print("out dim : ", out.size())
        return out

def ResNet18(num_classes = 2):
    return ResNet(architecture = [2, 2, 2, 2], enlargement = 1, num_classes = num_classes)

def ResNet34(num_classes = 2):
    return ResNet(architecture = [3, 4, 6, 3], enlargement = 1, num_classes = num_classes)

def ResNet50(num_classes = 2):
    return ResNet(architecture = [3, 4, 6, 3], enlargement = 4, num_classes = num_classes)