
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import math


class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(74420, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

        

class ResidualBlock(nn.Module):
    def __init__(self, filters, input_channels, conv_shortcut = False,  kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.filters = filters
        self.input_channels = input_channels
        self.conv_shortcut = conv_shortcut
        if self.conv_shortcut:
            self.conv_sc = nn.Conv2d(in_channels=input_channels, out_channels= 4*filters, kernel_size=1, stride=stride)
            self.bn_sc = nn.BatchNorm2d(num_features=4*filters)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=filters, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=filters)
    
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=filters)
        
        #conv3 keeps image size
        self.conv3 = nn.Conv2d(in_channels=filters, out_channels=4 * filters, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_features=4*filters)
        
    def forward(self, x):
        if self.conv_shortcut:
            shortcut = self.bn_sc(self.conv_sc(x))
        else:
            shortcut = x
       # print("shortcut size:", shortcut.size())
        x = F.relu(self.bn1(self.conv1(x)))
       #print("after first convolution", x.size())
        padding = math.ceil(0.5 * (x.size()[2] * (self.stride - 1) + self.kernel_size - self.stride))
        #print(padding)
        pad = nn.ZeroPad2d(padding)
        x = pad(x)
        #print("after padding", x.size())
        x = F.relu(self.bn2(self.conv2(x)))
      #  print("after second convolution", x.size())
        x = self.bn3(self.conv3(x))
       # print("after third convolution", x.size())
        x = torch.add(x, shortcut)
        x = F.relu(x)
        return x
      


class ResNet(nn.Module):
    def __init__(self, depth=10, n_classes=3, input_channels=3, filters=32, input_size=250):
        super(ResNet, self).__init__()
        self.depth = depth
        self.input_channels = input_channels
        # residual blocks keep the channels with same size as input images
        blocks = []
        blocks.append(ResidualBlock(filters=filters, input_channels=input_channels, conv_shortcut=True))
        for i in range(depth - 1):
            blocks.append(ResidualBlock(filters=filters, input_channels=4*filters))

        self.blocks = nn.ModuleList(blocks)
        self.avg_pool = nn.AvgPool2d(kernel_size=input_size) #global average pooling
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(in_features=4*filters, out_features=n_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        return x