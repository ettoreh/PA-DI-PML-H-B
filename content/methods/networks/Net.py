import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes, conv_size=[6, 16], kernel_size=5):
        super().__init__()
        
        flatten_size = conv_size[1] * kernel_size * kernel_size
        kernel_size = (kernel_size, kernel_size)
        
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=conv_size[0], 
            kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(
            in_channels=conv_size[0], 
            out_channels=conv_size[1], 
            kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2), 
            stride=(2, 2))
        self.fc1 = nn.Linear(
            in_features=flatten_size, 
            out_features=int(0.6*flatten_size))
        self.fc2 = nn.Linear(
            in_features=int(0.6*flatten_size), 
            out_features=int(0.36*flatten_size))
        self.fc3 = nn.Linear(
            in_features=int(0.36*flatten_size), 
            out_features=num_classes)
        
        self.layers = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'fc1': self.fc1,
            'fc2': self.fc2,
            'fc3': self.fc3
        }
        
        self.prune_layers = (
            (self.conv1, 'weight'),
            (self.conv2, 'weight'),
            (self.fc1, 'weight'),
            (self.fc2, 'weight'),
            (self.fc3, 'weight'),
        )
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    