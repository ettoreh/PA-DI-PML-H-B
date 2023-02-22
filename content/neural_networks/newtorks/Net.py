import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes, conv_size=[6, 16], kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv_size[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv_size[0], conv_size[1], kernel_size)
        flatten_size = conv_size[1] * kernel_size * kernel_size
        self.fc1 = nn.Linear(flatten_size, int(0.6*flatten_size))
        self.fc2 = nn.Linear(int(0.6*flatten_size), int(0.36*flatten_size))
        self.fc3 = nn.Linear(int(0.36*flatten_size), num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    