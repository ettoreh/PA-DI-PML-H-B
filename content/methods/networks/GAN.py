import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class RIGA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kfe, kme):
        self.generator = Generator(input_size, hidden_size, output_size)
        self.discriminator = Discriminator(input_size, hidden_size, output_size)
        self.extractor = Extractor(kfe, kme)



class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.selu(self.map1(x))
        x = F.selu(self.map2(x))
        x = F.selu(self.map3(x))
        return x
        
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        x = F.elu(self.map3(x))
        x = F.sigmoid(x)
        return x
    
class Extractor(nn.Module):
    def __init__(self, proportion, input_size, hidden_size, output_size):
        super(Extractor, self).__init__()
        size = int(proportion*input_size)
        good_shape = True
        while good_shape:
            self.d = np.random.choice([0,1], size=input_size, p=[1-proportion, proportion])
            good_shape = ~(np.sum(self.d)==size)
        self.map1 = nn.Linear(size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.tensor([y for (y, Y) in zip(x, self.d) if Y!=0])
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        x = F.relu(self.map3(x))
        return x 
    