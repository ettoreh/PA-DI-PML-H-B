import torch 
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        x = F.elu(self.map3(x))
        return x
    
class Extractor(nn.Module):
    def __init__(self, kfe, kme):
        super(Extractor, self).__init__()
        self.kfe = kfe
        self.kme = kme
        
    def forward(self, x):
        x = torch.mean(x, 0)
        if self.kfe is not None:
            x = x[self.kfe[0], self.kfe[1]]
        x = torch.matmul(self.kme, x.flatten())
        x = F.sigmoid(x)
        return x 