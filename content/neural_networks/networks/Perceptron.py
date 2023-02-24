import torch.nn as nn
import torch.nn.functional as F



class SinglePerceptron(nn.Module):    
    def __init__(self, input_size, hidden_neurons, output_size):
        super(SinglePerceptron, self).__init__()
        self.linear1= nn.Linear(input_size, hidden_neurons)
        self.linear2 = nn.Linear(hidden_neurons, output_size) 
       
    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return x
