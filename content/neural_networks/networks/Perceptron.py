import torch 
import torch.nn as nn
import torch.nn.functional as F

# Define the class for single layer NN
class SinglePerceptron(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(SinglePerceptron, self).__init__()
        # hidden layer 
        self.linear1= torch.nn.Linear(input_size, hidden_neurons)
        self.linear2 = torch.nn.Linear(hidden_neurons, output_size) 
        
    # prediction function
    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return x