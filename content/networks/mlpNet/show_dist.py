import torch
import torch.nn as nn

from matplotlib import pyplot as plt

from Net import Net



if __name__ == "__main__":
    
    model = Net(10)
    model_path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/models/cifar_net_20230216_003445'
    model.load_state_dict(torch.load(model_path))
    print('model loaded')
    
    print(model.conv1.weight.detach().numpy())