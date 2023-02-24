import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from .secret_matrix import get_secret_matrix



class WatermarkCrossEntropyLoss(nn.Module):
    def __init__(self, type, size, X=None, device='cpu'):
        super(WatermarkCrossEntropyLoss, self).__init__()
        self.X = X
        if X is None:
            self.X = get_secret_matrix(type, size)
        self.X = torch.tensor(self.X).to(torch.float32).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, weights, targets):
        # Compute the loss function
        weights = torch.mean(weights, 0)
        x = torch.matmul(self.X, weights.flatten())
        x = self.sigmoid(x)
        n = len(targets)
        loss = -(1/n)*torch.sum(targets * torch.log(x) + (1-targets) * torch.log(1-x))
        return loss

    def save(self, path):
        np.save(path, self.X)
        # pd.DataFrame(self.X).to_csv(destination, index=False)
        
    def load(self, path):
        self.X = np.laod(path)

