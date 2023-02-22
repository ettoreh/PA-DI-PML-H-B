import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from .secret_matrix import get_secret_matrix



class watermarkCrossEntropyLoss(nn.Module):
    def __init__(self, type, size, X=None):
        super(watermarkCrossEntropyLoss, self).__init__()
        self.X = X
        if X is None:
            self.X = get_secret_matrix(type, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, weights, targets):
        # Compute the loss function
        weights = torch.mean(weights, 0)
        x = torch.matmul(torch.tensor(self.X).double(), weights.flatten().double())
        x = self.sigmoid(x)
        n = len(targets)
        loss = -(1/n)*torch.sum(targets * torch.log(x) + (1-targets) * torch.log(1-x))
        return loss

    def save(self, path):
        np.save(path, self.X)
        # pd.DataFrame(self.X).to_csv(destination, index=False)
        
    def load(self, path):
        self.X = np.laod(path)

