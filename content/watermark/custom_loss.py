import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from .secret_matrix import get_secret_matrix



class watermarkCrossEntropyLoss(nn.Module):
    def __init__(self, type, size):
        super(watermarkCrossEntropyLoss, self).__init__()
        self.X = get_secret_matrix(type, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, weights, targets):
        # Compute the loss function
        weights = np.mean(weights, 0)
        x = np.dot(self.X, weights.flatten())
        x = self.sigmoid(torch.tensor(x))
        n = len(targets)
        loss = -(1/n)*torch.sum(targets * np.log(x) + (1-targets) * np.log(1-x))
        return loss

    def save(self, destination):
        np.save(destination, self.X)
        # pd.DataFrame(self.X).to_csv(destination, index=False)

