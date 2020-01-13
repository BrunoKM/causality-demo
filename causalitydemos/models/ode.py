import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 2),
        )

    def forward(self, t, y):
        return self.net(y)