import torch
import torch.nn.functional as F
import numpy as np
from typing import List


class NeuralNetwork(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super(NeuralNetwork, self).__init__()
        self.hidden_layers = [torch.nn.Linear(feature_dim, hidden_dims[0])]
        self.hidden_layers.extend(
            [torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
        self.predict_layer = torch.nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = torch.nn.LeakyReLU(hidden_layer(x))
        x = self.predict_layer(x)
        return x


class SinFeatureNetwork(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super(SinFeatureNetwork, self).__init__()
        self.hidden_layers_sin = [torch.nn.Linear(feature_dim, hidden_dims[0])]
        self.hidden_layers_sin.extend(
            [torch.nn.Linear(hidden_dims[i]*2, hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
        self.hidden_layers_relu = [torch.nn.Linear(feature_dim, hidden_dims[0])]
        self.hidden_layers_relu.extend(
            [torch.nn.Linear(hidden_dims[i]*2, hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
        self.predict_layer = torch.nn.Linear(hidden_dims[-1]*2, output_dim)

    def forward(self, x):
        for hidden_sin, hidden_relu in zip(self.hidden_layers_sin, self.hidden_layers_relu):
            x_sin = torch.sin(hidden_sin(x))
            x_relu = torch.nn.LeakyReLU(hidden_relu(x))
            x = torch.cat((x_sin, x_relu), dim=1)
        x = self.predict_layer(x)
        return x
