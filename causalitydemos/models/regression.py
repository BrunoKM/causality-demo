import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class NeuralNetwork(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: List[int], output_dim: int,
                 activation_function: torch.nn.Module = F.relu,
                 batch_norm: bool = False,
                 dropout_rate: float = 0.) -> None:
        super(NeuralNetwork, self).__init__()
        if dropout_rate > 1. or dropout_rate < 0.:
            raise ValueError(f'Dropout rate has to be between 0. and 1., but was {dropout_rate}')

        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.activation_func = activation_function
        # Add the first hidden layer
        self.hidden_layers = [torch.nn.Linear(feature_dim, hidden_dims[0])]
        # Add the remaining hidden layers
        self.hidden_layers.extend(
            [torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
        self.batch_norm_ops = [nn.BatchNorm1d(num_features=h_size) for h_size in hidden_dims]
        self.dropout_ops = [nn.Dropout(p=dropout_rate)]
        self.predict_layer = torch.nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_func(hidden_layer(x))
            if self.batch_norm:
                x = self.batch_norm_ops[i](x)
            if self.dropout_rate > 0.:
                x = self.dropout_ops[i](x)
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
