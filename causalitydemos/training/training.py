import math
import os
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Trainer(object):
    def __init__(self, model: torch.nn.Module,
                 criterion,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 optimizer,
                 scheduler=None,
                 batch_size: int = 64,
                 device=None,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 log_interval: int = 100) -> None:
        assert isinstance(model, nn.Module)
        assert isinstance(train_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)

        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pin_memory = pin_memory
        self.log_interval = log_interval

        self.trainloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory)
        self.testloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory)

        # Lists for storing training/test metrics
        self.train_loss, self.test_loss = [], []
        self.steps = 0
        self.epochs = 0

    def train(self, n_epochs: int) -> None:
        for epoch in range(self.epochs, self.epochs + n_epochs):
            # Train
            start_time = time.time()
            self._train_single_epoch()

            self.test()

            mean_train_loss = np.mean(self.train_loss[-math.floor(len(self.trainloader) / self.log_interval):])
            learning_rate_str = f"\tLR: {np.round(self.scheduler.get_lr(), 4)}" if self.scheduler else ""
            print(f"Epoch {self.epochs}:\tTest Loss: {np.round(self.test_loss[-1], 6)};\t"
                  f"Train Loss: {np.round(mean_train_loss, 6)};\t"
                  f"Time per epoch: {time.time() - start_time}s" + learning_rate_str)

            if self.scheduler:
                self.scheduler.step()

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        for i, data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, targets = data
            if self.device is not None:
                # Move data to adequate device
                inputs, targets = map(lambda x: x.to(self.device, non_blocking=self.pin_memory), (inputs, targets))
            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)  # No NaN loss
            loss.backward()
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                self.train_loss.append(loss.item())
        self.epochs += 1

    def test(self):
        """
        Single evaluation on the entire provided test dataset.
        Return accuracy, mean test loss, and an array of predicted probabilities
        """
        test_loss = 0.

        # Set model in eval mode
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                # Get inputs
                inputs, labels = data
                if self.device is not None:
                    inputs, labels = map(lambda x: x.to(self.device),
                                         (inputs, labels))
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels).item()
        test_loss = test_loss / len(self.testloader)

        # Log statistics
        self.test_loss.append(test_loss)
