import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from typing import List, Dict, Tuple, Optional
from cube import Cube
from cube_nn import CubeToTensor
from cube_datasets import TrainingValueDataset
# from evaluation import Evaluator, full_evaluation
from solvers import nn_solver
from tqdm import tqdm

import os
import time


def train_on_value_dataset(net: nn.Module, dataset: TrainingValueDataset, optimizer: optim.Optimizer, batch_size: int = 50, n_epochs: int = 10) -> float:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()

    # Initial evaluation: loss before training
    net.eval()
    initial_loss = 0.0
    with torch.no_grad():
        losses = []
        for inputs, targets in dataloader:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
        initial_loss = sum(losses) / len(losses)

    net.train()
    for epoch in range(n_epochs):
        losses_epoch = []
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

    return initial_loss
