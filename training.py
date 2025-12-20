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


def train_on_value_dataset(net: nn.Module, dataset: TrainingValueDataset, optimizer: optim.Optimizer, batch_size: int = 50, n_epochs: int = 10, device="cpu") -> float:
    criterion = nn.MSELoss()
    net.to(device)

    # Check if dataset is already on target device
    dataset_on_gpu = (hasattr(dataset, '_inputs') and dataset._inputs is not None and
                      dataset._inputs.device.type == 'cuda')
    use_pin_memory = (device != "cpu" and not dataset_on_gpu)

    # Initial evaluation: loss before training
    net.eval()
    initial_loss = 0.0
    with torch.no_grad():
        # Use pin_memory only for CPU datasets going to GPU
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                pin_memory=use_pin_memory, num_workers=0)
        losses = []
        for inputs, targets in dataloader:
            # If dataset is already on GPU, these are no-ops
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
        initial_loss = sum(losses) / len(losses)

    # Training
    net.train()
    for epoch in range(n_epochs):
        # Use pin_memory only for CPU datasets going to GPU
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                pin_memory=use_pin_memory, num_workers=0)
        losses_epoch = []
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
        print(
            f"Epoch {epoch + 1}/{n_epochs}, Loss: {sum(losses_epoch) / len(losses_epoch)}")

    return initial_loss
