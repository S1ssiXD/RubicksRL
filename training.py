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

# # Train network
# def train_network(network: nn.Module, cube_transform: CubeToTensor, output_transform: OutputToTensor, datasets: List[CubeDataset], batch_size: int = 50, epochs: int = 50, eval: int = 10, data_size: Optional[int] = None, evaluator: Evaluator = full_evaluation) -> Tuple[List, List]:
#     evals = []
#     losses = []

#     solver = nn_solver(network, cube_transform)
#     # evals.append(evaluator(solver))

#     training_datasets = [TrainingDataset(
#         dataset, cube_transform, output_transform) for dataset in datasets]
#     samplers = [RandomSampler(
#         dataset, replacement=True, num_samples=data_size) for dataset in training_datasets]
#     dataloaders = [DataLoader(
#         dataset, batch_size=batch_size, sampler=sampler) for dataset, sampler in zip(training_datasets, samplers)]

#     optimizer = optim.Adam(network.parameters(), lr=0.001)
#     criterion = nn.MSELoss()

#     print("Training network...")
#     for epoch in tqdm(range(epochs)):
#         network.train()
#         losses_epoch = []

#         for dataloader in dataloaders:
#             loss_dataset = []
#             for inputs, targets in dataloader:
#                 optimizer.zero_grad()
#                 outputs = network(inputs)
#                 loss = criterion(outputs, targets)
#                 loss_dataset.append(loss.item())
#                 loss.backward()
#                 optimizer.step()
#             loss_avg = sum(loss_dataset) / len(loss_dataset)
#             losses_epoch.append(loss_avg)
#         losses.append(losses_epoch)

#         if (epoch + 1) % eval == 0:
#             network.eval()
#             evals.append(evaluator(solver))
#             print(f"Epoch {epoch + 1}/{epochs}, Evaluation: {evals[-1]}")
#     print("Training finished.")
#     return evals, losses


# def train_on_unsolved(network: nn.Module, cube_transform: CubeToTensor, output_transform: OutputToTensor, batch_size: int = 50, epochs: int = 10, iterations: int = 10, dataset_size: int = 1000, evaluator: Evaluator = full_evaluation) -> Tuple[List, List]:
#     evals = []
#     losses = []
#     solver = nn_solver(network, cube_transform, max_moves=20)
#     full_dataset: CubeDataset = []

#     datasets_path = f"datasets/{time.strftime("%Y%m%d-%H%M%S")}/"
#     # Create directory for datasets if it doesn't exist
#     os.makedirs(datasets_path, exist_ok=True)

#     for i in range(iterations):
#         # generate dataset
#         new_dataset: CubeDataset = []
#         added = set()
#         # solved = set()

#         pbar = tqdm(total=dataset_size,
#                     desc=f"Iteration {i + 1}/{iterations}, dataset generation: ")
#         while len(new_dataset) < dataset_size:
#             pbar.n = len(new_dataset)
#             pbar.refresh()
#             cube = Cube()
#             sequence = cube.scramble(20)

#             new_cube = Cube()
#             j = 0
#             for move in sequence:
#                 new_cube.move(move)
#                 j += 1

#                 if new_cube in added:  # or new_cube in solved:
#                     continue

#                 new_dataset.append((new_cube.copy(), j))
#                 added.add(new_cube.copy())

#                 # sol = nn_solver(network, cube_transform, max_moves=j)(cube)
#                 # if sol is None:
#                 #     new_dataset.append((new_cube.copy(), j))
#                 #     added.add(new_cube.copy())
#                 # else:
#                 #     solved.add(new_cube.copy())
#                 #     if len(sol) < j:
#                 #         j = len(sol)
#         pbar.close()

#         full_dataset.extend(new_dataset)
#         save_dataset(new_dataset, f"{datasets_path}dataset_{i + 1}.pt")

#         training_new_dataset = TrainingDataset(
#             new_dataset, cube_transform, output_transform)
#         dataloader = DataLoader(
#             training_new_dataset, batch_size=batch_size, shuffle=True)

#         training_full_dataset = TrainingDataset(
#             full_dataset, cube_transform, output_transform)
#         # sampler = RandomSampler(
#         #     training_full_dataset, replacement=True, num_samples=dataset_size)
#         # dataloader_full = DataLoader(
#         #     training_full_dataset, batch_size=batch_size, sampler=sampler)
#         dataloader_full = DataLoader(
#             training_full_dataset, batch_size=batch_size, shuffle=True)

#         optimizer = optim.Adam(network.parameters(), lr=0.001)
#         criterion = nn.MSELoss()

#         network.train()
#         for epoch in tqdm(range(epochs), desc=f"Iteration {i + 1}/{iterations}, training on new dataset: "):
#             losses_epoch = []

#             for inputs, targets in dataloader:
#                 optimizer.zero_grad()
#                 outputs = network(inputs)
#                 loss = criterion(outputs, targets)
#                 losses_epoch.append(loss.item())
#                 loss.backward()
#                 optimizer.step()

#             for inputs, targets in dataloader_full:
#                 optimizer.zero_grad()
#                 outputs = network(inputs)
#                 loss = criterion(outputs, targets)
#                 losses_epoch.append(loss.item())
#                 loss.backward()
#                 optimizer.step()

#             losses.append(sum(losses_epoch) / len(losses_epoch))

#         network.eval()
#         evals.append(evaluator(solver))

#     return evals, losses
