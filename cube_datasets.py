import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from cube import Cube, moves
from tqdm import tqdm
from torch.utils.data import Dataset
from protocols import CubeToTensor, ValueFunction
from cube_nn import CubeValueNN
import random


CubeDataset = Dict[Cube, float]
""" Type alias for a dataset containing Cube objects as keys and their corresponding values (e.g. number of moves needed to solve the cube)."""


def generate_full_optimal_value_dataset(n_moves: int = 5) -> CubeDataset:
    """
    Generate a dataset of all possible cubes reachable within n_moves from the solved state, with their optimal values.

    Args:
        n_moves (int): Maximum number of moves used to scramble the cubes.
    Returns:
        CubeDataset: the generated dataset.
    """
    cubes: CubeDataset = {}

    solved_cube = Cube()
    cubes[solved_cube] = 0
    current_layer = [solved_cube]

    for depth in range(n_moves):
        print(f"Generating cubes with {depth+1} moves...")
        next_layer = []
        for cube in current_layer:
            new_cubes = cube.get_all_neighbors()
            for new_cube in new_cubes:
                if new_cube not in cubes:
                    cubes[new_cube] = depth + 1
                    next_layer.append(new_cube)
        print(f"Added {len(next_layer)} new cubes.")
        current_layer = next_layer
    return cubes


def get_full_optimal_value_dataset(n_moves: int = 5) -> CubeDataset:
    """
    Get the full optimal value dataset, generating it if not already existing.

    Args:
        n_moves (int): Maximum number of moves used to scramble the cubes.
    Returns:
        CubeDataset: the generated dataset.
    """
    path = f"datasets/full/full_{n_moves}.pt"
    try:
        cubes = load_cube_dataset(path)
        print(f"Loaded full optimal value dataset from {path}.")
    except FileNotFoundError:
        print(
            f"Full optimal value dataset not found at {path}, generating it...")
        cubes = generate_full_optimal_value_dataset(n_moves=n_moves)
        save_cube_dataset(cubes, path)
        print(f"Saved full optimal value dataset to {path}.")
    return cubes


def save_cube_dataset(cubes: CubeDataset, path: str) -> None:
    """
    Save the dataset to a file.

    Args:
        cubes (CubeDataset): The dataset to save.
        path (str): The file path where the dataset will be saved.
    """
    with open(path, 'wb') as f:
        torch.save(cubes, f)


def load_cube_dataset(path: str) -> CubeDataset:
    """
    Load the dataset from a file.

    Args:
        path (str): The file path from which the dataset will be loaded.
    Returns:
        CubeDataset: The loaded dataset.
    """
    with open(path, 'rb') as f:
        cubes = torch.load(f, weights_only=False)
    return cubes


class TrainingValueDataset(Dataset):
    """Optimized dataset for training with pre-allocated tensors."""

    def __init__(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor):
        """Initialize dataset either from list of tuples or pre-allocated tensors.

        Args:
            inputs: Pre-allocated tensor of all inputs, shape (N, ...)
            targets: Pre-allocated tensor of all targets, shape (N, 1)
        """
        assert inputs.shape[0] == targets.shape[0], "Inputs and targets must have the same number of samples."
        self._inputs: torch.Tensor = inputs
        self._targets: torch.Tensor = targets

    @classmethod
    def create_from_trajectories(cls, cube_to_tensor: CubeToTensor, n_trajectories: int = 1000, n_moves: int = 20, seed: Optional[int] = None, device="cpu", tqdm_position: int = 0) -> "TrainingValueDataset":
        """Create dataset from random scramble trajectories.

        For each of N trajectories, generates a random scramble with M moves and adds
        all intermediate states (excluding the solved state). Duplicates are kept to
        naturally weight the dataset - states closer to solved appear less frequently
        in random scrambles, preserving this natural distribution.

        Args:
            cube_to_tensor: Function to convert cubes to tensors
            n_trajectories: Number of random trajectories to generate
            n_moves: Number of scramble moves per trajectory
            seed: Random seed for reproducibility
            device: Device to store tensors ('cpu' or 'cuda')
            tqdm_position: Position for tqdm progress bar (for parallel execution)

        Returns:
            TrainingValueDataset with all trajectory states (excluding solved state)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Get sample tensor to determine shape
        sample_cube = Cube()
        sample_tensor = cube_to_tensor(sample_cube)
        if sample_tensor.dim() > 1 and sample_tensor.shape[0] == 1:
            sample_tensor = sample_tensor.squeeze(0)
        tensor_shape = sample_tensor.shape

        # Pre-allocate for total size: n_trajectories * n_moves (no solved state)
        size = n_trajectories * n_moves
        inputs = torch.zeros((size, *tensor_shape), dtype=torch.float32)
        targets = torch.zeros((size, 1), dtype=torch.float32)

        idx = 0

        # Generate trajectories
        for traj_idx in tqdm(range(n_trajectories), desc="Generating trajectories", leave=False, position=tqdm_position):
            # Start from solved state
            cube = Cube()
            trajectory_cubes = []
            trajectory_distances = []

            # Use cube.generate_scramble_sequence to generate optimized random sequence (removes trivial cases)
            # We need to manually apply moves to capture trajectory
            move_sequence = Cube.generate_scramble_sequence(
                n_moves, seed=seed + traj_idx if seed is not None else None)

            # Apply moves and collect trajectory
            for move_idx, move in enumerate(move_sequence):
                cube.move(move)
                trajectory_cubes.append(cube.copy())
                trajectory_distances.append(move_idx + 1)

            # Batch convert cubes to tensors
            batch_tensor = cube_to_tensor(trajectory_cubes)

            # Handle tensor dimensions
            if batch_tensor.dim() == 1:
                # Single cube case
                batch_tensor = batch_tensor.unsqueeze(0)
            elif batch_tensor.dim() == 2 and batch_tensor.shape[0] != len(trajectory_cubes):
                # Need to reshape
                if len(trajectory_cubes) == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)

            # Add to pre-allocated arrays
            batch_size = len(trajectory_cubes)
            inputs[idx:idx+batch_size] = batch_tensor
            targets[idx:idx+batch_size] = torch.tensor(
                trajectory_distances, dtype=torch.float32).unsqueeze(1)
            idx += batch_size

        # Move to device
        if device != "cpu":
            inputs = inputs.to(device)
            targets = targets.to(device)

        return cls(inputs=inputs, targets=targets)

    @classmethod
    def create_from_bellman_equation(cls, model: CubeValueNN, n_trajectories: int = 1000, n_moves: int = 20, seed: Optional[int] = None, device="cpu", tqdm_position: int = 0) -> "TrainingValueDataset":
        """Create dataset from random scramble trajectories using Bellman equation for targets.

        Similar to create_from_trajectories but uses Bellman equation V(c) = min(V(n) + 1)
        to compute target values instead of actual distance from solved state.

        Args:
            model: CubeValueNN model used to evaluate neighbor states
            n_trajectories: Number of random trajectories to generate
            n_moves: Number of scramble moves per trajectory
            seed: Random seed for reproducibility
            device: Device to store tensors ('cpu' or 'cuda')
            tqdm_position: Position for tqdm progress bar (for parallel execution)

        Returns:
            TrainingValueDataset with Bellman-derived target values
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        value_function = model.as_value_function()
        cube_to_tensor = model.get_cube_to_tensor()

        # Get sample tensor to determine shape
        sample_cube = Cube()
        sample_tensor = cube_to_tensor(sample_cube)
        if sample_tensor.dim() > 1 and sample_tensor.shape[0] == 1:
            sample_tensor = sample_tensor.squeeze(0)
        tensor_shape = sample_tensor.shape

        # Pre-allocate for total size: n_trajectories * n_moves (no solved state)
        size = n_trajectories * n_moves
        inputs = torch.zeros((size, *tensor_shape), dtype=torch.float32)
        targets = torch.zeros((size, 1), dtype=torch.float32)

        idx = 0

        # Generate trajectories
        for traj_idx in tqdm(range(n_trajectories), desc="Generating Bellman trajectories", leave=False, position=tqdm_position):
            # Start from solved state
            cube = Cube()
            trajectory_cubes = []

            # Generate optimized random sequence
            move_sequence = Cube.generate_scramble_sequence(
                n_moves, seed=seed + traj_idx if seed is not None else None)

            # Apply moves and collect trajectory
            for move_idx, move in enumerate(move_sequence):
                cube.move(move)
                trajectory_cubes.append(cube.copy())

            # Batch convert cubes to tensors
            batch_tensor = cube_to_tensor(trajectory_cubes)

            # Handle tensor dimensions
            if batch_tensor.dim() == 1:
                batch_tensor = batch_tensor.unsqueeze(0)
            elif batch_tensor.dim() == 2 and batch_tensor.shape[0] != len(trajectory_cubes):
                if len(trajectory_cubes) == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)

            # Compute Bellman targets for each cube in trajectory
            trajectory_values = []
            for move_idx, trajectory_cube in enumerate(trajectory_cubes):
                # Get all neighbors
                neighbor_cubes = trajectory_cube.get_all_neighbors()
                # Evaluate neighbors using value function
                neighbor_values = value_function(neighbor_cubes)
                # Bellman equation: V(c) = min(V(n)) + 1
                min_neighbor_value = float(neighbor_values.min())
                bellman_value = min_neighbor_value + 1
                # Use minimum of Bellman value and actual distance
                current_value = move_idx + 1
                final_value = min(bellman_value, current_value)
                trajectory_values.append(final_value)

            # Add to pre-allocated arrays
            batch_size = len(trajectory_cubes)
            inputs[idx:idx+batch_size] = batch_tensor
            targets[idx:idx+batch_size] = torch.tensor(
                trajectory_values, dtype=torch.float32).unsqueeze(1)
            idx += batch_size

        # Move to device
        if device != "cpu":
            inputs = inputs.to(device)
            targets = targets.to(device)

        return cls(inputs=inputs, targets=targets)

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        return self._inputs[idx], self._targets[idx]

    def to(self, device: str):
        """Move the entire dataset to a device.

        Args:
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            Self for method chaining
        """
        self._inputs = self._inputs.to(device)
        self._targets = self._targets.to(device)
        return self

    def pin_memory(self):
        """Pin tensors in memory for faster CPU->GPU transfer.

        Returns:
            Self for method chaining
        """
        if self._inputs.device.type == 'cpu':
            self._inputs = self._inputs.pin_memory()
            self._targets = self._targets.pin_memory()
        return self
