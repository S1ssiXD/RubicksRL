import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Protocol
from cube import Cube, moves
from tqdm import tqdm
from torch.utils.data import Dataset
from protocols import CubeToTensor, ValueFunction
from cube_nn import CubeValueNN
import random

N_OPTIMAL_UP_TO = 4


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
        # Process in batches for better memory locality
        for cube in current_layer:
            for move in moves:
                new_cube = cube.copy()
                new_cube.move(move)
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


def generate_similar_optimal_value_dataset(n_moves: int = 5) -> CubeDataset:
    """
    Generate a dataset of all possible cubes reachable within n_moves from the solved state, with their optimal values, but does not include cubes which have similar cubes already in.

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
            for move in moves:
                new_cube = cube.copy()
                new_cube.move(move)
                # Check if any similar cube already exists (optimized with any())
                similar_cubes = new_cube.get_all_similar_cubes()
                if not any(sim in cubes for sim in similar_cubes):
                    cubes[new_cube] = depth + 1
                    next_layer.append(new_cube)
        print(f"Added {len(next_layer)} new cubes.")
        current_layer = next_layer
    return cubes


def get_similar_optimal_value_dataset(n_moves: int = 5) -> CubeDataset:
    """
    Get the similar optimal value dataset, generating it if not already existing.

    Args:
        n_moves (int): Maximum number of moves used to scramble the cubes.
    Returns:
        CubeDataset: the generated dataset.
    """
    path = f"datasets/similar/similar_{n_moves}.pt"
    try:
        cubes = load_cube_dataset(path)
    except FileNotFoundError:
        print(
            f"Similar optimal value dataset not found at {path}, generating it...")
        cubes = generate_similar_optimal_value_dataset(n_moves=n_moves)
        save_cube_dataset(cubes, path)
        print(f"Saved similar optimal value dataset to {path}.")
    return cubes


def generate_single_optimal_value_dataset(n_moves: int = 5) -> CubeDataset:
    """
    Generate a dataset of all possible cubes reachable with exactly n_moves from the solved state.

    Args:
        n_moves (int): Number of moves needed to solve all cubes in the dataset.
    Returns:
        CubeDataset: the generated dataset.
    """
    if n_moves < 1:
        return {Cube(): 0}

    cubes_n_minus_1 = get_single_optimal_value_dataset(n_moves - 1)
    cubes_n_minus_2 = get_single_optimal_value_dataset(max(n_moves - 2, 0))

    # Convert to sets for O(1) lookup instead of O(n)
    prev_set = set(cubes_n_minus_1.keys())
    prev2_set = set(cubes_n_minus_2.keys())

    # Pre-allocate with estimated size
    cubes: CubeDataset = {}

    # Process all cubes from previous layer
    for cube in cubes_n_minus_1.keys():
        for move in moves:
            new_cube = cube.copy()
            new_cube.move(move)
            # Use set membership for faster lookup
            if new_cube not in prev_set and new_cube not in prev2_set:
                cubes[new_cube] = n_moves
    return cubes


def get_single_optimal_value_dataset(n_moves: int = 5) -> CubeDataset:
    """
    Get the single optimal value dataset, generating it if not already existing.

    Args:
        n_moves (int): Number of moves needed to solve all cubes in the dataset.
    Returns:
        CubeDataset: the generated dataset.
    """
    path = f"datasets/single_optimal/single_{n_moves}.pt"
    try:
        cubes = load_cube_dataset(path)
        # print(f"Loaded single optimal value dataset from {path}.")
    except FileNotFoundError:
        print(
            f"Single optimal value dataset not found at {path}, generating it...")
        cubes = generate_single_optimal_value_dataset(n_moves=n_moves)
        save_cube_dataset(cubes, path)
        print(f"Saved single optimal value dataset to {path}.")
    return cubes


def generate_single_upper_bound_value_dataset(n_moves: int = 20, n_cubes: int = 1000, seed: Optional[int] = None, n_max_attempts: int = 5) -> CubeDataset:
    """
    Generate a dataset of n_cubes cubes, each scrambled with n_moves from the solved state.

    Args:
        n_moves (int): Number of moves used to scramble each cube.
        n_cubes (int): Number of cubes to generate.
        seed (Optional[int]): Random seed for reproducibility.
    Returns:
        CubeDataset: the generated dataset.
    """
    rng = np.random.default_rng(seed)

    # Pre-allocate dictionary with expected size
    cubes: CubeDataset = {}

    # Generate in batches for better progress tracking
    batch_size = min(100, n_cubes)
    max_attempts = n_cubes * n_max_attempts

    with tqdm(total=n_cubes, desc="Generating cubes", leave=False) as pbar:
        attempts = 0
        while len(cubes) < n_cubes and attempts < max_attempts:
            # Generate random seeds in batch
            seeds = rng.integers(0, n_cubes * 10000, size=batch_size)

            for scramble_seed in seeds:
                if len(cubes) >= n_cubes:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    break

                cube = Cube(n_scramble_moves=n_moves,
                            scramble_seed=int(scramble_seed))
                if cube not in cubes:
                    cubes[cube] = n_moves
                    pbar.update(1)

    return cubes


def get_single_upper_bound_value_dataset(n_moves: int = 20, n_cubes: int = 1000, seed: Optional[int] = None) -> CubeDataset:
    """
    Get the single upper bound value dataset, generating it if not already existing.

    Args:
        n_moves (int): Number of moves used to scramble each cube.
        n_cubes (int): Number of cubes to generate.
        seed (Optional[int]): Random seed for reproducibility.
    Returns:
        CubeDataset: the generated dataset.
    """
    if seed is not None:
        path = f"datasets/single_upper_bound/single_upper_bound_{n_moves}_{n_cubes}_{seed}.pt"
        try:
            cubes = load_cube_dataset(path)
            # print(f"Loaded single upper bound value dataset from {path}.")
        except FileNotFoundError:
            print(
                f"Single upper bound value dataset not found at {path}, generating it...")
            cubes = generate_single_upper_bound_value_dataset(
                n_moves=n_moves, n_cubes=n_cubes, seed=seed)
            save_cube_dataset(cubes, path)
            print(f"Saved single upper bound value dataset to {path}.")
    else:
        cubes = generate_single_upper_bound_value_dataset(
            n_moves=n_moves, n_cubes=n_cubes, seed=seed)
    return cubes


def generate_approximate_value_dataset_from_bellman_equation(approximate_value_function: ValueFunction, n_moves: int = 20, n_cubes: int = 1000, n_max_attempts: int = 5, min_value: float = 1) -> CubeDataset:
    """Generate dataset with approximate values using Bellman equation.

    Args:
        approximate_value_function: Function to evaluate cube states
        n_moves: Number of scramble moves
        n_cubes: Target number of cubes to generate
        n_max_attempts: Maximum attempts multiplier
        min_value: Minimum value to assign
    Returns:
        Dictionary mapping cubes to their approximate values
    """
    cubes: CubeDataset = {}
    max_value = min(n_moves, 20)  # theory says max optimal value is 20
    max_attempts = n_cubes * n_max_attempts

    with tqdm(total=n_cubes, desc=f"Generating approximate values from Bellman equation for cubes with {n_moves} moves", leave=False) as pbar:
        attempts = 0
        while len(cubes) < n_cubes and attempts < max_attempts:
            attempts += 1
            cube = Cube(n_scramble_moves=n_moves)

            if cube in cubes or cube.is_solved():
                continue

            # Bellman equation: V(c) = min(V(n) + 1 for n in neighbors)
            neighbor_cubes = cube.get_all_neighbors()
            neighbor_values = approximate_value_function(neighbor_cubes)
            min_neighbor_value = float(neighbor_values.min())
            max_neighbor_value = float(neighbor_values.max())

            # Average of min and max, but at least min + 1
            value = max((max_neighbor_value + min_neighbor_value) / 2,
                        min_neighbor_value + 1)

            cubes[cube] = np.clip(value, min_value, max_value)
            pbar.update(1)

    return cubes


def extend_dataset_with_similar_cubes(cubes: CubeDataset, fraction: float = 0.1) -> CubeDataset:
    """
    Extend the dataset by adding all similar cubes for some of the cubes in the dataset.

    Args:
        cubes (CubeDataset): The original dataset.
        fraction (float): The fraction of cubes to extend with their similar cubes.
    Returns:
        CubeDataset: The extended dataset.
    """
    extended_cubes = cubes.copy()
    n_to_extend = int(len(cubes) * fraction)

    # Sample without creating full list
    cubes_items = list(cubes.items())
    random.shuffle(cubes_items)
    selected_items = cubes_items[:n_to_extend]

    # Use set for O(1) membership test
    existing_keys = set(extended_cubes.keys())

    for cube, value in tqdm(selected_items, desc="Extending dataset with similar cubes", leave=False):
        similar_cubes = cube.get_all_similar_cubes()
        for similar_cube in similar_cubes:
            if similar_cube not in existing_keys:
                extended_cubes[similar_cube] = value
                existing_keys.add(similar_cube)

    return extended_cubes


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

    def __init__(self, dataset: Optional[List[Tuple[torch.Tensor, int]]] = None,
                 inputs: Optional[torch.Tensor] = None,
                 targets: Optional[torch.Tensor] = None):
        """Initialize dataset either from list of tuples or pre-allocated tensors.

        Args:
            dataset: Legacy format - list of (input_tensor, target_value) tuples
            inputs: Pre-allocated tensor of all inputs, shape (N, ...)
            targets: Pre-allocated tensor of all targets, shape (N, 1)
        """
        if inputs is not None and targets is not None:
            self._inputs: Optional[torch.Tensor] = inputs
            self._targets: Optional[torch.Tensor] = targets
            # Don't store redundant data
            self._dataset: Optional[List[Tuple[torch.Tensor, int]]] = None
        elif dataset is not None:
            self._dataset = dataset
            self._inputs = None
            self._targets = None
        else:
            raise ValueError(
                "Either dataset or (inputs, targets) must be provided")

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
        cube_for_scramble = Cube()

        # Generate trajectories
        for traj_idx in tqdm(range(n_trajectories), desc="Generating trajectories", leave=False, position=tqdm_position):
            # Start from solved state
            cube = Cube()
            trajectory_cubes = []
            trajectory_distances = []

            # Use cube.scramble to generate optimized random sequence (removes trivial cases)
            # We need to manually apply moves to capture trajectory
            move_sequence = cube_for_scramble.scramble(
                n_moves, seed=seed + traj_idx if seed is not None else None)

            # Apply moves and collect trajectory (skip solved state at distance 0)
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
    def create_balanced(cls, cube_to_tensor: CubeToTensor, n_moves_max: int = 15, n_cubes_per_dataset: int = 1000, seed: Optional[int] = None, device="cpu") -> "TrainingValueDataset":
        if seed is not None:
            random.seed(seed)

        # Calculate total size and get sample tensor shape
        total_size = (n_moves_max + 1) * n_cubes_per_dataset

        # Get a sample tensor to determine shape
        sample_cube = Cube()
        sample_tensor = cube_to_tensor(sample_cube)
        if sample_tensor.dim() > 1 and sample_tensor.shape[0] == 1:
            sample_tensor = sample_tensor.squeeze(0)
        tensor_shape = sample_tensor.shape

        # Pre-allocate tensors on CPU first (more memory efficient)
        inputs = torch.zeros((total_size, *tensor_shape), dtype=torch.float32)
        targets = torch.zeros((total_size, 1), dtype=torch.float32)

        idx = 0
        for n_moves in range(n_moves_max + 1):
            if n_moves <= N_OPTIMAL_UP_TO:
                dataset = get_single_optimal_value_dataset(n_moves)
            else:
                dataset = get_single_upper_bound_value_dataset(
                    n_moves, n_cubes_per_dataset, seed=seed)
            if len(dataset) > n_cubes_per_dataset:
                dataset = random.sample(
                    list(dataset.items()), n_cubes_per_dataset)
            else:
                # repeat samples to reach n_cubes_per_dataset
                dataset = list(dataset.items())
                while len(dataset)*2 < n_cubes_per_dataset:
                    dataset.extend(dataset)
                dataset.extend(random.sample(
                    dataset, n_cubes_per_dataset - len(dataset)))

            # Batch convert cubes to tensors for efficiency
            cubes = [cube for cube, _ in dataset]
            values = [value for _, value in dataset]

            # Convert in batches to avoid memory issues
            batch_size = 256
            for i in tqdm(range(0, len(cubes), batch_size), leave=False, desc=f"Processing {n_moves}/{n_moves_max} moves"):
                batch_cubes = cubes[i:i+batch_size]
                batch_values = values[i:i+batch_size]

                batch_tensor = cube_to_tensor(batch_cubes)
                if batch_tensor.dim() > 2 and batch_tensor.shape[0] == len(batch_cubes):
                    # Already batched correctly
                    pass
                elif batch_tensor.dim() > 1 and len(batch_cubes) == 1:
                    batch_tensor = batch_tensor.squeeze(0).unsqueeze(0)

                batch_end = idx + len(batch_cubes)
                inputs[idx:batch_end] = batch_tensor
                targets[idx:batch_end] = torch.tensor(
                    batch_values, dtype=torch.float32).unsqueeze(1)
                idx += len(batch_cubes)

        # Move to device after all data is collected
        if device != "cpu":
            inputs = inputs.to(device)
            targets = targets.to(device)

        balanced_dataset = cls(inputs=inputs, targets=targets)

        # save dataset to file
        if seed is not None:
            path = f"datasets/balanced/dataset_{n_moves_max}_{n_cubes_per_dataset}_{seed}.pt"
            with open(path, "wb") as f:
                torch.save(balanced_dataset, f)

        return balanced_dataset

    @classmethod
    def create_from_bellman_equation(cls, model: CubeValueNN, n_moves_max: int = 15, n_cubes_per_move: int = 1000, n_moves_difference_max: Optional[int] = None, similar_cubes_extension: Optional[float] = None, device="cpu") -> "TrainingValueDataset":
        value_function = model.as_value_function()
        cube_to_tensor = model.get_cube_to_tensor()

        # First pass: collect all datasets to know total size
        all_datasets = []
        total_size = 0

        for n_moves in range(n_moves_max + 1):
            min_value = 1 if n_moves_difference_max is None else max(
                1, n_moves - n_moves_difference_max)
            if n_moves <= N_OPTIMAL_UP_TO:
                dataset = get_single_optimal_value_dataset(n_moves)
            else:
                dataset = generate_approximate_value_dataset_from_bellman_equation(
                    value_function, n_moves, n_cubes_per_move, min_value=min_value)
            if len(dataset) == 0:
                continue
            if len(dataset) > n_cubes_per_move:
                dataset = random.sample(
                    list(dataset.items()), n_cubes_per_move)
            else:
                # repeat samples to reach n_cubes_per_dataset
                dataset = list(dataset.items())
                while len(dataset)*2 < n_cubes_per_move:
                    dataset.extend(dataset)
                dataset.extend(random.sample(
                    dataset, n_cubes_per_move - len(dataset)))

            if similar_cubes_extension is not None:
                dataset_dict = dict(dataset)
                dataset_dict = extend_dataset_with_similar_cubes(
                    dataset_dict, fraction=similar_cubes_extension)
                dataset = list(dataset_dict.items())

            all_datasets.append((n_moves, dataset))
            total_size += len(dataset)

        # Get sample tensor shape
        sample_cube = Cube()
        sample_tensor = cube_to_tensor(sample_cube)
        if sample_tensor.dim() > 1 and sample_tensor.shape[0] == 1:
            sample_tensor = sample_tensor.squeeze(0)
        tensor_shape = sample_tensor.shape

        # Pre-allocate tensors on CPU first
        inputs = torch.zeros((total_size, *tensor_shape), dtype=torch.float32)
        targets = torch.zeros((total_size, 1), dtype=torch.float32)

        idx = 0
        for n_moves, dataset in all_datasets:
            cubes = [cube for cube, _ in dataset]
            values = [value for _, value in dataset]

            # Convert in batches
            batch_size = 256
            for i in tqdm(range(0, len(cubes), batch_size), leave=False, desc=f"Processing {n_moves}/{n_moves_max} moves"):
                batch_cubes = cubes[i:i+batch_size]
                batch_values = values[i:i+batch_size]

                batch_tensor = cube_to_tensor(batch_cubes)
                if batch_tensor.dim() > 2 and batch_tensor.shape[0] == len(batch_cubes):
                    pass
                elif batch_tensor.dim() > 1 and len(batch_cubes) == 1:
                    batch_tensor = batch_tensor.squeeze(0).unsqueeze(0)

                batch_end = idx + len(batch_cubes)
                inputs[idx:batch_end] = batch_tensor
                targets[idx:batch_end] = torch.tensor(
                    batch_values, dtype=torch.float32).unsqueeze(1)
                idx += len(batch_cubes)

        # Move to device after all data is collected
        if device != "cpu":
            inputs = inputs.to(device)
            targets = targets.to(device)

        balanced_dataset = cls(inputs=inputs, targets=targets)
        return balanced_dataset

    @classmethod
    def load(cls, path) -> "TrainingValueDataset":
        try:
            with open(path, "rb") as f:
                dataset = torch.load(f, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {path}")
        assert isinstance(
            dataset, TrainingValueDataset), f"Loaded object is not a TrainingValueDataset, got {type(dataset)}"
        return dataset

    def __len__(self):
        if self._inputs is not None:
            return len(self._inputs)
        assert self._dataset is not None
        return len(self._dataset)

    def __getitem__(self, idx):
        if self._inputs is not None:
            assert self._targets is not None
            return self._inputs[idx], self._targets[idx]
        assert self._dataset is not None
        return self._dataset[idx]

    def to(self, device: str):
        """Move the entire dataset to a device.

        Args:
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            Self for method chaining
        """
        if self._inputs is not None:
            assert self._targets is not None
            self._inputs = self._inputs.to(device)
            self._targets = self._targets.to(device)
        else:
            # Convert legacy format to tensor format and move
            assert self._dataset is not None
            inputs_list = []
            targets_list = []
            for inp, tgt in self._dataset:
                inputs_list.append(inp)
                targets_list.append(tgt)
            self._inputs = torch.stack(inputs_list).to(device)
            self._targets = torch.stack(targets_list).to(device)
            self._dataset = None  # Free memory
        return self

    def pin_memory(self):
        """Pin tensors in memory for faster CPU->GPU transfer.

        Returns:
            Self for method chaining
        """
        if self._inputs is not None and self._inputs.device.type == 'cpu':
            assert self._targets is not None
            self._inputs = self._inputs.pin_memory()
            self._targets = self._targets.pin_memory()
        return self
