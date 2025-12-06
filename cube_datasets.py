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

    new_cubes: List[Cube] = []
    solved_cube = Cube()
    new_cubes.append(solved_cube)
    cubes[solved_cube.copy()] = 0

    previously_added_cubes: List[Cube] = []
    previously_added_cubes = new_cubes.copy()

    for i in range(n_moves):
        print(f"Generating cubes with {i+1} moves...")
        new_cubes = []
        for cube in previously_added_cubes:
            for move in moves:
                new_cube = cube.copy()
                new_cube.move(move)
                if new_cube not in cubes:
                    cubes[new_cube.copy()] = i + 1
                    new_cubes.append(new_cube)
        print(f"Added {len(new_cubes)} new cubes.")
        previously_added_cubes = new_cubes.copy()
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

    new_cubes: List[Cube] = []
    solved_cube = Cube()
    new_cubes.append(solved_cube)
    cubes[solved_cube.copy()] = 0

    previously_added_cubes: List[Cube] = []
    previously_added_cubes = new_cubes.copy()

    for i in range(n_moves):
        print(f"Generating cubes with {i+1} moves...")
        new_cubes = []
        for cube in previously_added_cubes:
            for move in moves:
                new_cube = cube.copy()
                new_cube.move(move)
                similar_cubes = new_cube.get_all_similar_cubes()
                if not any(similar_cube in cubes for similar_cube in similar_cubes):
                    cubes[new_cube.copy()] = i + 1
                    new_cubes.append(new_cube)
        print(f"Added {len(new_cubes)} new cubes.")
        previously_added_cubes = new_cubes.copy()
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
    Generate a dataset of all possible cubes reachable with exacly n_moves from the solved state.

    Args:
        n_moves (int): Number of moves needed to solve all cubes in the dataset.
    Returns:
        CubeDataset: the generated dataset.
    """
    cubes: CubeDataset = {}

    if n_moves < 1:
        return {Cube(): 0}

    cubes_n_minus_1 = get_single_optimal_value_dataset(
        n_moves=max(n_moves-1, 0))
    cubes_n_minus_2 = get_single_optimal_value_dataset(
        n_moves=max(n_moves-2, 0))

    for cube in cubes_n_minus_1.keys():
        for move in moves:
            new_cube = cube.copy()
            new_cube.move(move)
            if new_cube not in cubes_n_minus_1 and new_cube not in cubes_n_minus_2:
                cubes[new_cube.copy()] = n_moves
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

    cubes: CubeDataset = {}
    pbar = tqdm(total=n_cubes, desc="Generating cubes", leave=False)
    i = 0
    while len(cubes) < n_cubes and i < n_cubes * n_max_attempts:
        i += 1
        cube = Cube(n_scramble_moves=n_moves, scramble_seed=int(
            rng.integers(0, n_cubes*1000)))
        if cube not in cubes:
            cubes[cube] = n_moves
            pbar.update(1)
    pbar.close()
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
    seed_str = f"_{seed}" if seed is not None else ""
    path = f"datasets/single_upper_bound/single_upper_bound_{n_moves}_{n_cubes}{seed_str}.pt"
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
    return cubes


def generate_approximate_value_dataset_from_bellman_equation(approximate_value_function: ValueFunction, n_moves: int = 20, n_cubes: int = 1000, n_max_attempts: int = 5, min_value: float = 1) -> CubeDataset:
    cubes: CubeDataset = {}
    pbar = tqdm(total=n_cubes,
                desc=f"Generating approximate values from Bellman equation for cubes with {n_moves} moves", leave=False)

    i = 0
    max_value = min(n_moves, 20)  # theory says max optimal value is 20
    while len(cubes) < n_cubes and i < n_cubes * n_max_attempts:
        i += 1
        cube = Cube(n_scramble_moves=n_moves)
        if cube not in cubes and not cube.is_solved():
            # Bellman equation: V(c) = min(V(n) + 1 for n in neighbors)
            neighbor_cubes = cube.get_all_neighbors()
            neighbor_values = approximate_value_function(neighbor_cubes)
            min_neighbor_value = neighbor_values.min()
            max_neighbor_value = neighbor_values.max()
            # value = min_neighbor_value + 1 # standard Bellman update
            # value = max(max_neighbor_value - 1, min_neighbor_value + 1)
            value = max((max_neighbor_value + min_neighbor_value) /
                        2, min_neighbor_value + 1)

            cubes[cube] = np.clip(value, min_value, max_value)
            pbar.update(1)
    pbar.close()
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
    cubes_list = list(cubes.items())
    random.shuffle(cubes_list)
    for cube, value in tqdm(cubes_list[:n_to_extend], desc="Extending dataset with similar cubes", leave=False):
        similar_cubes = cube.get_all_similar_cubes()
        for similar_cube in similar_cubes:
            if similar_cube not in extended_cubes:
                extended_cubes[similar_cube] = value
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

    def __init__(self, dataset: List[Tuple[torch.Tensor, int]]):
        self._dataset = dataset

    @classmethod
    def create_balanced(cls, cube_to_tensor: CubeToTensor, n_moves_max: int = 15, n_cubes_per_dataset: int = 1000, seed: Optional[int] = None) -> "TrainingValueDataset":
        if seed is not None:
            random.seed(seed)

        _dataset = []
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
            for cube, value in tqdm(dataset, leave=False, desc=f"Processing {n_moves}/{n_moves_max} moves"):
                tensor = cube_to_tensor(cube)
                # Remove the batch dimension (first dimension) since dataloader will add it back
                if tensor.dim() > 1 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                _dataset.append(
                    (tensor, torch.tensor(value, dtype=torch.float32).unsqueeze(0)))

        balanced_dataset = cls(_dataset)

        # save dataset to file
        seed_str = f"_{seed}" if seed is not None else ""
        path = f"datasets/balanced/dataset_{n_moves_max}_{n_cubes_per_dataset}{seed_str}.pt"
        with open(path, "wb") as f:
            torch.save(balanced_dataset, f)

        return balanced_dataset

    @classmethod
    def create_from_bellman_equation(cls, model: CubeValueNN, n_moves_max: int = 15, n_cubes_per_move: int = 1000, n_moves_difference_max: Optional[int] = None, similar_cubes_extension: Optional[float] = None) -> "TrainingValueDataset":
        _dataset = []

        value_function = model.as_value_function()
        cube_to_tensor = model.get_cube_to_tensor()

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

            for cube, value in tqdm(dataset, leave=False, desc=f"Processing {n_moves}/{n_moves_max} moves"):
                cube_tensor = cube_to_tensor(cube)
                # Remove the batch dimension (first dimension) since dataloader will add it back
                if cube_tensor.dim() > 1 and cube_tensor.shape[0] == 1:
                    cube_tensor = cube_tensor.squeeze(0)
                _dataset.append(
                    (cube_tensor, torch.tensor(value, dtype=torch.float32).unsqueeze(0)))

        balanced_dataset = cls(_dataset)
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
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
