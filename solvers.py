""" Algorithm to solve a Rubik's Cube."""
import numpy as np
from typing import List, Optional, Protocol, Deque, Tuple
from collections import deque

from cube import Cube, Move, moves
from cube_nn import ValueFunction
from cube_datasets import CubeDataset, get_full_optimal_value_dataset

import heapq
import time

Solution = Optional[List[Move]]
""" Type alias for a solution, which is a list of moves or None if no solution is found. """


class Solver(Protocol):
    """ A protocol for a solver that takes a Cube and returns a Solution.

    Solver(cube: Cube) -> Solution """

    def __call__(self, cube: Cube) -> Solution:
        ...


def solve_cube_bfs(cube: Cube, max_depth: int = 5) -> Solution:
    """
    Solve the cube using BFS up to max_depth.
    Returns a list of moves if solved, or None if no solution is found.
    """
    if cube.is_solved():
        return []

    # BFS queue: each entry is (cube, moves_to_here)
    queue = deque([(cube.copy(), [])])
    visited = {cube.copy()}  # Store seen cubes

    while queue:
        current_cube, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for move in moves:
            new_cube = current_cube.copy()
            new_cube.move(move)

            if new_cube in visited:
                continue
            visited.add(new_cube.copy())

            new_path = path + [move]
            if new_cube.is_solved():
                return new_path

            queue.append((new_cube, new_path))
    return None


def solve_from_dataset(cube: Cube, dataset: CubeDataset) -> Solution:
    """
    Solve the cube using a dataset of known cubes and their optimal value.
    Only works if the cube is directly in the dataset, and the dataset contains the full path to the solution.

    Args:
        cube (Cube): The Cube object to solve.
        dataset (CubeDataset): A dataset of cubes and their optimal values.
    Returns:
        Solution: the solution.
    """
    if cube.is_solved():
        return []
    if not cube in dataset:
        return None

    cube_value = dataset[cube]
    for move in moves:
        new_cube = cube.copy()
        new_cube.move(move)
        if dataset.get(new_cube, float('inf')) < cube_value:
            sol = solve_from_dataset(new_cube, dataset)
            if sol is not None:
                return [move] + sol
    return None


class BFSSolverDataset:
    """
    A solver that uses BFS and a given dataset to find the optimal solution.
    """

    def __init__(self, dataset_moves: int = 5, max_bfs_depth: int = 5):
        """
        Initializes the BFSSolverDataset.

        Args:
            dataset_moves (int): The maximum number of moves in the dataset cubes.
            max_bfs_depth (int): The maximum depth to search for a solution using BFS.
        """

        self._dataset = get_full_optimal_value_dataset(dataset_moves)
        self._max_bfs_depth = max_bfs_depth

    def __call__(self, cube: Cube) -> Solution:
        # Check if the cube is already in the dataset
        if cube in self._dataset:
            return solve_from_dataset(cube, self._dataset)

        # If not, use BFS to find a solution up to max_bfs_depth
        # BFS queue: each entry is (cube, moves_to_here)
        queue: Deque[Tuple[Cube, List[Move]]] = deque([(cube.copy(), [])])
        visited = {cube.copy()}  # Store seen cubes

        while queue:
            current_cube, path = queue.popleft()

            for move in moves:
                new_cube = current_cube.copy()
                new_cube.move(move)

                if new_cube in visited:
                    continue

                if new_cube in self._dataset:
                    solution = solve_from_dataset(new_cube, self._dataset)
                    if solution is not None:
                        return path + [move] + solution

                if len(path) + 1 < self._max_bfs_depth:
                    visited.add(new_cube.copy())
                    new_path = path + [move]
                    queue.append((new_cube, new_path))
        return None


class SuperSolver:
    """
    A unified A* solver with optional features including weighting, noise injection, and random restarts.

    Features:
    - Standard A*: weight=1.0, noise=0.0, max_restarts=0
    - Weighted A*: weight > 1.0 makes search more greedy
    - Noisy A*: noise > 0.0 adds random perturbations to exploration
    - Restarting A*: max_restarts > 0 tries from different random starting positions
    """

    def __init__(self,
                 value_function: ValueFunction,
                 weight: float = 1.0,
                 noise: float = 0.0,
                 max_moves: int = 40,
                 max_queue_size: int = 1000,
                 t_max: float = 60,
                 max_restarts: int = 0,
                 restart_sequence_length: int = 3,
                 batch_size: int = 1,
                 seed: Optional[int] = None,
                 dataset_moves: Optional[int] = None):
        """
        Initializes the AStarSolver.

        Args:
            value_function (ValueFunction): A function that takes a Cube and returns a value 
                (e.g. the estimated number of moves to solve it).
            weight (float): The weight to apply to the cost so far. Higher values make the search 
                more greedy. Default is 1.0 (standard A*).
            noise (float): Standard deviation of Gaussian noise to add to cost estimates. 
                Default is 0.0 (no noise).
            max_moves (int): The maximum number of moves to search for a solution.
            max_queue_size (int): The maximum size of the priority queue to prevent memory issues.
            t_max (float): The maximum time to search for a solution in seconds. 
                If max_restarts > 0, this is the time per attempt.
            max_restarts (int): The maximum number of restart attempts from random positions. 
                Default is 0 (no restarts).
            restart_sequence_length (int): The length of random move sequences to apply when restarting.
            batch_size (int): The number of cubes to pop and process in parallel from the priority queue. 
                Default is 1 (no batching).
            seed (Optional[int]): Random seed for deterministic noise generation. Only used if noise > 0.
            dataset_moves (Optional[int]): If provided, loads a dataset of optimal values for cubes up to this number of moves.
        """
        self._value_function = value_function
        self._weight = weight
        self._noise = noise
        self._max_moves = max_moves
        self._max_queue_size = max_queue_size
        self._t_max = t_max
        self._max_restarts = max_restarts
        self._restart_sequence_length = restart_sequence_length
        self._batch_size = batch_size
        self._seed = seed
        self._dataset_moves = dataset_moves if dataset_moves is not None else 0
        self._dataset = get_full_optimal_value_dataset(self._dataset_moves)

    def _run_astar_attempt(self, cube: Cube, prefix_moves: List[Move], t_max_attempt: float) -> Solution:
        """
        Run a single A* search attempt from a given cube state.

        Args:
            cube (Cube): The starting cube state.
            prefix_moves (List[Move]): The moves already applied to reach this state.

        Returns:
            Solution: The solution if found, None otherwise.
        """
        start_time = time.time()

        if cube.is_solved():
            return prefix_moves

        # Initialize random seed if noise is enabled
        if self._noise > 0 and self._seed is not None:
            cube_seed = (self._seed + hash(cube)) % (2**31)
            np.random.seed(cube_seed)

        # Priority queue: (estimated_total_cost, cost_so_far, counter, cube, moves_to_here)
        queue = []
        counter = 0

        initial_cost = self._value_function(cube)
        heapq.heappush(queue, (initial_cost, 0, counter,
                       cube.copy(), prefix_moves))
        visited = {cube.copy(): 0}

        while queue and len(queue) <= self._max_queue_size and time.time() - start_time < t_max_attempt:
            # Pop batch_size cubes from the queue
            batch_items = []
            for _ in range(min(self._batch_size, len(queue))):
                if not queue:
                    break
                batch_items.append(heapq.heappop(queue))

            # Process each cube in the batch
            batch_cubes = []
            batch_metadata = []

            for _, cost_so_far, _, current_cube, path in batch_items:
                if current_cube.is_solved():
                    return path

                if cost_so_far >= self._max_moves:
                    continue

                batch_cubes.append(current_cube)
                batch_metadata.append((cost_so_far, path))

            # If no valid cubes to process, continue
            if not batch_cubes:
                continue

            # Get all neighbors for all cubes in batch
            all_neighbors = []
            all_neighbor_metadata = []

            for cube_idx, current_cube in enumerate(batch_cubes):
                cost_so_far, path = batch_metadata[cube_idx]
                new_cost = cost_so_far + 1
                neighbors = current_cube.get_all_neighbors()

                for i, neighbor in enumerate(neighbors):
                    # Check if neighbor is in dataset and can lead to solution
                    if neighbor in self._dataset:
                        solution = solve_from_dataset(neighbor, self._dataset)
                        if solution is not None:
                            return path + [moves[i]] + solution
                    if new_cost < self._max_moves - self._dataset_moves:
                        all_neighbors.append(neighbor)
                        all_neighbor_metadata.append(
                            (new_cost, path, neighbor))

            # Batch evaluate all neighbors at once
            if all_neighbors:
                neighbors_values = self._value_function(all_neighbors)

                # Process results and add to queue
                for i, (new_cost, path, neighbor) in enumerate(all_neighbor_metadata):
                    if neighbor in visited and visited[neighbor] <= new_cost:
                        continue

                    visited[neighbor] = new_cost

                    # Calculate estimated cost with optional noise
                    estimated_cost = self._weight * \
                        new_cost + neighbors_values[i]
                    if self._noise > 0:
                        estimated_cost += np.random.normal(0, self._noise)

                    # Find which move was used
                    move_idx = i % len(moves)
                    new_path = path + [moves[move_idx]]
                    counter += 1
                    heapq.heappush(queue, (estimated_cost, new_cost,
                                   counter, neighbor, new_path))

        return None

    def __call__(self, cube: Cube) -> Solution:
        """
        Solve the cube using A* with optional restarts.

        Args:
            cube (Cube): The cube to solve.

        Returns:
            Solution: The solution if found, None otherwise.
        """
        t_start = time.time()
        if cube.is_solved():
            return []

        # First attempt: try from the original cube state
        t_max_attempt = self._t_max / (self._max_restarts + 1)
        solution = self._run_astar_attempt(cube.copy(), [], t_max_attempt)
        if solution is not None:
            return solution

        # If restarts are disabled or first attempt succeeded, return
        if self._max_restarts == 0:
            return None

        # Try from different starting points
        for _ in range(self._max_restarts):
            # Generate a random move sequence
            new_cube = cube.copy()
            random_sequence = new_cube.scramble(self._restart_sequence_length)

            # Try solving from this new state
            t_now = time.time()
            t_max_attempt = min(self._t_max / (self._max_restarts + 1),
                                self._t_max - (t_now - t_start))
            solution = self._run_astar_attempt(
                new_cube, random_sequence, t_max_attempt)

            if solution is not None:
                return solution

        # All attempts failed
        return None
