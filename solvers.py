""" Algorithm to solve a Rubik's Cube."""
import numpy as np
import torch
from typing import List, Optional, Protocol, Deque, Tuple
from collections import deque
from torch import nn

from cube import Cube, Move, Face, MoveSpecifier, moves
from cube_nn import CubeValueNN
from cube_datasets import CubeDataset, get_full_optimal_value_dataset, get_similar_optimal_value_dataset
from protocols import ValueFunction, CubeToTensor

import heapq
import time

Solution = Optional[List[Move]]
""" Type alias for a solution, which is a list of moves or None if no solution is found. """


class Solver(Protocol):
    """ A protocol for a solver that takes a Cube and returns a Solution.

    Solver(cube: Cube) -> Solution """

    def __call__(self, cube: Cube) -> Solution:
        ...


def nn_solver(model: CubeValueNN, max_moves: Optional[int] = None) -> Solver:
    """
    Create a solver that uses a neural network to estimate the value of a cube.

    Args:
        network (nn.Module): The neural network model.
        cube_transform (CubeToTensor): A function to transform Cube objects to tensors.
    Returns:
        Solver: the solver.
    """

    def solver(cube: Cube) -> Solution:
        if max_moves is None:
            return solve_cube_with_value_function(cube, model.as_value_function())
        return solve_cube_with_value_function(cube, model.as_value_function(), max_moves)
    return solver


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


def solve_cube_with_value_function(cube: Cube, value_function: ValueFunction, max_moves: int = 40) -> Solution:
    """
    Solve the cube using a neural network that estimate the number of moves needed.

    Args:
        cube (Cube): The Cube object to solve.
        value_function (ValueFunction): A function that takes a Cube and returns a value (e.g the estimated number of moves to solve it). Lower values are better.
        max_moves (int): The maximum number of moves to search for a solution.
    Returns:
        Solution: the solution.
    """

    sol = []
    if cube.is_solved():
        return sol

    visited = {cube.copy()}

    while len(sol) < max_moves and not cube.is_solved():
        # Check which move leads to the lowest estimated value
        best_value = float('inf')
        # Default move, will be replaced
        best_move = Move(Face.FRONT, MoveSpecifier.CLOCKWISE)
        for move in moves:
            new_cube = cube.copy()
            new_cube.move(move)

            if new_cube in visited:
                continue

            value = value_function(new_cube)

            if value < best_value:
                best_value = value
                best_move = move
        sol.append(best_move)
        cube.move(best_move)
        visited.add(cube.copy())

    return sol if cube.is_solved() else None


def solve_from_dataset(cube: Cube, dataset: CubeDataset) -> Solution:
    """
    Solve the cube using a dataset of known cubes and their optimal value.

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


def solve_from_similar_dataset(cube: Cube, dataset: CubeDataset) -> Solution:
    """
    Solve the cube using a dataset of known cubes and their optimal value, exploiting cube similarity.

    Args:
        cube (Cube): The Cube object to solve.
        dataset (CubeDataset): A dataset of cubes and their optimal values.
    Returns:
        Solution: the solution.
    """
    if cube.is_solved():
        return []

    similar_cubes = cube.get_all_similar_cubes()
    if not any(similar_cube in dataset for similar_cube in similar_cubes):
        return None

    cube_value = min(dataset.get(similar_cube, float('inf'))
                     for similar_cube in similar_cubes)
    for move in moves:
        new_cube = cube.copy()
        new_cube.move(move)
        similar_cubes = new_cube.get_all_similar_cubes()
        if any(dataset.get(similar_cube, float('inf')) < cube_value for similar_cube in similar_cubes):
            sol = solve_from_similar_dataset(new_cube, dataset)
            if sol is not None:
                return [move] + sol
    return None


class DepthValueSolver:
    """
    A solver that uses value function to guide the search, and looks for a solution up to a maximum depth.
    """

    def __init__(self, value_function: ValueFunction, max_depth: int = 3, max_moves: int = 40):
        """
        Initializes the DepthValueSolver.

        Args:
            value_function (ValueFunction): A function that takes a Cube and returns a value (e.g. the estimated number of moves to solve it).
            max_depth (int): The maximum depth to search for a solution.
            max_moves (int): The maximum number of moves to search for a solution.
        """
        self._value_function = value_function
        self._max_depth = max_depth
        self._max_moves = max_moves

        self._check_moves = []
        previous_check_moves = [[]]
        for depth in range(1, max_depth):
            new_check_moves = []
            for move_sequence in previous_check_moves:
                for move in moves:
                    new_sequence = move_sequence + [move]
                    new_check_moves.append(new_sequence)
            self._check_moves.extend(new_check_moves)
            previous_check_moves = new_check_moves

        self._perform_moves = []
        for move_sequence in previous_check_moves:
            for move in moves:
                new_sequence = move_sequence + [move]
                self._perform_moves.append(new_sequence)

    def __call__(self, cube: Cube) -> Solution:

        sol = []
        if cube.is_solved():
            return sol

        # Check if we can solve the cube with less than max_depth moves
        for move_sequence in self._check_moves:
            new_cube = cube.copy()
            new_cube.scramble(move_sequence)
            if new_cube.is_solved():
                return move_sequence

        visited = {cube.copy()}

        i = 0
        while i < self._max_moves - self._max_depth and not cube.is_solved():

            best_move = moves[0]
            best_output = float('inf')

            for move_sequence in self._perform_moves:
                new_cube = cube.copy()
                new_cube.scramble(move_sequence[:1])
                if new_cube in visited:
                    continue
                new_cube.scramble(move_sequence[1:])

                if new_cube.is_solved():
                    sol.extend(move_sequence)
                    return sol

                output = self._value_function(new_cube)

                if output < best_output:
                    best_output = output
                    best_move = move_sequence[0]

            cube.move(best_move)
            visited.add(cube.copy())
            sol.append(best_move)
            i += 1

        return sol if cube.is_solved() else None


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


class BFSSolverSimilarDataset:
    """
    A solver that uses BFS and cube similarity and given dataset to find the optimal solution.
    """

    def __init__(self, dataset_moves: int = 5, max_bfs_depth: int = 5):
        """
        Initializes the BFSSolverSimilarDataset.

        Args:
            dataset_moves (int): The maximum number of moves in the dataset cubes.
            max_bfs_depth (int): The maximum depth to search for a solution using BFS.
        """

        self._dataset = get_similar_optimal_value_dataset(dataset_moves)
        self._max_bfs_depth = max_bfs_depth

    def __call__(self, cube: Cube) -> Solution:
        # Check if the cube (or a similar) is already in the dataset
        sol_from_dataset = solve_from_similar_dataset(cube, self._dataset)
        if sol_from_dataset is not None:
            return sol_from_dataset

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

                similar_cubes = new_cube.get_all_similar_cubes()
                if any(similar_cube in self._dataset for similar_cube in similar_cubes):
                    solution = solve_from_similar_dataset(
                        new_cube, self._dataset)
                    if solution is not None:
                        return path + [move] + solution

                if len(path) + 1 < self._max_bfs_depth:
                    visited.add(new_cube.copy())
                    new_path = path + [move]
                    queue.append((new_cube, new_path))
        return None


class WeightedAStarSolver:
    """
    A solver that uses the A* algorithm with a given value function to find a solution.
    If the value function is perfect, this will find the optimal solution, otherwise it might find a suboptimal solution.
    """

    def __init__(self, value_function: ValueFunction, weight: float = 1.0, max_moves: int = 40, max_queue_size: int = 1000, t_max: float = 60):
        """
        Initializes the AStarSolver.

        Args:
            value_function (ValueFunction): A function that takes a Cube and returns a value (e.g. the estimated number of moves to solve it).
            weight (float): The weight to apply to the cost so far. Higher values make the search more greedy. Default is 1.0 (standard A*).
            max_moves (int): The maximum number of moves to search for a solution.
            max_queue_size (int): The maximum size of the priority queue to prevent memory issues.
            t_max (float): The maximum time to search for a solution in seconds.
        """
        self._value_function = value_function
        self._weight = weight
        self._max_moves = max_moves
        self._max_queue_size = max_queue_size
        self._t_max = t_max

    def __call__(self, cube: Cube) -> Solution:
        start_time = time.time()

        sol = []
        if cube.is_solved():
            return sol

        # Priority queue: each entry is (estimated_total_cost, cost_so_far, counter, cube, moves_to_here)
        queue = []
        # To avoid comparison of Cube objects in heapq (break ties)
        counter = 0
        heapq.heappush(queue, (self._value_function(cube),
                       0, counter, cube.copy(), []))
        visited = {cube.copy(): 0}  # Store seen cubes with their cost

        while queue and len(queue) <= self._max_queue_size and time.time() - start_time < self._t_max:
            _, cost_so_far, _, current_cube, path = heapq.heappop(
                queue)

            if current_cube.is_solved():
                return path

            if cost_so_far >= self._max_moves:
                continue

            new_cost = cost_so_far + 1
            neighbors = current_cube.get_all_neighbors()
            neighbors_values = self._value_function(neighbors)
            for i in range(len(neighbors)):

                if neighbors[i] in visited and visited[neighbors[i]] <= new_cost:
                    continue

                visited[neighbors[i]] = new_cost
                estimated_cost = self._weight * new_cost + neighbors_values[i]
                new_path = path + [moves[i]]
                counter += 1
                heapq.heappush(
                    queue, (estimated_cost, new_cost, counter, neighbors[i], new_path))
        return None


class NoisyWeightedAStarSolver:
    """
    A solver that uses the A* algorithm with a given value function to find a solution.
    If the value function is perfect, this will find the optimal solution, otherwise it might find a suboptimal solution.
    """

    def __init__(self, value_function: ValueFunction, weight: float = 1.0, noise: float = 0.5, max_moves: int = 40, max_queue_size: int = 1000, t_max: float = 60, seed: Optional[int] = None):
        """
        Initializes the AStarSolver.

        Args:
            value_function (ValueFunction): A function that takes a Cube and returns a value (e.g. the estimated number of moves to solve it).
            weight (float): The weight to apply to the cost so far. Higher values make the search more greedy. Default is 1.0 (standard A*).
            max_moves (int): The maximum number of moves to search for a solution.
            max_queue_size (int): The maximum size of the priority queue to prevent memory issues.
            t_max (float): The maximum time to search for a solution in seconds.
        """
        self._value_function = value_function
        self._weight = weight
        self._noise = noise
        self._max_moves = max_moves
        self._max_queue_size = max_queue_size
        self._t_max = t_max
        self._seed = seed

    def __call__(self, cube: Cube) -> Solution:
        start_time = time.time()

        if self._seed is not None:
            np.random.seed(self._seed)

        sol = []
        if cube.is_solved():
            return sol

        # Priority queue: each entry is (estimated_total_cost, cost_so_far, counter, cube, moves_to_here)
        queue = []
        # To avoid comparison of Cube objects in heapq (break ties)
        counter = 0
        heapq.heappush(queue, (self._value_function(cube),
                       0, counter, cube.copy(), []))
        visited = {cube.copy(): 0}  # Store seen cubes with their cost

        while queue and len(queue) <= self._max_queue_size and time.time() - start_time < self._t_max:
            _, cost_so_far, _, current_cube, path = heapq.heappop(
                queue)

            if current_cube.is_solved():
                return path

            if cost_so_far >= self._max_moves:
                continue

            new_cost = cost_so_far + 1
            neighbors = current_cube.get_all_neighbors()
            neighbors_values = self._value_function(neighbors)
            for i in range(len(neighbors)):

                if neighbors[i] in visited and visited[neighbors[i]] <= new_cost:
                    continue

                visited[neighbors[i]] = new_cost
                estimated_cost = self._weight * new_cost + \
                    neighbors_values[i] + np.random.normal(0, self._noise)
                new_path = path + [moves[i]]
                counter += 1
                heapq.heappush(
                    queue, (estimated_cost, new_cost, counter, neighbors[i], new_path))
        return None


class RestartingWeightedAStarSolver:
    """
    A solver that uses the Weighted A* algorithm, but if it fails or times out, 
    it tries again from a different initial state by applying random move sequences.
    """

    def __init__(self, value_function: ValueFunction, weight: float = 1.0, max_moves: int = 40,
                 max_queue_size: int = 1000, t_max_per_attempt: float = 10,
                 max_restarts: int = 5, restart_sequence_length: int = 3):
        """
        Initializes the RestartingWeightedAStarSolver.

        Args:
            value_function (ValueFunction): A function that takes a Cube and returns a value.
            weight (float): The weight to apply to the cost so far. Higher values make the search more greedy.
            max_moves (int): The maximum number of moves to search for a solution.
            max_queue_size (int): The maximum size of the priority queue.
            t_max_per_attempt (float): The maximum time per A* attempt in seconds.
            max_restarts (int): The maximum number of restart attempts.
            restart_sequence_length (int): The length of random move sequences to apply when restarting.
        """
        self._value_function = value_function
        self._weight = weight
        self._max_moves = max_moves
        self._max_queue_size = max_queue_size
        self._t_max_per_attempt = t_max_per_attempt
        self._max_restarts = max_restarts
        self._restart_sequence_length = restart_sequence_length

    def _run_weighted_astar(self, cube: Cube, prefix_moves: List[Move]) -> Solution:
        """
        Run the Weighted A* algorithm from a given cube state.

        Args:
            cube (Cube): The starting cube state.
            prefix_moves (List[Move]): The moves already applied to reach this state.

        Returns:
            Solution: The solution if found, None otherwise.
        """
        start_time = time.time()

        if cube.is_solved():
            return prefix_moves

        # Priority queue: (estimated_total_cost, cost_so_far, counter, cube, moves_to_here)
        queue = []
        counter = 0
        heapq.heappush(queue, (self._value_function(cube), 0,
                       counter, cube.copy(), prefix_moves))
        visited = {cube.copy(): 0}

        while queue and len(queue) <= self._max_queue_size and time.time() - start_time < self._t_max_per_attempt:
            _, cost_so_far, _, current_cube, path = heapq.heappop(queue)

            if current_cube.is_solved():
                return path

            if cost_so_far >= self._max_moves:
                continue

            new_cost = cost_so_far + 1
            neighbors = current_cube.get_all_neighbors()
            neighbors_values = self._value_function(neighbors)

            for i in range(len(neighbors)):
                if neighbors[i] in visited and visited[neighbors[i]] <= new_cost:
                    continue

                visited[neighbors[i]] = new_cost
                estimated_cost = self._weight * new_cost + neighbors_values[i]
                new_path = path + [moves[i]]
                counter += 1
                heapq.heappush(queue, (estimated_cost, new_cost,
                               counter, neighbors[i], new_path))

        return None

    def __call__(self, cube: Cube) -> Solution:
        """
        Solve the cube using Weighted A* with restarts from random positions.

        Args:
            cube (Cube): The cube to solve.

        Returns:
            Solution: The solution if found, None otherwise.
        """
        # First attempt: try from the original cube state
        solution = self._run_weighted_astar(cube.copy(), [])
        if solution is not None:
            return solution

        # If first attempt failed, try from different starting points
        for restart_idx in range(self._max_restarts):
            # Generate a random move sequence
            random_sequence = [moves[np.random.randint(0, len(moves))]
                               for _ in range(self._restart_sequence_length)]

            # Apply the random sequence to get a new starting state
            new_start_cube = cube.copy()
            new_start_cube.move(random_sequence)

            # Try solving from this new state
            # The solution should include the random sequence at the beginning
            solution = self._run_weighted_astar(
                new_start_cube, random_sequence)

            if solution is not None:
                return solution

        # All attempts failed
        return None


class RandomSequenceSolver:

    def __init__(self, value_function: ValueFunction, sequence_length: int = 3, max_sequences: int = 40, goal_factor: float = 0.6, t_max: float = 60) -> None:
        """
        Initializes the RandomSequenceSolver.

        Args:
            value_function (ValueFunction): A function that takes a Cube and returns a value (e.g. the estimated number of moves to solve it).
            sequence_length (int): The length of random move sequences to try.
            max_sequences (int): The maximum number of random sequences to try.
            goal_factor (float): The factor by which the value must decrease to accept a move sequence.
            t_max (float): The maximum time to search for a solution in seconds.
        """
        self._value_function = value_function
        self._sequence_length = sequence_length
        self._max_sequences = max_sequences
        self._goal_factor = goal_factor
        self._t_max = t_max

        self._full_optimal_4 = get_full_optimal_value_dataset(4)

    def __call__(self, cube: Cube) -> Solution:
        start_time = time.time()

        sol = []
        if cube.is_solved():
            return sol

        visited = {cube.copy()}
        current_value = self._value_function(cube)

        while not cube.is_solved() and time.time() - start_time < self._t_max:

            target_value = current_value - self._sequence_length * self._goal_factor

            best_sequence = None
            best_value = np.inf

            for _ in range(self._max_sequences):
                sequence = [moves[np.random.randint(0, len(moves))]
                            for _ in range(self._sequence_length)]
                new_cube = cube.copy()
                new_cube.move(sequence)

                if new_cube in visited:
                    continue

                value = self._value_function(new_cube)

                if value < best_value:
                    best_sequence = sequence
                    best_value = value

                if value < target_value:
                    break

            if best_sequence is None:
                return None

            sol.extend(best_sequence)
            cube.move(best_sequence)
            visited.add(cube.copy())
            current_value = best_value

            if cube in self._full_optimal_4:
                sol_from_dataset = solve_from_dataset(
                    cube, self._full_optimal_4)
                if sol_from_dataset is not None:
                    sol.extend(sol_from_dataset)

            print(
                f"Current solution length: {len(sol)}, cube value: {current_value}")

        return sol if cube.is_solved() else None
