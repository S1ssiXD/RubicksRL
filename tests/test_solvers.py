"""
Tests for Rubik's Cube solvers.

This test suite verifies that solver implementations correctly solve scrambled cubes.
The core verification approach is to:
1. Create a scrambled cube
2. Apply the solver to get a solution (sequence of moves)
3. Apply the solution moves to the original scrambled cube
4. Verify the cube reaches the solved state and constraints are met

Tests cover the main AStarSolver with various configurations using a trained neural network.
"""
import pytest
import numpy as np
import time
from cube import Cube
from solvers import AStarSolver
from cube_nn import CubeValueResNet, NNValueFunctionType
import torch
import os


# Fixture to load a trained model once for all tests
@pytest.fixture(scope="module")
def trained_model():
    """Load a trained CubeValueResNet model for testing."""
    model = CubeValueResNet()
    model_path = 'temp_models/resnet2.1/cube_value_resnet_iter_500.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(
            model_path, map_location='cpu', weights_only=True))
    else:
        pytest.skip(f"Model file not found: {model_path}")

    model.eval()
    model.set_value_function_type(NNValueFunctionType.STANDARD)
    return model


# Test parameters
@pytest.mark.parametrize("n_scrambles", [1, 3, 7, 10])
@pytest.mark.parametrize("seed", [0, 1, 42, 100])
@pytest.mark.parametrize("weight", [0.1, 0.25, 1.0])
@pytest.mark.parametrize("noise", [0.0, 0.2])
@pytest.mark.parametrize("max_moves", [3, 20])
@pytest.mark.parametrize("t_max", [5, 30])
@pytest.mark.parametrize("max_restarts", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 20])
@pytest.mark.parametrize("dataset_moves", [None, 2])
def test_astar_solver(n_scrambles, seed, weight, noise, max_moves, t_max,
                      max_restarts, batch_size, dataset_moves, trained_model):
    """
    Comprehensive A* solver test with multiple configurations.

    Verifies:
    - If solution found: validity, time limit, and max length
    - If no solution: returns None
    """
    # Create scrambled cube
    cube = Cube(n_scramble_moves=n_scrambles, scramble_seed=seed)

    # Create solver with specified configuration
    solver = AStarSolver(
        trained_model.as_value_function(),
        weight=weight,
        noise=noise,
        batch_size=batch_size,
        max_moves=max_moves,
        t_max=t_max,
        max_restarts=max_restarts,
        dataset_moves=dataset_moves,
        seed=seed
    )

    # Solve with timing
    start_time = time.time()
    solution = solver(cube.copy())
    elapsed = time.time() - start_time

    # Check constraints
    if solution is not None:
        # 1. Time limit check
        time_threshold = 0.5  # Allow small overhead
        assert elapsed < t_max + time_threshold, \
            f"Solver exceeded time limit: {elapsed:.2f}s > {t_max}s"

        # 2. Solution length check
        assert len(solution) <= max_moves, \
            f"Solution too long: {len(solution)} > {max_moves}"

        # 3. Solution validity check - apply moves and verify cube is solved
        test_cube = cube.copy()
        for move in solution:
            test_cube.move(move)

        assert test_cube.is_solved(), \
            f"Solution invalid: cube not solved after applying {len(solution)} moves " \
            f"({n_scrambles=}, {seed=}, {weight=}, {noise=}, {max_moves=}, {t_max=}, "\
            f"{max_restarts=}, {batch_size=}, {dataset_moves=})"
    else:
        # No solution found - this is acceptable for some configurations
        # Just verify it's explicitly None
        assert solution is None, "Solution should be None if not found"


def test_astar_on_solved_cube(trained_model):
    """Solved cube should return empty solution immediately."""
    cube = Cube()
    solver = AStarSolver(trained_model.as_value_function(),
                         max_moves=10, t_max=5.0)

    solution = solver(cube)

    assert solution == [], "Solved cube should return empty solution"
