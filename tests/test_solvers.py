"""
Tests for Rubik's Cube solvers.
"""
import pytest
import os
import torch
from cube import Cube
from solvers import solve_cube_bfs, BFSSolverDataset, SuperSolver
from cube_nn import CubeValueResNet, NNValueFunctionType


@pytest.fixture(scope="module")
def trained_model():
    """Load a trained CubeValueResNet model for testing."""
    model = CubeValueResNet()
    model_path = 'temp_models/resnet2.2/cube_value_resnet_iter_500.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(
            model_path, map_location='cpu', weights_only=True))
    else:
        pytest.skip(f"Model file not found: {model_path}")

    model.eval()
    model.set_value_function_type(NNValueFunctionType.STANDARD)
    return model


def verify_solution(cube: Cube, solution):
    """Helper function to verify a solution solves the cube."""
    if solution is None:
        return False

    test_cube = cube.copy()
    for move in solution:
        test_cube.move(move)

    return test_cube.is_solved()


# BFS Solver Tests
def test_bfs_solved_cube():
    """BFS should return empty solution for already solved cube."""
    cube = Cube()
    solution = solve_cube_bfs(cube, max_depth=5)
    assert solution == []


def test_bfs_simple_scramble():
    """BFS should solve a simple 3-move scramble."""
    cube = Cube(n_scramble_moves=3, scramble_seed=42)
    solution = solve_cube_bfs(cube, max_depth=5)
    assert solution is not None
    assert verify_solution(cube, solution)

    solution = solve_cube_bfs(cube, max_depth=2)
    assert solution is None  # Should not find solution within depth 2


# Dataset Solver Tests
def test_dataset_solver_solved_cube():
    """Dataset solver should return empty solution for solved cube."""
    solver = BFSSolverDataset(dataset_moves=3, max_bfs_depth=5)
    cube = Cube()
    solution = solver(cube)
    assert solution == []


def test_dataset_solver_simple_scramble():
    """Dataset solver should solve a simple scramble."""
    solver = BFSSolverDataset(dataset_moves=3, max_bfs_depth=5)
    cube = Cube(n_scramble_moves=3, scramble_seed=42)
    solution = solver(cube)
    assert solution is not None
    assert verify_solution(cube, solution)


# SuperSolver (A*) Tests
def test_supersolver_solved_cube(trained_model):
    """SuperSolver should return empty solution for solved cube."""
    cube = Cube()
    solver = SuperSolver(trained_model.as_value_function(),
                         max_moves=10, t_max=5.0)
    solution = solver(cube)
    assert solution == []


def test_supersolver_basic(trained_model):
    """SuperSolver with standard A* configuration should solve simple scrambles."""
    cube = Cube(n_scramble_moves=5, scramble_seed=42)
    solver = SuperSolver(
        trained_model.as_value_function(),
        weight=1.0,
        max_moves=20,
        t_max=10.0
    )
    solution = solver(cube)
    assert solution is not None
    assert len(solution) <= 20
    assert verify_solution(cube, solution)


def test_supersolver_with_dataset(trained_model):
    """SuperSolver with dataset assistance should solve efficiently."""
    cube = Cube(n_scramble_moves=4, scramble_seed=0)
    solver = SuperSolver(
        trained_model.as_value_function(),
        max_moves=15,
        t_max=10.0,
        dataset_moves=3
    )
    solution = solver(cube)
    assert solution is not None
    assert verify_solution(cube, solution)


def test_supersolver_weighted(trained_model):
    """SuperSolver with weighting should solve scrambles."""
    cube = Cube(n_scramble_moves=5, scramble_seed=1)
    solver = SuperSolver(
        trained_model.as_value_function(),
        weight=0.5,
        max_moves=20,
        t_max=10.0
    )
    solution = solver(cube)
    assert solution is not None
    assert verify_solution(cube, solution)


def test_supersolver_with_batching(trained_model):
    """SuperSolver with batching should solve scrambles."""
    cube = Cube(n_scramble_moves=5, scramble_seed=2)
    solver = SuperSolver(
        trained_model.as_value_function(),
        batch_size=10,
        max_moves=20,
        t_max=10.0
    )
    solution = solver(cube)
    assert solution is not None
    assert verify_solution(cube, solution)


def test_supersolver_with_restarts(trained_model):
    """SuperSolver with restarts should solve scrambles."""
    cube = Cube(n_scramble_moves=6, scramble_seed=3)
    solver = SuperSolver(
        trained_model.as_value_function(),
        max_moves=20,
        t_max=15.0,
        max_restarts=2
    )
    solution = solver(cube)
    assert solution is not None
    assert verify_solution(cube, solution)
