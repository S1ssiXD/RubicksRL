""" Functions for evaluating the performance of a solver function. """
import numpy as np
from typing import Dict, Protocol, Optional
from solvers import Solver
from cube_datasets import CubeDataset, evaluation_easy_dataset, evaluation_medium_dataset, evaluation_hard_dataset
from tqdm import tqdm


class Evaluator(Protocol):
    """
    A protocol for an evaluator for a solver.

    Evaluator(solver: Solver) -> Dict
    """

    def __call__(self, solver: Solver) -> Dict:
        ...


def evaluate_solver(solver: Solver, dataset: CubeDataset, n: Optional[int] = None) -> Dict:
    """
    Evaluate a solver function on a given dataset of cubes.

    Args:
        solver (Solver): The solver function to evaluate.
        dataset (CubeDataset): A dataset of cubes to evaluate the solver on.
        n (Optional[int]): The number of cubes to evaluate. If None, evaluates all cubes in the dataset.
    Returns:
        Dict: A dictionary containing the evaluation results.
    """

    results = {}
    n_moves = []
    n_moves_diff = []
    solved = 0
    unsolved = 0
    failed = 0

    if n is not None:
        dataset = dataset[:n]

    # for cube, value in tqdm(dataset):
    for cube, value in dataset:
        solution = solver(cube.copy())
        if solution is None:
            unsolved += 1
            n_moves.append(np.nan)
            n_moves_diff.append(np.nan)
        else:
            # Check if the solution is valid
            cube_ = cube.copy()
            cube_.scramble(solution)
            if not cube_.is_solved():
                failed += 1
                n_moves.append(np.nan)
                n_moves_diff.append(np.nan)
            else:
                n_moves.append(len(solution))
                n_moves_diff.append(len(solution) - value)
                solved += 1
    results = {
        "solved": solved,
        "unsolved": unsolved,
        "failed": failed,
        "n_moves": n_moves,
        "n_moves_diff": n_moves_diff,
    }
    return results


def full_evaluation(solver: Solver, n: Optional[int] = None) -> Dict:
    """
    Perform a full evaluation of the solver on all datasets.

    Args:
        solver (Solver): The solver function to evaluate.
        n (Optional[int]): The number of cubes to evaluate in each dataset. If None, evaluates all cubes.
    Returns:
        Dict[str, Dict]: A dictionary containing evaluation results for each dataset.
    """
    results = {}

    results['easy'] = evaluate_solver(solver, evaluation_easy_dataset, n)
    results['medium'] = evaluate_solver(solver, evaluation_medium_dataset, n)
    results['hard'] = evaluate_solver(solver, evaluation_hard_dataset, n)

    return results


def easy_evaluation(solver: Solver) -> Dict:
    """
    Evaluate the solver on the easy dataset.

    Args:
        solver (Solver): The solver function to evaluate.
    Returns:
        Dict: Evaluation results for the easy dataset.
    """
    return evaluate_solver(solver, evaluation_easy_dataset)


def medium_evaluation(solver: Solver) -> Dict:
    """
    Evaluate the solver on the medium dataset.

    Args:
        solver (Solver): The solver function to evaluate.
    Returns:
        Dict: Evaluation results for the medium dataset.
    """

    return evaluate_solver(solver, evaluation_medium_dataset)


def hard_evaluation(solver: Solver) -> Dict:
    """
    Evaluate the solver on the hard dataset.

    Args:
        solver (Solver): The solver function to evaluate.
    Returns:
        Dict: Evaluation results for the hard dataset.
    """
    return evaluate_solver(solver, evaluation_hard_dataset)
