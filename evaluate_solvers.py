"""Evaluate multiple solvers and hyperparameter combinations on random cubes.

Creates a dataset of N random cubes (scrambled by M moves) and evaluates
several solver classes from `solvers.py` over a grid of hyperparameters.
Results are saved to `evaluation_results.csv` and a short summary is printed.

Usage:
    python evaluate_solvers.py --model-path temp_models/resnet2/cube_value_resnet_iter_6800.pth

If `--model-path` is not provided the script will still run but the value-based
solvers will use a freshly initialized network (not recommended).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import product
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from cube import Cube
from cube_nn import CubeValueResNet, NNValueFunctionType
from solvers import (
    Solver,
    BFSSolverDataset,
    BFSSolverSimilarDataset,
    DepthValueSolver,
    NoisyWeightedAStarSolver,
    RandomSequenceSolver,
    RestartingWeightedAStarSolver,
    WeightedAStarSolver,
    nn_solver,
)


def generate_cubes(n: int, scramble_moves: int, seed: int = 1) -> List[Cube]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return [Cube(n_scramble_moves=scramble_moves) for _ in range(n)]


def evaluate_solver_on_cubes(solver: Solver, cubes: List[Cube], solver_name: str = "") -> Dict[str, Any]:
    times: List[float] = []
    lengths: List[int] = []
    solved = 0

    for c in tqdm(cubes, desc=solver_name, leave=False):
        cube = c.copy()

        t0 = time.time()
        sol = solver(cube)
        t1 = time.time()

        elapsed = t1 - t0
        times.append(elapsed)

        if sol is None:
            lengths.append(0)
        else:
            solved += 1
            lengths.append(len(sol))

    avg_time_all = mean(times) if times else float('inf')
    avg_time_solved = mean([t for t, l in zip(times, lengths) if l > 0]) if any(
        l > 0 for l in lengths) else float('inf')
    avg_len_solved = mean([l for l in lengths if l > 0]) if any(
        l > 0 for l in lengths) else float('inf')

    return {
        'n_cubes': len(list(cubes)),
        'solved_count': solved,
        'solved_fraction': solved / max(1, len(list(cubes))),
        'avg_time_all': avg_time_all,
        'avg_time_solved': avg_time_solved,
        'avg_len_solved': avg_len_solved,
        'times': times,
        'lengths': lengths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model state_dict (optional)')
    parser.add_argument('--n-cubes', type=int, default=100,
                        help='Number of random cubes to evaluate')
    parser.add_argument('--scramble-moves', type=int, default=50,
                        help='Number of scramble moves for each cube')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--out-csv', type=str,
                        default='evaluation_results.csv', help='CSV output filename')
    args = parser.parse_args()

    # Load network
    net = CubeValueResNet()
    if args.model_path is not None:
        try:
            net.load_state_dict(torch.load(args.model_path))
            print(f"Loaded model from {args.model_path}")
        except Exception as e:
            print(f"Warning: could not load model from {args.model_path}: {e}")

    # Use STANDARD value function for speed
    net.set_value_function_type(NNValueFunctionType.STANDARD)
    value_function = net.as_value_function()

    # # Build list of solver configurations to evaluate
    # solver_builders: List[Tuple[str, Callable[[],
    #                                           Callable[[Cube], Any]], Dict[str, Any]]] = []

    # # Weighted A* grid
    # for weight, max_moves, max_queue_size, t_max in product([0.08, 0.12, 0.18], [40, 50], [10000, 100000], [30, 60]):
    #     name = 'WeightedAStar'
    #     params = {'weight': weight, 'max_moves': max_moves,
    #               'max_queue_size': max_queue_size, 't_max': t_max}

    #     def make_builder(w=weight, mm=max_moves, mq=max_queue_size, tm=t_max):
    #         return lambda: WeightedAStarSolver(value_function, weight=w, max_moves=mm, max_queue_size=mq, t_max=tm)
    #     solver_builders.append((name, make_builder(), params))

    # # Noisy Weighted A*\n+
    # for weight, noise in product([0.12, 0.18], [0.1, 0.5]):
    #     name = 'NoisyWeightedAStar'
    #     params = {'weight': weight, 'noise': noise}

    #     def make_builder(w=weight, n=noise):
    #         return lambda: NoisyWeightedAStarSolver(value_function, weight=w, noise=n, max_moves=40, max_queue_size=100000, t_max=60)
    #     solver_builders.append((name, make_builder(), params))

    # # Restarting Weighted A*\n+
    # for t_max_per_attempt, max_restarts, restart_len in product([6, 12], [3, 9], [2, 3]):
    #     name = 'RestartingWeightedAStar'
    #     params = {'t_max_per_attempt': t_max_per_attempt,
    #               'max_restarts': max_restarts, 'restart_len': restart_len}

    #     def make_builder(t=t_max_per_attempt, r=max_restarts, rl=restart_len):
    #         return lambda: RestartingWeightedAStarSolver(value_function, weight=0.15, max_moves=40, max_queue_size=100000, t_max_per_attempt=t, max_restarts=r, restart_sequence_length=rl)
    #     solver_builders.append((name, make_builder(), params))

    # # Random Sequence Solver
    # for seq_len, max_seq in product([3, 4], [1000, 2000]):
    #     name = 'RandomSequenceSolver'
    #     params = {'sequence_length': seq_len, 'max_sequences': max_seq}

    #     def make_builder(sl=seq_len, ms=max_seq):
    #         return lambda: RandomSequenceSolver(value_function, sequence_length=sl, max_sequences=ms, goal_factor=0.6, t_max=60)
    #     solver_builders.append((name, make_builder(), params))

    # # DepthValueSolver
    # for max_depth in [2, 3]:
    #     name = 'DepthValueSolver'
    #     params = {'max_depth': max_depth}

    #     def make_builder(md=max_depth):
    #         return lambda: DepthValueSolver(value_function, max_depth=md, max_moves=40)
    #     solver_builders.append((name, make_builder(), params))

    # # BFSSolverDataset and BFSSolverSimilarDataset (vary dataset_moves)
    # for dataset_moves in [4, 5]:
    #     name = 'BFSSolverDataset'
    #     params = {'dataset_moves': dataset_moves}

    #     def make_builder(dm=dataset_moves):
    #         return lambda: BFSSolverDataset(dataset_moves=dm, max_bfs_depth=5)
    #     solver_builders.append((name, make_builder(), params))

    #     name2 = 'BFSSolverSimilarDataset'
    #     params2 = {'dataset_moves': dataset_moves}

    #     def make_builder2(dm=dataset_moves):
    #         return lambda: BFSSolverSimilarDataset(dataset_moves=dm, max_bfs_depth=5)
    #     solver_builders.append((name2, make_builder2(), params2))

    # # Pure NN greedy solver (follows gradient of value function)
    # for max_moves in [40, 50]:
    #     name = 'NNGreedy'
    #     params = {'max_moves': max_moves}

    #     def make_builder(mm=max_moves):
    #         return lambda: nn_solver(net, max_moves=mm)
    #     solver_builders.append((name, make_builder(), params))

    # # Build cube dataset
    # cubes = generate_cubes(args.n_cubes, args.scramble_moves, seed=args.seed)

    # # Evaluate all solvers
    # rows = []
    # print(
    #     f"Starting evaluation of {len(solver_builders)} solver configurations on {args.n_cubes} cubes...")
    # for idx, (name, builder, params) in enumerate(solver_builders, start=1):
    #     print(
    #         f"[{idx}/{len(solver_builders)}] Evaluating {name} with params {params}")
    #     solver = builder()
    #     # Evaluate on copies to avoid side effects
    #     results = evaluate_solver_on_cubes(solver, (c.copy() for c in cubes))
    #     row = {
    #         'solver': name,
    #         'params': json.dumps(params, sort_keys=True),
    #         'solved_count': results['solved_count'],
    #         'solved_fraction': results['solved_fraction'],
    #         'avg_time_all': results['avg_time_all'],
    #         'avg_time_solved': results['avg_time_solved'],
    #         'avg_len_solved': results['avg_len_solved'],
    #     }
    #     rows.append(row)

    # # Save CSV
    # fieldnames = ['solver', 'params', 'solved_count', 'solved_fraction',
    #               'avg_time_all', 'avg_time_solved', 'avg_len_solved']
    # with open(args.out_csv, 'w', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for r in rows:
    #         writer.writerow(r)

    # # Print best results sorted by solved_fraction desc, then avg_time_solved asc
    # rows_sorted = sorted(
    #     rows, key=lambda r: (-r['solved_fraction'], r['avg_time_solved'], r['avg_len_solved']))
    # print('\nTop 5 solver configurations:')
    # for r in rows_sorted[:5]:
    #     print(f"{r['solver']} {r['params']} -> solved {r['solved_count']}/{args.n_cubes}, avg_time_solved={r['avg_time_solved']:.3f}, avg_len_solved={r['avg_len_solved']}")

    max_moves = 40
    max_queue_size = 1000000
    t_max = 60

    solvers = {
        'NWA_0.18_0.15_30s': NoisyWeightedAStarSolver(value_function, weight=0.18, noise=0.15, max_moves=max_moves, max_queue_size=max_queue_size, t_max=30),
        'NWA_0.18_0.15_20s': NoisyWeightedAStarSolver(value_function, weight=0.18, noise=0.15, max_moves=max_moves, max_queue_size=max_queue_size, t_max=20),
        'NWA_0.18_0.15_15s': NoisyWeightedAStarSolver(value_function, weight=0.18, noise=0.15, max_moves=max_moves, max_queue_size=max_queue_size, t_max=15),
        'NWA_0.18_0.15_12s': NoisyWeightedAStarSolver(value_function, weight=0.18, noise=0.15, max_moves=max_moves, max_queue_size=max_queue_size, t_max=12),
    }

    cubes = generate_cubes(args.n_cubes, args.scramble_moves, seed=args.seed)

    rows = []
    for solver_name, solver in solvers.items():
        print(f"Evaluating solver: {solver_name}")
        results = evaluate_solver_on_cubes(
            solver, cubes, solver_name=solver_name)
        print(
            f"Solved {results['solved_count']}/{len(cubes)} cubes "
            f"({results['solved_fraction']*100:.2f}%), "
            f"avg_time_all={results['avg_time_all']:.3f}s, "
            f"avg_time_solved={results['avg_time_solved']:.3f}s, "
            f"avg_len_solved={results['avg_len_solved']}"
        )
        rows.append({
            'solver': solver_name,
            'solved_count': results['solved_count'],
            'solved_fraction': results['solved_fraction'],
            'avg_time_all': results['avg_time_all'],
            'avg_time_solved': results['avg_time_solved'],
            'avg_len_solved': results['avg_len_solved'],
        })

    # Save to CSV
    fieldnames = ['solver', 'solved_count', 'solved_fraction',
                  'avg_time_all', 'avg_time_solved', 'avg_len_solved']
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {args.out_csv}")


if __name__ == '__main__':
    main()
