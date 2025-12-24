""" This script sets the project goals and checks if they are met. 
The project can be considered complete if all benchmarks pass successfully. """
from cube_nn import CubeValueResNet
import torch
from cube import Cube
from solvers import SuperSolver
from cube_nn import NNValueFunctionType
from tqdm import tqdm
import time

### --------------- Project requirements ---------------- ###
N_CUBES = 1000           # evaluate on 1000 cubes
SUCCESS_RATE = 1.0       # a solution must be found for all cubes
N_SCRAMBLE_MOVES = 50    # cubes must be scrambled completely
MAX_SOLUTION_MOVES = 40  # solutions must be shorther than the scramble sequence
MAX_TIME = 60.0          # solutions must be found in less than one minute
### ----------------------------------------------------- ###

# Generate evaluation cubes
cubes = [Cube(n_scramble_moves=N_SCRAMBLE_MOVES, scramble_seed=i)
         for i in range(N_CUBES)]

# Load trained model
net = CubeValueResNet()
# net.load_state_dict(torch.load(
#     f'temp_models/resnet2/cube_value_resnet_iter_7000.pth'))
# net.load_state_dict(torch.load(
#     'temp_models/resnet2.1/cube_value_resnet_iter_500.pth'))
net.load_state_dict(torch.load(
    'temp_models/resnet2.2/cube_value_resnet_iter_500.pth'))
net = net.to("cuda")
net.set_value_function_type(NNValueFunctionType.STANDARD)

# Define solver
solver = SuperSolver(net.as_value_function(), weight=0.25, noise=0.0,
                     max_moves=40, max_queue_size=1000000, t_max=60, max_restarts=2, batch_size=25, seed=42, dataset_moves=6)


def verify_solution(cube, solution):
    """Verify that the solution correctly solves the cube."""
    test_cube = cube.copy()
    for move in solution:
        test_cube.move(move)
    return test_cube.is_solved()


solved_cubes = 0
sol_lengths = []
sol_times = []
idx_not_solved = []
for i, cube in enumerate(tqdm(cubes, desc=f'Evaluating on cubes', leave=False)):
    start_time = time.time()
    solution = solver(cube)
    end_time = time.time()
    if solution is not None and not verify_solution(cube, solution):
        solution = None  # Mark as not solved if verification fails
        print(f'Cube {i} solution verification failed.')
    if solution is not None:
        solved_cubes += 1
        sol_lengths.append(len(solution))
        sol_times.append(end_time - start_time)
    else:
        idx_not_solved.append(i)

print(f'Solved: {solved_cubes}/{N_CUBES}')
print(f'Fraction solved: {solved_cubes / N_CUBES}')
print(
    f'Average solution length: {sum(sol_lengths) / len(sol_lengths) if sol_lengths else 0}')
print(
    f'Average solution time: {sum(sol_times) / len(sol_times) if sol_times else 0}')
print(
    f'Max solution length: {max(sol_lengths) if sol_lengths else 0}')
print(
    f'Max solution time: {max(sol_times) if sol_times else 0}')
print('Indices of not solved cubes:', idx_not_solved)
print()


success = solved_cubes / N_CUBES >= SUCCESS_RATE and max(
    sol_lengths) <= MAX_SOLUTION_MOVES and max(sol_times) <= MAX_TIME
print('##########################################')
print(f'Project completed: {success}')
