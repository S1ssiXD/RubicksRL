from cube_nn import CubeValueResNet, CubeValueTransformer
from training import train_on_value_dataset
from torch import optim
import matplotlib.pyplot as plt
import torch
from cube_nn import NNValueFunctionType
from concurrent.futures import ProcessPoolExecutor
import os
import torch.multiprocessing as mp

# Set multiprocessing start method to 'spawn' to avoid CUDA fork issues
mp.set_start_method('spawn', force=True)

device = "cuda"
name = "resnet2.2"
N_ROLLOUTS = 20000
N_MOVES_MAX = 20
N_EPOCHS = 1
BATCH_SIZE = 100
N_ITERATIONS = 500


def generate_dataset_worker(iteration, n_rollouts, n_moves_max):
    """Worker function to generate dataset on CPU while GPU trains

    This runs in a separate process to avoid GIL contention with the main training loop.
    Uses tqdm position 1 for progress bar to avoid conflicts with training (position 0).
    """
    from cube_datasets import TrainingValueDataset
    from cube_nn import cube_to_tensor_one_hot

    print(f"[Dataset Gen] Starting generation for iteration {iteration}")
    new_dataset = TrainingValueDataset.create_from_trajectories(
        cube_to_tensor_one_hot,
        n_rollouts,
        n_moves_max,
        device="cpu",
        tqdm_position=1  # Use position 1 for dataset generation
    )

    return new_dataset


if __name__ == '__main__':
    # Create directories for saving models and figures
    os.makedirs(f'temp_models/{name}', exist_ok=True)
    os.makedirs(f'figs/{name}', exist_ok=True)

    # Create a new net
    net = CubeValueResNet()
    # net = CubeValueTransformer()

    losses = []
    i = 0

    # Load an existing net to continue training
    # I0 = 500
    # net = CubeValueResNet()
    # net.load_state_dict(torch.load(
    #     f'temp_models/{name}/cube_value_resnet_iter_{I0}.pth'))
    # losses = torch.load(f'temp_models/{name}/losses_list_{I0}.pth')
    # i = I0

    # set the STANDARD value function for speed
    net.set_value_function_type(NNValueFunctionType.STANDARD)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=1) as executor:
        # Start generating the FIRST dataset in background
        print("Starting generation of first dataset...")
        future_dataset = executor.submit(
            generate_dataset_worker, i+1, N_ROLLOUTS, N_MOVES_MAX)

        for j in range(N_ITERATIONS):
            i += 1

            # Wait for current dataset to be ready and move it to GPU
            print(f"[Training] Waiting for dataset {i}...")
            current_dataset = future_dataset.result()
            current_dataset._inputs = current_dataset._inputs.to(device)
            current_dataset._targets = current_dataset._targets.to(device)

            # Immediately start generating the NEXT dataset in parallel (before training starts)
            if j < N_ITERATIONS - 1:  # Don't generate after the last iteration
                future_dataset = executor.submit(
                    generate_dataset_worker, i+1, N_ROLLOUTS, N_MOVES_MAX)

            # Train on current dataset (on GPU) - use position 0 for training progress bars
            print(f"[Training] Starting training iteration {i}")
            loss = train_on_value_dataset(net, current_dataset, optimizer, n_epochs=N_EPOCHS,
                                          batch_size=BATCH_SIZE, device=device, tqdm_position=0)
            losses.append(loss)

            # Save the model and plot of losses every 10 iterations
            if (i + 1) % 10 == 0:
                torch.save(
                    net.state_dict(), f'temp_models/{name}/cube_value_resnet_iter_{i+1}.pth')
                torch.save(losses, f'temp_models/{name}/losses_list_{i+1}.pth')
                # Skip initial loss for better visualization
                plt.plot(losses[1:])
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss Over Iterations')
                plt.savefig(f'figs/{name}/loss_plot_iter_{i+1}.png')
                plt.clf()

    print("Training completed!")
