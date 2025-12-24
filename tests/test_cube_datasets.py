import pytest
import numpy as np
import torch
from cube import Cube
from cube_nn import (
    cube_to_tensor_direct,
    cube_to_tensor_one_hot,
    CubeValueNNFC,
)
from cube_datasets import (
    generate_full_optimal_value_dataset,
    get_full_optimal_value_dataset,
    save_cube_dataset,
    load_cube_dataset,
    TrainingValueDataset,
)
import tempfile
import os


# Parametrized fixtures for cube_to_tensor functions (similar to test_cube_nn)
@pytest.fixture(params=[
    (cube_to_tensor_direct, (54,)),
    (cube_to_tensor_one_hot, (54, 6)),
], ids=["direct", "one_hot"])
def cube_to_tensor_with_shape(request):
    """Fixture providing cube_to_tensor function and expected tensor shape."""
    return request.param


class TestDatasetSaveLoad:
    """Test saving and loading datasets."""

    def test_save_and_load_dataset(self):
        """Test that we can save and load a dataset."""
        # Generate small dataset
        dataset = generate_full_optimal_value_dataset(n_moves=2)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
            temp_path = f.name

        try:
            save_cube_dataset(dataset, temp_path)
            loaded_dataset = load_cube_dataset(temp_path)

            # Verify same size
            assert len(dataset) == len(loaded_dataset)

            # Verify same cubes and values
            for cube, value in dataset.items():
                assert cube in loaded_dataset
                assert loaded_dataset[cube] == value
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestFullOptimalValueDataset:
    """Test full optimal value dataset generation."""

    def test_generate_full_optimal_value_dataset_small(self):
        """Test generating a small optimal value dataset."""
        dataset = generate_full_optimal_value_dataset(n_moves=2)

        # Should have solved state + all 1-move states + all 2-move states
        assert len(dataset) > 0

        # Solved cube should have value 0
        solved_cube = Cube()
        assert solved_cube in dataset
        assert dataset[solved_cube] == 0

        # All values should be between 0 and 2
        values = list(dataset.values())
        assert all(0 <= v <= 2 for v in values)

        # Should have cubes at each depth
        assert 0 in values  # solved state
        assert 1 in values  # 1-move states
        assert 2 in values  # 2-move states

    def test_generate_full_optimal_value_dataset_structure(self):
        """Test that dataset has correct structure with neighbors."""
        dataset = generate_full_optimal_value_dataset(n_moves=3)

        solved_cube = Cube()
        neighbors = solved_cube.get_all_neighbors()

        for neighbor in neighbors:
            assert neighbor in dataset
            assert dataset[neighbor] == 1

    def test_get_full_optimal_value_dataset_loads_existing(self, monkeypatch, capsys):
        """Test that get_full_optimal_value_dataset loads from cache when available."""
        n_moves = 2

        # Create a temporary directory and use it as the datasets/full/ folder
        with tempfile.TemporaryDirectory() as temp_dir:
            # Monkey patch the function to use temp directory
            original_get = get_full_optimal_value_dataset

            def patched_get(n_moves):
                import cube_datasets
                original_load = cube_datasets.load_cube_dataset
                original_save = cube_datasets.save_cube_dataset

                def temp_load(path):
                    # Replace path with temp path
                    temp_path = os.path.join(temp_dir, os.path.basename(path))
                    return original_load(temp_path)

                def temp_save(cubes, path):
                    # Replace path with temp path
                    temp_path = os.path.join(temp_dir, os.path.basename(path))
                    return original_save(cubes, temp_path)

                monkeypatch.setattr(
                    cube_datasets, 'load_cube_dataset', temp_load)
                monkeypatch.setattr(
                    cube_datasets, 'save_cube_dataset', temp_save)

                return original_get(n_moves)

            # First call should generate and save
            dataset1 = patched_get(n_moves)
            output1 = capsys.readouterr()
            assert "generating it" in output1.out.lower() or "Saved full optimal" in output1.out

            # Second call should load from cache
            dataset2 = patched_get(n_moves)
            output2 = capsys.readouterr()
            assert "Loaded full optimal" in output2.out

            # Datasets should be identical
            assert len(dataset1) == len(dataset2)
            for cube in dataset1:
                assert cube in dataset2
                assert dataset1[cube] == dataset2[cube]

    def test_get_full_optimal_value_dataset_generates_if_missing(self, capsys):
        """Test that get_full_optimal_value_dataset generates dataset if not found."""
        n_moves = 2

        # Use a non-existent path by monkeypatching
        with tempfile.TemporaryDirectory() as temp_dir:
            import cube_datasets
            original_load = cube_datasets.load_cube_dataset
            original_save = cube_datasets.save_cube_dataset
            original_generate = cube_datasets.generate_full_optimal_value_dataset

            temp_path = os.path.join(temp_dir, f"full_{n_moves}.pt")

            def temp_load(path):
                return original_load(temp_path)

            def temp_save(cubes, path):
                return original_save(cubes, temp_path)

            # Temporarily patch
            cube_datasets.load_cube_dataset = temp_load
            cube_datasets.save_cube_dataset = temp_save

            try:
                # Should generate since file doesn't exist
                dataset = get_full_optimal_value_dataset(n_moves)
                output = capsys.readouterr()

                # Should have generated
                assert "generating it" in output.out.lower() or "not found" in output.out.lower()
                assert len(dataset) > 0

                # File should now exist
                assert os.path.exists(temp_path)
            finally:
                # Restore original functions
                cube_datasets.load_cube_dataset = original_load
                cube_datasets.save_cube_dataset = original_save
                cube_datasets.generate_full_optimal_value_dataset = original_generate


class TestTrainingValueDatasetFromTrajectories:
    """Test TrainingValueDataset.create_from_trajectories."""

    def test_create_from_trajectories_basic(self, cube_to_tensor_with_shape):
        """Test basic trajectory dataset creation with parametrized encodings."""
        cube_to_tensor, expected_shape = cube_to_tensor_with_shape

        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor=cube_to_tensor,
            n_trajectories=10,
            n_moves=5,
            seed=42,
            device="cpu"
        )

        # Should have n_trajectories * n_moves samples
        assert len(dataset) == 10 * 5

        # Check data shapes
        inputs, targets = dataset[0]
        assert inputs.shape == expected_shape
        assert targets.shape == (1,)

        # Check target values are in expected range
        for i in range(len(dataset)):
            _, target = dataset[i]
            assert 1 <= target.item() <= 5

    def test_create_from_trajectories_reproducibility(self, cube_to_tensor_with_shape):
        """Test that same seed produces same dataset."""
        cube_to_tensor, _ = cube_to_tensor_with_shape

        dataset1 = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor=cube_to_tensor,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        dataset2 = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor=cube_to_tensor,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        # Should be identical
        assert len(dataset1) == len(dataset2)
        for i in range(len(dataset1)):
            inp1, tgt1 = dataset1[i]
            inp2, tgt2 = dataset2[i]
            assert torch.allclose(inp1, inp2)
            assert torch.allclose(tgt1, tgt2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_from_trajectories_cuda(self):
        """Test trajectory dataset creation on CUDA."""
        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor=cube_to_tensor_direct,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cuda"
        )

        # Data should be on CUDA
        inputs, targets = dataset[0]
        assert inputs.device.type == "cuda"
        assert targets.device.type == "cuda"


class TestTrainingValueDatasetFromBellman:
    """Test TrainingValueDataset.create_from_bellman_equation."""

    def test_create_from_bellman_basic_cpu(self, cube_to_tensor_with_shape):
        """Test basic Bellman dataset creation on CPU with parametrized encodings."""
        cube_to_tensor, expected_shape = cube_to_tensor_with_shape

        # Create simple model
        model = CubeValueNNFC(cube_to_tensor=cube_to_tensor, hidden_size=[32])

        dataset = TrainingValueDataset.create_from_bellman_equation(
            model=model,
            n_trajectories=10,
            n_moves=5,
            seed=42,
            device="cpu"
        )

        # Should have n_trajectories * n_moves samples
        assert len(dataset) == 10 * 5

        # Check data shapes
        inputs, targets = dataset[0]
        assert inputs.shape == expected_shape
        assert targets.shape == (1,)

        # Check that dataset was created (targets exist and are finite)
        for i in range(len(dataset)):
            _, target = dataset[i]
            assert torch.isfinite(target).all()
            # Target should be a scalar
            assert target.numel() == 1

    def test_create_from_bellman_values_bounded(self):
        """Test that Bellman values are bounded by actual distance."""
        # Create simple model
        model = CubeValueNNFC(
            cube_to_tensor=cube_to_tensor_direct, hidden_size=[32])

        dataset = TrainingValueDataset.create_from_bellman_equation(
            model=model,
            n_trajectories=10,
            n_moves=5,
            seed=42,
            device="cpu"
        )

        # Bellman values should be at most the actual move count
        for i in range(len(dataset)):
            _, target = dataset[i]
            # Since we have n_moves=5, max target should be 5
            assert target.item() <= 5

    def test_create_from_bellman_reproducibility(self, cube_to_tensor_with_shape):
        """Test that same seed produces same Bellman dataset."""
        cube_to_tensor, _ = cube_to_tensor_with_shape

        model = CubeValueNNFC(cube_to_tensor=cube_to_tensor, hidden_size=[32])

        # Set model to eval and initialize weights deterministically
        model.eval()
        torch.manual_seed(42)

        dataset1 = TrainingValueDataset.create_from_bellman_equation(
            model=model,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        dataset2 = TrainingValueDataset.create_from_bellman_equation(
            model=model,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        # Should be identical
        assert len(dataset1) == len(dataset2)
        for i in range(len(dataset1)):
            inp1, tgt1 = dataset1[i]
            inp2, tgt2 = dataset2[i]
            assert torch.allclose(inp1, inp2)
            assert torch.allclose(tgt1, tgt2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_from_bellman_cuda(self):
        """Test Bellman dataset creation with model on CUDA."""
        # Create model on CUDA
        model = CubeValueNNFC(
            cube_to_tensor=cube_to_tensor_direct, hidden_size=[32])
        model = model.to("cuda")

        dataset = TrainingValueDataset.create_from_bellman_equation(
            model=model,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cuda"
        )

        # Data should be on CUDA
        inputs, targets = dataset[0]
        assert inputs.device.type == "cuda"
        assert targets.device.type == "cuda"

        # Check we can move back to CPU
        dataset_cpu = dataset.to("cpu")
        inputs_cpu, targets_cpu = dataset_cpu[0]
        assert inputs_cpu.device.type == "cpu"
        assert targets_cpu.device.type == "cpu"

    def test_create_from_bellman_different_hidden_sizes(self, cube_to_tensor_with_shape):
        """Test Bellman dataset with different hidden layer sizes."""
        cube_to_tensor, expected_shape = cube_to_tensor_with_shape

        # Test with deeper network
        model = CubeValueNNFC(
            cube_to_tensor=cube_to_tensor, hidden_size=[64, 32])

        dataset = TrainingValueDataset.create_from_bellman_equation(
            model=model,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        assert len(dataset) == 5 * 3
        inputs, targets = dataset[0]
        assert inputs.shape == expected_shape


class TestTrainingValueDatasetMethods:
    """Test TrainingValueDataset utility methods."""

    def test_dataset_to_device(self):
        """Test moving dataset between devices."""
        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor=cube_to_tensor_direct,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        # Initially on CPU
        inputs, targets = dataset[0]
        assert inputs.device.type == "cpu"

        # Can move back to CPU (no-op but should work)
        dataset.to("cpu")
        inputs, targets = dataset[0]
        assert inputs.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dataset_to_cuda(self):
        """Test moving dataset to CUDA."""
        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor=cube_to_tensor_direct,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        # Move to CUDA
        dataset.to("cuda")
        inputs, targets = dataset[0]
        assert inputs.device.type == "cuda"

    def test_dataset_pin_memory(self):
        """Test pinning dataset memory."""
        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor=cube_to_tensor_direct,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        # Pin memory
        dataset.pin_memory()

        # Should still be accessible
        inputs, targets = dataset[0]
        assert inputs.device.type == "cpu"
