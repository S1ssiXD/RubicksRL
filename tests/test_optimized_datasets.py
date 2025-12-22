"""Tests for optimized dataset creation and training."""

import pytest
import torch
import numpy as np
from cube import Cube
from cube_nn import CubeValueNNFC, cube_to_tensor_direct, cube_to_tensor_one_hot
from cube_datasets import TrainingValueDataset
from training import train_on_value_dataset
import torch.optim as optim


class TestTrainingValueDataset:
    """Tests for the optimized TrainingValueDataset class."""

    def test_init_with_tensors(self):
        """Test initialization with pre-allocated tensors."""
        inputs = torch.randn(100, 54)
        targets = torch.randn(100, 1)

        dataset = TrainingValueDataset(inputs=inputs, targets=targets)

        assert len(dataset) == 100
        assert dataset._inputs is not None
        assert dataset._targets is not None
        assert dataset._dataset is None

    def test_init_with_legacy_format(self):
        """Test initialization with legacy list format."""
        legacy_data = [(torch.randn(54), torch.tensor([1.0]))
                       for _ in range(50)]

        dataset = TrainingValueDataset(dataset=legacy_data)

        assert len(dataset) == 50
        assert dataset._dataset is not None
        assert dataset._inputs is None
        assert dataset._targets is None

    def test_init_raises_without_data(self):
        """Test that initialization fails without any data."""
        with pytest.raises(ValueError, match="Either dataset or"):
            TrainingValueDataset()

    def test_getitem_tensor_format(self):
        """Test __getitem__ with tensor format."""
        inputs = torch.randn(10, 54)
        targets = torch.randn(10, 1)
        dataset = TrainingValueDataset(inputs=inputs, targets=targets)

        inp, tgt = dataset[5]

        assert torch.equal(inp, inputs[5])
        assert torch.equal(tgt, targets[5])

    def test_getitem_legacy_format(self):
        """Test __getitem__ with legacy format."""
        legacy_data = [(torch.randn(54), torch.tensor([float(i)]))
                       for i in range(10)]
        dataset = TrainingValueDataset(dataset=legacy_data)

        inp, tgt = dataset[3]

        assert torch.equal(inp, legacy_data[3][0])
        assert torch.equal(tgt, legacy_data[3][1])

    def test_to_device_cpu_to_cuda(self):
        """Test moving dataset from CPU to CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        inputs = torch.randn(10, 54)
        targets = torch.randn(10, 1)
        dataset = TrainingValueDataset(inputs=inputs, targets=targets)

        dataset.to("cuda")

        assert dataset._inputs.device.type == "cuda"
        assert dataset._targets.device.type == "cuda"

    def test_to_device_cuda_to_cpu(self):
        """Test moving dataset from CUDA to CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        inputs = torch.randn(10, 54).cuda()
        targets = torch.randn(10, 1).cuda()
        dataset = TrainingValueDataset(inputs=inputs, targets=targets)

        dataset.to("cpu")

        assert dataset._inputs.device.type == "cpu"
        assert dataset._targets.device.type == "cpu"

    def test_to_device_legacy_format_conversion(self):
        """Test that to() converts legacy format to tensor format."""
        legacy_data = [(torch.randn(54), torch.tensor([1.0]))
                       for _ in range(10)]
        dataset = TrainingValueDataset(dataset=legacy_data)

        assert dataset._dataset is not None
        assert dataset._inputs is None

        dataset.to("cpu")

        assert dataset._inputs is not None
        assert dataset._targets is not None
        assert dataset._dataset is None

    def test_pin_memory(self):
        """Test pinning memory for CPU tensors."""
        inputs = torch.randn(10, 54)
        targets = torch.randn(10, 1)
        dataset = TrainingValueDataset(inputs=inputs, targets=targets)

        dataset.pin_memory()

        assert dataset._inputs.is_pinned()
        assert dataset._targets.is_pinned()

    def test_pin_memory_cuda_no_op(self):
        """Test that pin_memory is no-op for CUDA tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        inputs = torch.randn(10, 54).cuda()
        targets = torch.randn(10, 1).cuda()
        dataset = TrainingValueDataset(inputs=inputs, targets=targets)

        # Should not raise, just be a no-op
        dataset.pin_memory()

    def test_create_balanced_cpu(self):
        """Test creating a balanced dataset on CPU."""
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=3,
            n_cubes_per_dataset=10,
            seed=42,
            device="cpu"
        )

        expected_size = (3 + 1) * 10  # (n_moves_max + 1) * n_cubes_per_dataset
        assert len(dataset) == expected_size
        assert dataset._inputs is not None
        assert dataset._targets is not None
        assert dataset._inputs.device.type == "cpu"
        assert dataset._targets.device.type == "cpu"

    def test_create_balanced_cuda(self):
        """Test creating a balanced dataset on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=3,
            n_cubes_per_dataset=10,
            seed=42,
            device="cuda"
        )

        expected_size = (3 + 1) * 10
        assert len(dataset) == expected_size
        assert dataset._inputs.device.type == "cuda"
        assert dataset._targets.device.type == "cuda"

    def test_create_balanced_correct_shapes(self):
        """Test that created dataset has correct tensor shapes."""
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=5,
            seed=42,
            device="cpu"
        )

        inp, tgt = dataset[0]

        assert inp.shape == (54,)  # cube_to_tensor_direct produces (54,)
        assert tgt.shape == (1,)

    def test_create_balanced_one_hot(self):
        """Test creating dataset with one-hot encoding."""
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_one_hot,
            n_moves_max=2,
            n_cubes_per_dataset=5,
            seed=42,
            device="cpu"
        )

        inp, tgt = dataset[0]

        assert inp.shape == (54, 6)  # one-hot encoding
        assert tgt.shape == (1,)

    def test_create_balanced_values_in_range(self):
        """Test that target values are in expected range."""
        n_moves_max = 5
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=n_moves_max,
            n_cubes_per_dataset=20,
            seed=42,
            device="cpu"
        )

        all_targets = dataset._targets.numpy().flatten()

        assert all_targets.min() >= 0
        assert all_targets.max() <= n_moves_max

    def test_create_balanced_deterministic(self):
        """Test that same seed produces same dataset."""
        dataset1 = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=3,
            n_cubes_per_dataset=10,
            seed=123,
            device="cpu"
        )

        dataset2 = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=3,
            n_cubes_per_dataset=10,
            seed=123,
            device="cpu"
        )

        assert torch.allclose(dataset1._inputs, dataset2._inputs)
        assert torch.allclose(dataset1._targets, dataset2._targets)

    def test_create_from_bellman_equation(self):
        """Test creating dataset from Bellman equation."""
        # Create a simple model
        model = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])
        model.eval()

        dataset = TrainingValueDataset.create_from_bellman_equation(
            model,
            n_moves_max=3,
            n_cubes_per_move=10,
            device="cpu"
        )

        assert len(dataset) > 0
        assert dataset._inputs is not None
        assert dataset._targets is not None

    def test_batch_conversion_efficiency(self):
        """Test that batch conversion works correctly."""
        # Create dataset and verify all cubes were converted
        n_moves_max = 4
        n_cubes_per_dataset = 20

        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=n_moves_max,
            n_cubes_per_dataset=n_cubes_per_dataset,
            seed=42,
            device="cpu"
        )

        expected_size = (n_moves_max + 1) * n_cubes_per_dataset
        assert len(dataset) == expected_size

        # Verify no NaN or inf values from conversion
        assert not torch.isnan(dataset._inputs).any()
        assert not torch.isinf(dataset._inputs).any()
        assert not torch.isnan(dataset._targets).any()
        assert not torch.isinf(dataset._targets).any()

    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=10,
            seed=42,
            device="cpu"
        )

        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        for batch_inputs, batch_targets in dataloader:
            assert batch_inputs.shape[0] <= 5
            assert batch_targets.shape[0] <= 5
            assert batch_inputs.shape[1] == 54
            assert batch_targets.shape[1] == 1
            break  # Just test first batch

    def test_create_from_trajectories_cpu(self):
        """Test creating dataset from random trajectories on CPU."""
        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor_direct,
            n_trajectories=10,
            n_moves=5,
            seed=42,
            device="cpu"
        )

        # Should have exactly n_trajectories * n_moves samples (no solved state)
        assert len(dataset) == 10 * 5

        # Check tensor shapes
        inp, tgt = dataset[0]
        assert inp.shape == (54,)
        assert tgt.shape == (1,)

        # Check that values are in valid range (1 to n_moves, no 0)
        assert dataset._targets.min() >= 1
        assert dataset._targets.max() <= 5

    def test_create_from_trajectories_cuda(self):
        """Test creating dataset from random trajectories on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor_direct,
            n_trajectories=10,
            n_moves=5,
            seed=42,
            device="cuda"
        )

        assert len(dataset) > 0
        assert dataset._inputs.device.type == "cuda"
        assert dataset._targets.device.type == "cuda"

    def test_create_from_trajectories_deterministic(self):
        """Test that same seed produces same trajectory dataset."""
        dataset1 = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor_direct,
            n_trajectories=5,
            n_moves=4,
            seed=123,
            device="cpu"
        )

        dataset2 = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor_direct,
            n_trajectories=5,
            n_moves=4,
            seed=123,
            device="cpu"
        )

        assert len(dataset1) == len(dataset2)
        assert torch.allclose(dataset1._inputs, dataset2._inputs)
        assert torch.allclose(dataset1._targets, dataset2._targets)

    def test_create_from_trajectories_excludes_solved_state(self):
        """Test that trajectory dataset excludes the solved state."""
        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor_direct,
            n_trajectories=5,
            n_moves=3,
            seed=42,
            device="cpu"
        )

        # Should NOT have any states with distance 0 (solved state excluded)
        has_solved = (dataset._targets == 0).any()
        assert not has_solved.item(), "Dataset should exclude solved state (no distance 0)"

    def test_create_from_trajectories_sequential_distances(self):
        """Test that trajectory includes sequential distances."""
        dataset = TrainingValueDataset.create_from_trajectories(
            cube_to_tensor_direct,
            n_trajectories=10,
            n_moves=5,
            seed=42,
            device="cpu"
        )

        # Should have distances from 1 to 5 (no 0 since solved state excluded)
        unique_distances = torch.unique(dataset._targets).cpu().numpy()
        # Should have all distances from 1 to 5
        assert 0 not in unique_distances, "Solved state (0) should be excluded"
        assert all(1 <= d <= 5 for d in unique_distances)
        # Likely to have all values 1-5 with 10 trajectories
        assert len(unique_distances) >= 3


class TestTrainingFunction:
    """Tests for the optimized training function."""

    def test_train_on_value_dataset_cpu(self):
        """Test training on CPU."""
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=10,
            seed=42,
            device="cpu"
        )

        net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        initial_loss = train_on_value_dataset(
            net, dataset, optimizer,
            batch_size=5,
            n_epochs=2,
            device="cpu"
        )

        assert isinstance(initial_loss, float)
        assert initial_loss > 0

    def test_train_on_value_dataset_cuda(self):
        """Test training on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=10,
            seed=42,
            device="cuda"
        )

        net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        initial_loss = train_on_value_dataset(
            net, dataset, optimizer,
            batch_size=5,
            n_epochs=2,
            device="cuda"
        )

        assert isinstance(initial_loss, float)
        assert initial_loss > 0
        # Verify model is on CUDA
        assert next(net.parameters()).device.type == "cuda"

    def test_train_on_value_dataset_reduces_loss(self):
        """Test that training actually reduces loss."""
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=50,
            seed=42,
            device="cpu"
        )

        net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[64, 64])
        optimizer = optim.Adam(net.parameters(), lr=1e-2)

        # Get initial loss
        initial_loss = train_on_value_dataset(
            net, dataset, optimizer,
            batch_size=10,
            n_epochs=0,  # Just evaluate
            device="cpu"
        )

        # Train for a few epochs
        train_on_value_dataset(
            net, dataset, optimizer,
            batch_size=10,
            n_epochs=5,
            device="cpu"
        )

        # Evaluate final loss
        net.eval()
        criterion = torch.nn.MSELoss()
        final_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(dataset), 10):
                batch_inputs = dataset._inputs[i:i+10]
                batch_targets = dataset._targets[i:i+10]
                outputs = net(batch_inputs)
                final_loss += criterion(outputs, batch_targets).item()
        final_loss /= (len(dataset) // 10)

        # Loss should decrease with training
        assert final_loss < initial_loss

    def test_train_cpu_dataset_on_cuda(self):
        """Test training with CPU dataset on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Dataset on CPU
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=10,
            seed=42,
            device="cpu"
        )

        net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        # Train on CUDA (should transfer data)
        initial_loss = train_on_value_dataset(
            net, dataset, optimizer,
            batch_size=5,
            n_epochs=2,
            device="cuda"
        )

        assert isinstance(initial_loss, float)
        assert next(net.parameters()).device.type == "cuda"

    def test_train_with_pinned_memory(self):
        """Test training with pinned memory optimization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=10,
            seed=42,
            device="cpu"
        )

        # Pin memory
        dataset.pin_memory()
        assert dataset._inputs.is_pinned()

        net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        initial_loss = train_on_value_dataset(
            net, dataset, optimizer,
            batch_size=5,
            n_epochs=2,
            device="cuda"
        )

        assert isinstance(initial_loss, float)

    def test_train_different_batch_sizes(self):
        """Test training with different batch sizes."""
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=20,
            seed=42,
            device="cpu"
        )

        for batch_size in [1, 5, 10, 20]:
            net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])
            optimizer = optim.Adam(net.parameters(), lr=1e-3)

            initial_loss = train_on_value_dataset(
                net, dataset, optimizer,
                batch_size=batch_size,
                n_epochs=1,
                device="cpu"
            )

            assert isinstance(initial_loss, float)
            assert initial_loss > 0


class TestOptimizationPerformance:
    """Tests to verify optimization correctness."""

    def test_tensor_format_benefits(self):
        """Test the key benefits of tensor format: GPU storage and memory efficiency."""
        # Create both formats
        size = 1000
        inputs = torch.randn(size, 54)
        targets = torch.randn(size, 1)

        dataset_tensor = TrainingValueDataset(inputs=inputs, targets=targets)
        legacy_data = [(inputs[i].clone(), targets[i].clone())
                       for i in range(size)]
        dataset_legacy = TrainingValueDataset(dataset=legacy_data)

        # Test 1: Tensor format can move to GPU efficiently
        if torch.cuda.is_available():
            dataset_tensor_gpu = TrainingValueDataset(
                inputs=inputs.clone(), targets=targets.clone())
            dataset_tensor_gpu.to("cuda")
            assert dataset_tensor_gpu._inputs.device.type == "cuda"
            assert dataset_tensor_gpu._targets.device.type == "cuda"

        # Test 2: Tensor format uses less Python object overhead
        # The tensors are stored contiguously, not as individual objects
        assert dataset_tensor._inputs is not None
        assert dataset_tensor._targets is not None
        assert dataset_tensor._dataset is None  # No list of tuples

        # Legacy format needs list of tuples
        assert dataset_legacy._dataset is not None
        assert len(dataset_legacy._dataset) == size

    def test_gpu_dataset_no_transfer_overhead(self):
        """Test that GPU dataset has minimal transfer overhead."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        import time

        # Small dataset for quick test
        dataset_gpu = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=50,
            seed=42,
            device="cuda"
        )

        dataset_cpu = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=2,
            n_cubes_per_dataset=50,
            seed=42,
            device="cpu"
        )

        net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])

        # Time GPU dataset training (1 epoch)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        start = time.time()
        train_on_value_dataset(net, dataset_gpu, optimizer,
                               batch_size=10, n_epochs=1, device="cuda")
        gpu_time = time.time() - start

        # Time CPU dataset training (1 epoch)
        net = CubeValueNNFC(cube_to_tensor_direct, hidden_size=[32])
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        start = time.time()
        train_on_value_dataset(net, dataset_cpu, optimizer,
                               batch_size=10, n_epochs=1, device="cuda")
        cpu_time = time.time() - start

        # GPU dataset should be faster (or at least not much slower)
        print(
            f"\nGPU dataset time: {gpu_time:.3f}s, CPU dataset time: {cpu_time:.3f}s")
        assert gpu_time <= cpu_time * 1.2  # Allow some variance

    def test_batch_conversion_preserves_data(self):
        """Test that batch conversion produces same results as individual conversion."""
        cubes = [Cube(n_scramble_moves=i, scramble_seed=42) for i in range(10)]

        # Individual conversion
        individual_tensors = [cube_to_tensor_direct(cube) for cube in cubes]
        individual_stacked = torch.stack(
            [t.squeeze(0) for t in individual_tensors])

        # Batch conversion
        batch_tensor = cube_to_tensor_direct(cubes)

        # Should produce same results
        assert torch.allclose(individual_stacked, batch_tensor, atol=1e-6)

    def test_dataset_values_consistent(self):
        """Test that dataset values are consistent with cube scramble distances."""
        dataset = TrainingValueDataset.create_balanced(
            cube_to_tensor_direct,
            n_moves_max=3,
            n_cubes_per_dataset=50,
            seed=42,
            device="cpu"
        )

        # All values should be reasonable
        targets = dataset._targets.numpy().flatten()
        assert (targets >= 0).all()
        assert (targets <= 20).all()  # Max optimal distance is 20

        # Should have variety of values
        unique_values = np.unique(targets)
        assert len(unique_values) >= 3  # At least 0, 1, 2, 3 move solutions
