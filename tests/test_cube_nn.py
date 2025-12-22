import pytest
import numpy as np
import torch
from cube import Cube
from cube_nn import (
    cube_to_tensor_direct,
    cube_to_tensor_one_hot,
    cube_to_tensor_one_hot_similar,
    CubeValueNNFC,
    CubeValueNNConv,
    CubeValueResNet,
    CubeValueTransformer,
    CubeValueTransformerV2,
    NNValueFunctionType
)


### ------------ Cube to Tensor Conversion Tests ------------ ###

@pytest.mark.parametrize("cube_to_tensor,expected_shape", [
    (cube_to_tensor_direct, (1, 54)),
    (cube_to_tensor_one_hot, (1, 54, 6)),
    (cube_to_tensor_one_hot_similar, (1, 24, 54, 6)),
])
def test_cube_to_tensor_conversion(cube_to_tensor, expected_shape):
    # Test single cube
    cube = Cube()
    tensor = cube_to_tensor(cube)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == expected_shape
    assert tensor.dtype == torch.float32

    # Test batch of cubes
    N = 5
    cubes = [Cube() for _ in range(N)]
    tensor = cube_to_tensor(cubes)
    assert isinstance(tensor, torch.Tensor)
    batch_shape = (N,) + expected_shape[1:]  # Replace first dimension with N
    assert tensor.shape == batch_shape
    assert tensor.dtype == torch.float32


### ------------ Neural Network Forward Pass Tests ------------ ###

@pytest.fixture(params=[
    CubeValueNNFC(cube_to_tensor_direct, hidden_size=[128, 64]),
    CubeValueNNFC(cube_to_tensor_one_hot, hidden_size=[256, 256, 64]),
    CubeValueNNConv(),
    CubeValueResNet(),
    CubeValueTransformer(),
    CubeValueTransformerV2()
])
def model(request):
    """Fixture that provides different cube value neural network models."""
    return request.param


def test_neural_network_forward_pass(model):
    model.eval()

    # Get the cube_to_tensor function from the model
    cube_to_tensor = model.get_cube_to_tensor()

    # Test single cube
    cube = Cube()
    tensor = cube_to_tensor(cube)

    with torch.no_grad():
        output = model(tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1)

    # Test batch of cubes
    N = 5
    cubes = [Cube() for _ in range(N)]
    tensor = cube_to_tensor(cubes)

    with torch.no_grad():
        output = model(tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (N, 1)


@pytest.mark.parametrize("nn_value_function_type", NNValueFunctionType)
def test_nn_value_function(model, nn_value_function_type):
    cube = Cube()
    N = 5
    cubes = [Cube() for _ in range(N)]

    model.set_value_function_type(nn_value_function_type)
    val_fn = model.as_value_function()
    val = val_fn(cube)
    assert isinstance(val, np.ndarray)
    assert val.shape == ()
    assert float(val) == float(val)  # Check it's a scalar

    vals = val_fn(cubes)
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (N,)

    # Check consistency between single and batch calls
    cube1 = Cube(n_scramble_moves=10, scramble_seed=42)
    cube2 = Cube(n_scramble_moves=10, scramble_seed=43)
    val1 = val_fn(cube1)
    val2 = val_fn(cube2)
    vals = val_fn([cube1, cube2])
    assert np.isclose(vals[0], val1)
    assert np.isclose(vals[1], val2)
