""" Neural Networks for Cube solving. """
import numpy as np
import torch
from torch import nn
from typing import List
from cube import Cube
from abc import abstractmethod, ABC
from protocols import CubeToTensor, ValueFunction
from enum import Enum, auto

### ------------ Cube to Tensor Conversion Functions ------------ ###


def cube_to_tensor_direct(cube: Cube | List[Cube]) -> torch.Tensor:
    """
    Direct conversion of a Cube object to a tensor representation.

    Args:
        cube (Cube): The Cube object to convert.

    Returns:
        torch.Tensor: A tensor representation of the cube state, of shape (N, 54).
    """
    if isinstance(cube, Cube):
        cube = [cube]
    return torch.tensor(np.vstack([c._cube.flatten() for c in cube]), dtype=torch.float32)


def cube_to_tensor_one_hot(cube: Cube | List[Cube]) -> torch.Tensor:
    """
    Converts a Cube object to a one-hot encoded tensor representation.

    Args:
        cube (Cube): The Cube object to convert.

    Returns:
        torch.Tensor: A one-hot encoded tensor representation of the cube state, of shape (N, 54, 6).
    """
    if isinstance(cube, Cube):
        cube = [cube]
    return torch.nn.functional.one_hot(torch.tensor(np.vstack([c._cube.flatten() for c in cube]), dtype=torch.long), num_classes=6).to(dtype=torch.float32)


### ------------ Neural Network Models ------------ ###

class NNValueFunctionType(Enum):
    STANDARD = auto()
    NEIGHBORS_MIN = auto()
    NEIGHBORS_AVG = auto()
    NEIGHBORS_MAX = auto()
    NEIGHBORS_MID = auto()
    SIMILAR_MIN = auto()
    SIMILAR_AVG = auto()
    SIMILAR_MAX = auto()
    SIMILAR_MID = auto()


class CubeValueNN(nn.Module, ABC):
    """ Abstract base class for cube value neural networks. """

    def __init__(self):
        super(CubeValueNN, self).__init__()
        self._nn_value_function_type: NNValueFunctionType = NNValueFunctionType.STANDARD

    @abstractmethod
    def get_cube_to_tensor(self) -> CubeToTensor:
        """ Returns the cube to tensor conversion function used by the model. """
        pass

    def as_value_function(self) -> ValueFunction:
        """ Converts the neural network model into a value function that can handle both single Cube and list of Cubes. """
        cube_to_tensor = self.get_cube_to_tensor()
        device = next(self.parameters()).device

        if self._nn_value_function_type == NNValueFunctionType.STANDARD:
            def value_function(cube: Cube | List[Cube]) -> np.ndarray:
                self.eval()
                with torch.no_grad():
                    tensor = cube_to_tensor(cube).to(device)
                    return self(tensor).cpu().numpy().squeeze()

        else:
            match self._nn_value_function_type:
                case NNValueFunctionType.NEIGHBORS_MIN | NNValueFunctionType.SIMILAR_MIN:
                    def agg_func(values: np.ndarray) -> np.ndarray:
                        return values.min()
                case NNValueFunctionType.NEIGHBORS_AVG | NNValueFunctionType.SIMILAR_AVG:
                    def agg_func(values: np.ndarray) -> np.ndarray:
                        return values.mean()
                case NNValueFunctionType.NEIGHBORS_MAX | NNValueFunctionType.SIMILAR_MAX:
                    def agg_func(values: np.ndarray) -> np.ndarray:
                        return values.max()
                case NNValueFunctionType.NEIGHBORS_MID | NNValueFunctionType.SIMILAR_MID:
                    def agg_func(values: np.ndarray) -> np.ndarray:
                        return np.max(values) - np.min(values) / 2 + np.min(values)

            if self._nn_value_function_type in {NNValueFunctionType.SIMILAR_MIN, NNValueFunctionType.SIMILAR_AVG, NNValueFunctionType.SIMILAR_MAX, NNValueFunctionType.SIMILAR_MID}:
                cube_method = Cube.get_all_similar_cubes
            elif self._nn_value_function_type in {NNValueFunctionType.NEIGHBORS_MIN, NNValueFunctionType.NEIGHBORS_AVG, NNValueFunctionType.NEIGHBORS_MAX, NNValueFunctionType.NEIGHBORS_MID}:
                cube_method = Cube.get_all_neighbors

            def value_function(cube: Cube | List[Cube]) -> np.ndarray:
                self.eval()
                with torch.no_grad():
                    if isinstance(cube, Cube):
                        used_cubes = cube_method(cube)
                        tensor = cube_to_tensor(used_cubes).to(device)
                        values = self(tensor).cpu().numpy().squeeze()
                        return np.array(agg_func(values))
                    else:
                        all_used_cubes = []
                        for c in cube:
                            all_used_cubes.extend(cube_method(c))
                        tensor = cube_to_tensor(all_used_cubes).to(device)
                        all_values = self(tensor).cpu().numpy().squeeze()

                        N = len(cube_method(Cube()))
                        agg_vals = [agg_func(
                            all_values[i * N:(i + 1) * N]) for i in range(len(cube))]
                        return np.array(agg_vals)
        return value_function

    def set_value_function_type(self, vf_type: NNValueFunctionType):
        """ Sets the value function type for the neural network.

        Args:
            vf_type (NNValueFunctionType): The type of value function to use.
        """
        self._nn_value_function_type = vf_type


class CubeValueNNFC(CubeValueNN):
    """A simple neural network with fully connected layers."""

    def __init__(self, cube_to_tensor: CubeToTensor, hidden_size: List[int] = []):
        """
        Initializes the CubeValueNNFC model.

        Args:
            cube_to_tensor (CubeToTensor): The function to convert a Cube object to a tensor representation.
            hidden_size (List[int]): List of integers defining the sizes of hidden layers.
        """
        super(CubeValueNNFC, self).__init__()
        self.cube_to_tensor = cube_to_tensor
        # infer input size from cube_to_tensor
        input_size = self.cube_to_tensor(Cube()).numel()

        output_size = 1

        layers = []
        for size in hidden_size:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size

        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, cube_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            cube_tensor (torch.Tensor): The tensor representation of the cube state.

        Returns:
            torch.Tensor: The output of the neural network.
        """
        return self.model(cube_tensor.flatten(start_dim=1))

    def get_cube_to_tensor(self) -> CubeToTensor:
        """
        Returns the cube to tensor conversion function used by the model.

        Returns:
            CubeToTensor: The cube to tensor conversion function.
        """
        return self.cube_to_tensor


class CubeValueResNet(CubeValueNN):
    """A simple neural network with residual blocks for cube value estimation (estimated number of moves needed to solve the cube).

    Uses a one-hot encoded (N, 54, 6) tensor representation of the cube state.
    """

    def __init__(self):
        """
        Initializes the CubeValueResNet model.
        """
        super(CubeValueResNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(54 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, cube_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            cube_tensor (torch.Tensor): The tensor representation of the cube state, of shape (N, 54, 6) or (54, 6).

        Returns:
            torch.Tensor: The output of the neural network.
        """
        return self.layers(cube_tensor.flatten(start_dim=1))

    def get_cube_to_tensor(self) -> CubeToTensor:
        """
        Returns the cube to tensor conversion function used by the model.

        Returns:
            CubeToTensor: The cube to tensor conversion function.
        """
        return cube_to_tensor_one_hot


class ResidualBlock(nn.Module):
    """A single residual block for use in ResNet architectures."""

    def __init__(self, channels: int, layers: int = 2):
        """
        Initializes the ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels. If None, set to in_channels.
            layers (int): Number of hidden layers in the block.
        """
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.BatchNorm1d(channels),
                    nn.ReLU()
                ) for _ in range(layers - 1)
            ],
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """
        return nn.ReLU()(self.block(x) + x)
