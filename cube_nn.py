""" Neural Networks for Cube solving. """
import numpy as np
import torch
from torch import nn
from typing import List, Protocol, Optional
from cube import Cube, moves
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


def cube_to_tensor_one_hot_similar(cube: Cube | List[Cube]) -> torch.Tensor:
    """
    Converts a Cube object to a one-hot encoded tensor representation, with similar cubes. Mainly used for convolutional neural networks.

    Args:
        cube (Cube): The Cube object to convert.

    Returns:
        torch.Tensor: A one-hot encoded tensor representation of the cube state and all the similar cubes, of shape (N, 24, 54, 6). (24 similar cubes, 6x9=54 stickers, 6 colors)
    """
    if isinstance(cube, Cube):
        cube = [cube]
    return torch.stack([cube_to_tensor_one_hot(c.get_all_similar_cubes()) for c in cube], dim=0)


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


class CubeValueNNConv(CubeValueNN):
    """A convolutional neural network for cube value estimation (estimated number of moves needed to solve the cube).

    Uses a (N, 24, 54, 6) tensor representation of the cube state and similar cubes as one-hot encoding.
    """

    def __init__(self):
        """
        Initializes the CubeValueNNConv model.
        """
        super(CubeValueNNConv, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(24, 512, kernel_size=(54, 6)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
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
            cube_tensor (torch.Tensor): The tensor representation of the cube state and similar cubes, of shape (N, 24, 54, 6).

        Returns:
            torch.Tensor: The output of the neural network.
        """
        return self.model(cube_tensor)

    def get_cube_to_tensor(self) -> CubeToTensor:
        """
        Returns the cube to tensor conversion function used by the model.

        Returns:
            CubeToTensor: The cube to tensor conversion function.
        """
        return cube_to_tensor_one_hot_similar


class CubeValueResNet(CubeValueNN):
    """A simple neural network with residual blocks for cube value estimation (estimated number of moves needed to solve the cube).

    Uses a one-hot encoded (N, 54, 6) tensor representation of the cube state.
    """

    def __init__(self):
        """
        Initializes the CubeValueResNet model.

        Args:
            num_blocks (int): Number of residual blocks to use.
        """
        super(CubeValueResNet, self).__init__()

        # self.layers = nn.Sequential(
        #     nn.Linear(54 * 6, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     ResidualBlock(1024, 3),
        #     ResidualBlock(1024),
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )
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


class CubeValueResNetV2(CubeValueNN):
    """A simple neural network with residual blocks for cube value estimation (estimated number of moves needed to solve the cube).

    Uses a one-hot encoded (N, 54, 6) tensor representation of the cube state.
    """

    def __init__(self):
        """
        Initializes the CubeValueResNet model.

        Args:
            num_blocks (int): Number of residual blocks to use.
        """
        super(CubeValueResNetV2, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(54 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            ResidualBlock(2048),
            ResidualBlock(2048, layers=3),
            ResidualBlock(2048),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
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


class CubeValueTransformer(CubeValueNN):
    """A transformer-based neural network with self-attention for cube value estimation.

    Uses a one-hot encoded (N, 54, 6) tensor representation of the cube state.
    Each of the 54 stickers is treated as a token with 6-dimensional features (one-hot color).
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 16, num_layers: int = 3, dim_feedforward: int = 1024):
        """
        Initializes the CubeValueTransformer model.

        Args:
            embed_dim (int): Dimension of the embedding space for each sticker token.
            num_heads (int): Number of attention heads in multi-head attention.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network in transformer layers.
        """
        super(CubeValueTransformer, self).__init__()

        # Project from 6-dimensional one-hot to embed_dim
        self.input_projection = nn.Linear(6, embed_dim)

        # Learnable positional encoding for 54 sticker positions
        self.positional_encoding = nn.Parameter(torch.randn(1, 54, embed_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Output layers - aggregate sequence and predict value
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, cube_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            cube_tensor (torch.Tensor): The tensor representation of the cube state, of shape (N, 54, 6).

        Returns:
            torch.Tensor: The output of the neural network, of shape (N, 1).
        """
        # Project to embedding dimension: (batch_size, 54, embed_dim)
        x = self.input_projection(cube_tensor)

        # Add positional encoding
        x = x + self.positional_encoding

        # Apply transformer encoder: (batch_size, 54, embed_dim)
        x = self.transformer_encoder(x)

        # Global average pooling over sequence dimension: (batch_size, embed_dim)
        x = x.mean(dim=1)

        # Output prediction: (batch_size, 1)
        return self.output_layers(x)

    def get_cube_to_tensor(self) -> CubeToTensor:
        """
        Returns the cube to tensor conversion function used by the model.

        Returns:
            CubeToTensor: The cube to tensor conversion function.
        """
        return cube_to_tensor_one_hot


class AttentionBlock(nn.Module):
    """An attention-based block."""

    def __init__(self, embed_dim: int, num_heads: int):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        # Feedforward network with residual connection and layer normalization
        ff_output = self.feedforward(x)
        x = self.norm(x + ff_output)

        return x


class CubeValueTransformerV2(CubeValueNN):
    """A transformer-based neural network with self-attention for cube value estimation.

    Each of the 54 stickers is a token with 324-dimensional (54*6) one-hot encoding.
    For each token, exactly one element is 1 (indicating position and color), rest are 0.
    Positional information is embedded in the token itself rather than learned separately.
    Uses nn.MultiheadAttention directly instead of nn.TransformerEncoder.
    """

    def __init__(self):
        """
        Initializes the CubeValueTransformerV2 model.
        """
        super(CubeValueTransformerV2, self).__init__()

        input_dim: int = 6
        embed_dim: int = 512
        forward_skip_dim: int = 512
        self.pos_encodings = nn.Parameter(torch.randn(1, 54, embed_dim))
        self.forward_skip = nn.Sequential(
            nn.Linear(6*54, forward_skip_dim),
            nn.BatchNorm1d(forward_skip_dim),
            nn.ReLU(),
        )

        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Attention blocks
        self.attention_blocks = nn.Sequential(
            AttentionBlock(embed_dim, num_heads=4),
        )

        # Output layers to predict value
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim + forward_skip_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            ResidualBlock(1024),
            ResidualBlock(1024),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, cube_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            cube_tensor (torch.Tensor): The tensor representation of the cube state, of shape (N, 54, 6).

        Returns:
            torch.Tensor: The output of the neural network, of shape (N, 1).
        """
        batch_size = cube_tensor.shape[0]

        # Create 54 tokens, each of dimension 324 (54*6)
        # For each sticker position i (0-53), create a one-hot vector across all 324 positions
        # where position i*6 + color_value is set to 1

        # Convert (N, 54, 6) one-hot to (N, 54, 324) tokens
        # x = torch.zeros((batch_size, 54, 324),
        #                 dtype=cube_tensor.dtype, device=cube_tensor.device)
        # for i in range(54):
        #     x[:, i, i*6:(i+1)*6] = cube_tensor[:, i, :]

        # Apply embedding and attention blocks
        x = self.embedding(cube_tensor)  # (N, 54, embed_dim)
        x = x + self.pos_encodings
        x = self.attention_blocks(x)  # (N, 54, embed_dim)

        # Prepare forward skip connection
        skip = self.forward_skip(
            cube_tensor.flatten(start_dim=1))  # (N, forward_skip_dim)

        # Pool over sequence dimension to get (N, embed_dim)
        x = x.mean(dim=1)  # Average pooling over the 54 tokens
        x = torch.cat([x, skip], dim=1)  # Concatenate skip connection

        # Apply output layers
        return self.output_layers(x)

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
