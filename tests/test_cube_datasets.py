import pytest
import numpy as np
from cube import Cube
from cube_nn import (
    cube_to_tensor_direct,
    cube_to_tensor_one_hot,
    cube_to_tensor_one_hot_similar,
    CubeValueNNFC,
    CubeValueNNConv,
    CubeValueResNet,
)
