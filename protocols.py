from typing import Protocol, List
import torch
import numpy as np

from cube import Cube


### ------------ Cube to Tensor Conversion Protocol ------------ ###

class CubeToTensor(Protocol):
    """ Protocol for a function that converts a Cube object to a tensor representation. """

    def __call__(self, cube: Cube | List[Cube]) -> torch.Tensor:
        """ Converts a Cube object to a tensor representation. """
        ...


### ------------ Value Function Protocol ------------ ###

class ValueFunction(Protocol):
    """ A protocol for a value function that takes a Cube and returns a float value. Must work with batching (list of Cubes).

    ValueFunction(cube: Cube | List[Cube]) -> np.ndarray """

    def __call__(self, cube: Cube | List[Cube]) -> np.ndarray:
        ...
