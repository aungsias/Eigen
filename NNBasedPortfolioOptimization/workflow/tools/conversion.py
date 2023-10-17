import numpy as np
import torch

from typing import Tuple, List, Union

def to_tensors(*arrays: np.ndarray) -> Tuple[torch.Tensor, ...]:

    """
    Converts a collection of multidimensional numpy arrays into PyTorch tensors.

    Parameters:
    - *arrays (np.ndarray): A single multidimensional arrays or collection of \
        multidimensional arrays.
    
    Returns:
    - Tuple[torch.Tensor, ...]: A tuple of PyTorch tensors.
    """

    tensors = []
    for array in arrays:
        tensors.append(torch.tensor(array, dtype=torch.float32))
    return tuple(tensors)
