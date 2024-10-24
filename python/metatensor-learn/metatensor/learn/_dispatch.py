"""Helper functions to dispatch methods between numpy and torch."""

from typing import List, Optional, Union

import numpy as np


try:
    import torch
    from torch import Tensor as TorchTensor
except ImportError:

    class TorchTensor:
        pass


from ._backend import Array


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


def int_array_like(int_list: Union[List[int], List[List[int]]], like):
    """
    Converts the input list of int to a numpy array or torch tensor
    based on the type of `like`.
    """
    if isinstance(like, TorchTensor):
        if torch.jit.isinstance(int_list, List[int]):
            return torch.tensor(int_list, dtype=torch.int64, device=like.device)
        else:
            return torch.tensor(int_list, dtype=torch.int64, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(int_list).astype(np.int64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


# def norm(array, axis=None):
#     """Compute the 2-norm (Frobenius norm for matrices) of the input array.

#     This calls the equivalent of ``np.linalg.norm(array, axis=axis)``, see this
#     function for more documentation.
#     """
#     if isinstance(array, TorchTensor):
#         return np.linalg.norm(array, axis=axis)
#     elif isinstance(array, np.ndarray):
#         return torch.linalg.norm(array, dim=axis)
#     else:
#         raise TypeError(UNKNOWN_ARRAY_TYPE)


def abs(array):
    """
    Returns the absolute value of the elements in the array.

    It is equivalent of np.abs(array) and torch.abs(tensor)
    """
    if isinstance(array, TorchTensor):
        return torch.abs(array)
    elif isinstance(array, np.ndarray):
        return np.abs(array).astype(array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def svd(array, full_matrices=True):
    """Compute the Singular Value Decomposition (SVD) of the input array.

    This calls the equivalent of ``np.linalg.svd(array, full_matrices=full_matrices)``,
    see this function for more documentation.
    """
    if isinstance(array, TorchTensor):
        return torch.linalg.svd(array, full_matrices=full_matrices)
    elif isinstance(array, np.ndarray):
        return np.linalg.svd(array, full_matrices=full_matrices)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def argmax(array, axis: int):
    """
    Returns the argmax of the array.

    It is equivalent of np.argmax(array) and torch.argmax(tensor)
    """
    if isinstance(array, TorchTensor):
        return torch.argmax(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.argmax(array, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def sign(array):
    """
    Returns an indication of the sign of the elements in the array.

    It is equivalent of np.sign(array) (as defined in v2.0) and torch.sgn(tensor)
    """
    if isinstance(array, TorchTensor):
        return torch.sgn(array)
    elif isinstance(array, np.ndarray):
        if np.issubdtype(array.dtype, np.complexfloating):
            return array / np.abs(array)
        else:
            return np.sign(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def sum(array, axis: Optional[int] = None):
    """
    Returns the sum of the elements in the array at the axis.

    It is equivalent of np.sum(array, axis=axis) and torch.sum(tensor, dim=axis)
    """
    if isinstance(array, TorchTensor):
        return torch.sum(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.sum(array, axis=axis).astype(array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def cumsum(array, axis: Optional[int] = None):
    """
    Returns the cumulative sum of the elements in the array at the axis.

    It is equivalent of np.cumsum(array, axis=axis) and torch.cumsum(tensor, dim=axis)
    """
    if isinstance(array, TorchTensor):
        return torch.cumsum(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.cumsum(array, axis=axis).astype(array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def arange_like(
    start: Optional[int] = 0,
    end: int = None,
    step: Optional[int] = 1,
    like: Array = None,
):
    """
    Return evenly spaced values within a given interval.

    It is equivalent of np.arange(start, end, step) and torch.arange(start, end, step)
    """
    if isinstance(like, TorchTensor):
        return torch.arange(start, end, step)
    elif isinstance(like, np.ndarray):
        return np.arange(start, end, step).astype(like.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def copy(array):
    if isinstance(array, TorchTensor):
        return torch.clone(array)
    elif isinstance(array, np.ndarray):
        return np.copy(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def stack(arrays, axis=0):
    if isinstance(arrays[0], TorchTensor):
        return torch.stack(arrays, dim=axis)
    elif isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
