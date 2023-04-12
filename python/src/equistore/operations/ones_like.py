from typing import List, Union

from equistore import TensorBlock, TensorMap

from . import _dispatch
from .equal_metadata import _check_gradient_presence


def ones_like(
    tensor: TensorMap,
    parameters: Union[List[str], str] = None,
    requires_grad: bool = False,
) -> TensorMap:
    """Return a new :py:class:`TensorMap` with the same metadata as tensor,
    and all values equal to one.

    :param tensor: Input tensor from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradient of ``tensor`` are present in the new tensor.
                       If empty list ``[]`` no gradients information are copied.
    :param requires_grad: If autograd should record operations for the returned
                          tensor. This option is only relevant for torch.

    >>> import numpy as np
    >>> import equistore
    >>> from equistore import TensorBlock, TensorMap, Labels
    >>> np.random.seed(1)

    First we create a :py:class:`TensorMap` with just one block with two
    gradients, named ``alpha`` and ``beta``, containing random data:

    >>> block = TensorBlock(
    ...     values=np.random.rand(4, 3),
    ...     samples=Labels.arange("sample", 4),
    ...     components=[],
    ...     properties=Labels.arange("property", 3),
    ... )
    >>> block.add_gradient(
    ...     parameter="alpha",
    ...     data=np.random.rand(2, 3, 3),
    ...     samples=Labels(["sample", "atom"], np.array([[0, 0], [0, 2]])),
    ...     components=[Labels.arange("component", 3)],
    ... )
    >>> block.add_gradient(
    ...     parameter="beta",
    ...     data=np.random.rand(1, 3),
    ...     samples=Labels(["sample"], np.array([[0]])),
    ...     components=[],
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor = TensorMap(keys, [block])
    >>> print(tensor.block(0))
    TensorBlock
        samples (4): ['sample']
        components (): []
        properties (3): ['property']
        gradients: ['alpha', 'beta']

    Then we use ``ones_like`` to create a :py:class:`TensorMap` with the same
    metadata as ``tensor``, but with all values set to 1.

    >>> tensor_ones = equistore.ones_like(tensor)
    >>> print(tensor_ones.block(0))
    TensorBlock
        samples (4): ['sample']
        components (): []
        properties (3): ['property']
        gradients: ['alpha', 'beta']
    >>> print(tensor_ones.block(0).values)
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    >>> print(tensor_ones.block(0).gradient("alpha").data)
    [[[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]
    <BLANKLINE>
     [[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]]

    Note that if we copy just the gradient ``alpha``, ``beta`` is no longer
    available.

    >>> tensor_ones = equistore.ones_like(tensor, parameters="alpha")
    >>> print(tensor_ones.block(0).gradients_list())
    ['alpha']
    """

    blocks = []
    for block in tensor.blocks():
        blocks.append(
            ones_like_block(
                block=block, parameters=parameters, requires_grad=requires_grad
            )
        )
    return TensorMap(tensor.keys, blocks)


def ones_like_block(
    block: TensorBlock,
    parameters: Union[List[str], str] = None,
    requires_grad: bool = False,
) -> TensorBlock:
    """Return a new :py:class:`TensorBlock` with the same metadata as block,
    and all values equal to one.

    :param block: Input block from which the metadata is taken.
    :param parameters: Which gradient parameter to copy. If ``None`` (default)
                       all gradient of ``tensor`` are present in the new tensor.
                       If empty list ``[]`` no gradients information are copied.
    :param requires_grad: If autograd should record operations for the returned tensor.
                          This option is only relevant for torch.
    """

    values = _dispatch.ones_like(block.values, requires_grad=requires_grad)
    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    if isinstance(parameters, str):
        parameters = [parameters]

    if parameters is None:
        parameters = block.gradients_list()
    else:
        _check_gradient_presence(block=block, parameters=parameters, fname="ones_like")

    for parameter in parameters:
        gradient = block.gradient(parameter)
        gradient_data = _dispatch.ones_like(gradient.data)

        result_block.add_gradient(
            parameter,
            gradient_data,
            gradient.samples,
            gradient.components,
        )

    return result_block
