import functools
import operator
from typing import List

import numpy as np

from equistore.core import Labels, TensorBlock, TensorMap

from ._utils import _check_same_keys


def join(tensors: List[TensorMap], axis: str):
    """Join a sequence of :py:class:`TensorMap` along an axis.

    The ``axis`` parameter specifies the type join. For example, if
    ``axis='properties'`` it will be the `tensor_maps` will be joined along the
    `properties` dimension and for ``axis='samples'`` they will be the along the
    samples dimension.

    ``join`` will create an additional label `tensor` specifiying the original index in
    the list of `tensor_maps`.  If `sample`/`property` names are not the same in all
    `tensor_maps` they will be unified with a general name ``"property"``.

    :param tensors:
        sequence of :py:class:`TensorMap` for join
    :param axis:
        A string indicating how the tensormaps are stacked. Allowed
        values are ``'properties'`` or ``'samples'``.

    :return tensor_joined:
        The stacked :py:class:`TensorMap` with more properties or samples
        than the input TensorMap.
    """

    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "the `TensorMap`s to join must be provided as a list or a tuple"
        )

    if len(tensors) < 1:
        raise ValueError("provide at least one `TensorMap` for joining")

    if axis not in ("samples", "properties"):
        raise ValueError(
            "Only `'properties'` or `'samples'` are "
            "valid values for the `axis` parameter."
        )

    if len(tensors) == 1:
        return tensors[0]

    for ts_to_join in tensors[1:]:
        _check_same_keys(tensors[0], ts_to_join, "join")

    # Deduce if sample/property names are the same in all tensors.
    # If this is not the case we have to change unify the corresponding labels later.
    if axis == "samples":
        names_list = [tensor.sample_names for tensor in tensors]
    else:
        names_list = [tensor.property_names for tensor in tensors]

    # We use functools to flatten a list of sublists::
    #
    #   [('a', 'b', 'c'), ('a', 'b')] -> ['a', 'b', 'c', 'a', 'b']
    #
    # A nested list with sublist of different shapes can not be handled by np.unique.
    unique_names = np.unique(functools.reduce(operator.concat, names_list))

    # Label names are unique: We can do an equal check only checking the lengths.
    names_are_same = np.all(
        len(unique_names) == np.array([len(names) for names in names_list])
    )

    # It's fine to lose metadata on the property axis, less so on the sample axis!
    if axis == "samples" and not names_are_same:
        raise ValueError(
            "Sample names are not the same! Joining along samples with different "
            "sample names will loose information and is not supported."
        )

    keys_names = ("tensor",) + tensors[0].keys.names
    keys_values = []
    blocks = []

    for i, tensor in enumerate(tensors):
        keys_values += [(i,) + value for value in tensor.keys.tolist()]

        for _, block in tensor:
            # We would already raised an error if `axis == "samples"`. Therefore, we can
            # neglect the check for `axis == "properties"`.
            if names_are_same:
                properties = block.properties
            else:
                properties = Labels.arange("property", len(block.properties))

            new_block = TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            for parameter, gradient in block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )

                new_block.add_gradient(
                    parameter=parameter,
                    gradient=TensorBlock(
                        values=gradient.values,
                        samples=gradient.samples,
                        components=gradient.components,
                        properties=new_block.properties,
                    ),
                )

            blocks.append(new_block)

    keys = Labels(names=keys_names, values=np.array(keys_values))
    tensor = TensorMap(keys=keys, blocks=blocks)

    if axis == "samples":
        tensor_joined = tensor.keys_to_samples("tensor")
    else:
        tensor_joined = tensor.keys_to_properties("tensor")

    # TODO: once we have functions to manipulate meta data we can try to
    # remove the `tensor` label entry after joining.

    return tensor_joined