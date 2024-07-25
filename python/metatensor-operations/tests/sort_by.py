import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


@pytest.fixture
def tensor_two_samples():
    # samples are descending, components and properties are ascending
    block_1 = TensorBlock(
        values=np.array(
            [
                [3, 5],
                [1, 2],
                [-1, -2],
                [11, 22],
                [41, 42],
            ]
        ),
        samples=Labels(
            ["s", "a"], np.array([[2, 5], [0, 0], [0, 9], [0, 1], [0, 110]])
        ),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                    [[17, 18], [19, 20]],
                    [[21, 22], [23, 24]],
                    [[25, 26], [27, 28]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array(
                    [
                        [2, 1],
                        [1, 1],
                        [0, 1],
                        [0, 2],
                        [3, 1],
                        [4, 1],
                        [4, 0],
                    ]
                ),
            ),
            components=[
                Labels(["c"], np.array([[1], [0]])),
            ],
            properties=block_1.properties,
        ),
    )
    keys = Labels(names=["key_1"], values=np.array([[0]]))
    # block order is descending
    return TensorMap(keys, [block_1])


@pytest.fixture
def tensor_two_samples_ascending_a():
    block_1 = TensorBlock(
        values=np.array(
            [
                [1, 2],
                [11, 22],
                [3, 5],
                [-1, -2],
                [41, 42],
            ]
        ),
        samples=Labels(
            ["s", "a"], np.array([[0, 0], [0, 1], [2, 5], [0, 9], [0, 110]])
        ),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[5, 6], [7, 8]],
                    [[17, 18], [19, 20]],
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                    [[1, 2], [3, 4]],
                    [[25, 26], [27, 28]],
                    [[21, 22], [23, 24]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array(
                    [
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [2, 2],
                        [3, 1],
                        [4, 0],
                        [4, 1],
                    ]
                ),
            ),
            components=[
                Labels(["c"], np.array([[1], [0]])),
            ],
            properties=block_1.properties,
        ),
    )
    keys = Labels(names=["key_1"], values=np.array([[0]]))
    # block order is descending
    return TensorMap(keys, [block_1])


@pytest.fixture
def tensor_two_samples_descending_a():
    block_1 = TensorBlock(
        values=np.array(
            [
                [41, 42],
                [-1, -2],
                [3, 5],
                [11, 22],
                [1, 2],
            ]
        ),
        samples=Labels(
            ["s", "a"], np.array([[0, 110], [0, 9], [2, 5], [0, 1], [0, 0]])
        ),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    keys = Labels(names=["key_1"], values=np.array([[0]]))
    # block order is descending
    return TensorMap(keys, [block_1])


def test_raise_error(tensor):
    error_message = (
        "'name' is allowed only if 'axes' is one of"
        "'samples', 'components','properties' but"
        "'axes' is a List"
    )
    with pytest.raises(ValueError, match=error_message):
        metatensor.operations.sort_by(tensor, axes=["samples", "components"], name="s")

    error_message = (
        "'name' is allowed only if 'axes' is one of"
        "'samples', 'components','properties' but"
        "'axes'=='all'"
    )
    with pytest.raises(ValueError, match=error_message):
        metatensor.operations.sort_by(tensor, axes="all", name="s")


def test_sort_by_two_sample(tensor_two_samples, tensor_two_samples_ascending_a):
    print(
        "ii",
        metatensor.sort_by_block(tensor_two_samples.block(0), axes="samples", name="a")
        .gradient("g")
        .samples,
    )
    print("jj", tensor_two_samples_ascending_a.block(0).gradient("g").samples)

    metatensor.allclose_block_raise(
        metatensor.sort_by_block(tensor_two_samples.block(0), axes="samples", name="a"),
        tensor_two_samples_ascending_a.block(0),
    )


def test_sort_by_two_sample_descending(
    tensor_two_samples, tensor_two_samples_descending_a
):
    t = metatensor.remove_gradients(tensor_two_samples)
    metatensor.allclose_block_raise(
        metatensor.sort_by_block(t.block(0), axes="samples", name="a", descending=True),
        tensor_two_samples_descending_a.block(0),
    )
