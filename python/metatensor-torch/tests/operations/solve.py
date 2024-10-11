import io

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from packaging import version


def test_solve():
    X_tensor = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1, 0], [0, 1]], dtype=torch.float64),
                samples=Labels.range("s", 2),
                components=[],
                properties=Labels.range("p1", 2),
            )
        ],
    )
    Y_tensor = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1, 0], [0, 1]], dtype=torch.float64),
                samples=Labels.range("s", 2),
                components=[],
                properties=Labels.range("p2", 2),
            )
        ],
    )
    solution_tensor = metatensor.torch.solve(X_tensor, Y_tensor)

    # check output type
    assert isinstance(solution_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert solution_tensor._type().name() == "TensorMap"

    # check content
    expected_solution = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1, 0], [0, 1]], dtype=torch.float64),
                samples=Labels.range("p2", 2),
                components=[],
                properties=Labels.range("p1", 2),
            )
        ],
    )
    assert metatensor.torch.equal(solution_tensor, expected_solution)


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.solve, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
