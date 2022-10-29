import pytest
import torch

try:
    from torchtyping import TensorType

    TORCHTYPING_AVAILABLE = True
except ImportError:
    TORCHTYPING_AVAILABLE = False

from torch_geometric.data import Batch, Data
from torch_geometric.tp import BatchT, DataT, typecheck


def get_sample_tensor_data() -> Data:
    d = Data()
    d.x = torch.randn((1, 2, 3))
    return d


def get_sample_string_data() -> Data:
    d = Data()
    d.x = "not a tensor"
    return d


def get_sample_mixed_data() -> Data:
    d = Data()
    d.x = torch.randn((1, 2, 3))
    d.y = "not a tensor"
    return d


def test_typecheck_attribute():
    d = get_sample_data()

    @typecheck
    def forward(d: DataT["x":torch.Tensor]):
        return d

    assert d == forward(
        get_sample_tensor_data), "Return value is not the same as input"

    with pytest.raises(TypeError):
        forward(get_sample_string_data())


def test_fully_specified():
    d = get_sample_mixed_data()

    @typecheck
    def forward(d: DataT["x":torch.Tensor]):
        return d

    with pytest.raises(TypeError):
        forward(d)


def test_fuzzy_specification():
    d = get_sample_mixed_data()

    @typecheck
    def forward(d: DataT["x":torch.Tensor, ...]):
        return d

    assert d == forward(d)


@pytest.mark.skipif(not TORCHTYPING_AVAILABLE,
                    reason="torchtyping not available")
def test_tensor_shape_specification():
    d = get_sample_mixed_data()

    @typecheck
    def forward(d: DataT["x":TensorType[-1, -1, 3], ...]):
        return d

    assert d == forward(d)

    @typecheck
    def forward(d: DataT["x":TensorType[-1, 20, 3], ...]):
        return d

    with pytest.raises(TypeError):
        forward(d)


@pytest.mark.skipif(not TORCHTYPING_AVAILABLE,
                    reason="torchtyping not available")
def test_batch_shape_specification():
    d = get_sample_mixed_data()
    b = Batch.from_data_list([d])

    @typecheck
    def forward(d: BatchT["x":TensorType[-1, -1, 3], ...]):
        return d

    assert b == forward(b)
