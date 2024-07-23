from functools import partial
from typing import Dict, NewType

import pytest
import torch

try:
    from jaxtyping import Float

    JAXTYPING_AVAILABLE = True
except ImportError:
    JAXTYPING_AVAILABLE = False

from torch import Tensor

from torch_geometric.data import Batch, Data
from torch_geometric.data.typehinting import BatchT, DataT, typecheck

assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)


def get_sample_tensor_data() -> Data:
    d = Data()
    d.x = torch.randn((1, 2, 3))
    return d


def get_sample_int_data() -> Data:
    d = Data()
    d.x = torch.randint(0, 10, (1, 2, 3))
    return d


def get_sample_float_data() -> Data:
    d = Data()
    d.x = torch.rand((1, 2, 3)).float()
    return d


def get_sample_bool_data() -> Data:
    d = Data()
    d.x = torch.randint(0, 2, (1, 2, 3), dtype=torch.bool)
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


def get_sample_dictionary_data() -> Data:
    d = Data()
    d.x = torch.randn((1, 2, 3))
    d.dictionary = {"key": "value"}
    return d


def test_typecheck_attribute():
    d = get_sample_tensor_data()

    @typecheck
    def forward(d: DataT["x"]):
        return d

    assert d == forward(d), "return value does not match."

    non_compliant = Data()
    non_compliant.y = torch.randn((1, 2, 3))

    with pytest.raises(TypeError):
        forward(non_compliant)


def test_typecheck_attribute_type():
    d = get_sample_tensor_data()

    @typecheck
    def forward(d: DataT["x" : torch.Tensor]):  # noqa
        return d

    assert d == forward(d), "Return value is not the same as input"

    with pytest.raises(TypeError):
        forward(get_sample_string_data())


def test_fully_specified():
    d = get_sample_mixed_data()

    @typecheck
    def forward(d: DataT["x" : torch.Tensor]):  # noqa
        return d

    with pytest.raises(TypeError):
        forward(d)


def test_fuzzy_specification():
    d = get_sample_mixed_data()

    @typecheck
    def forward(d: DataT["x" : torch.Tensor, ...]):  # noqa
        return d

    assert d == forward(d)


@pytest.mark.skipif(not JAXTYPING_AVAILABLE, reason="jaxtyping not available")
def test_tensor_shape_specification():
    d = get_sample_mixed_data()

    @typecheck
    def forward(d: DataT["x" : Float[Tensor, "... 3"], ...]):  # noqa
        return d

    assert d == forward(d)

    @typecheck
    def forward(d: DataT["x" : Float[Tensor, "... 20 3"], ...]):  # noqa
        return d

    with pytest.raises(TypeError):
        forward(d)


@pytest.mark.skipif(not JAXTYPING_AVAILABLE, reason="jaxtyping not available")
def test_batch_shape_specification():
    d = get_sample_mixed_data()
    b = Batch.from_data_list([d])

    @typecheck
    def forward(d: BatchT["x" : Float[Tensor, "... 3"], ...]):  # noqa
        return d

    assert b == forward(b)


@pytest.mark.skipif(not JAXTYPING_AVAILABLE, reason="jaxtyping not available")
def test_return_types():
    d = get_sample_tensor_data()

    @typecheck
    def forward(
        x: DataT["x" : torch.Tensor],  # noqa
    ) -> DataT["x" : Float[Tensor, "... 3"]]:  # noqa
        return x

    forward(d)

    assert d == forward(d), "Return value is not the same as input."

    @typecheck
    def forward(
        x: DataT["x" : torch.Tensor],  # noqa
    ) -> DataT["x" : Float[Tensor, "... 30"]]:  # noqa
        return x

    with pytest.raises(TypeError):
        forward(d)


def test_simple_typehint():
    d = get_sample_tensor_data()

    @typecheck
    def forward(x: DataT) -> DataT:
        return x

    out = forward(d)
    assert out == d, "Return value is not the same as input."

    @typecheck
    def forward(d: BatchT) -> BatchT:
        return d

    out = forward(Batch.from_data_list([d]))


def test_data_type_cannot_be_instantiated():
    with pytest.raises(TypeError):
        x = DataT()  # noqa


def test_batch_type_cannot_be_instantiated():
    with pytest.raises(TypeError):
        x = BatchT()  # noqa


def test_non_torch_type():
    d = get_sample_dictionary_data()

    @typecheck
    def forward(
        x: DataT["x" : torch.Tensor, "dictionary":Dict]  # noqa
    ) -> DataT["x" : torch.Tensor, "dictionary":Dict]:  # noqa
        return x

    forward(d)


def test_new_type():
    example_type = NewType("example_type", torch.Tensor)

    @typecheck(strict=False)
    def forward(x: example_type):
        return x

    x = torch.randn((1, 2, 3))
    assert_equal(x, forward(x))

    @typecheck(strict=True)
    def forward(x: DataT["x":example_type]):
        return x

    with pytest.raises(TypeError):
        forward("string")

    x = get_sample_tensor_data()
    assert x == forward(x), "Return value does not match"

    with pytest.raises(TypeError):
        x = get_sample_string_data()
        forward(x)


@pytest.mark.skipif(not JAXTYPING_AVAILABLE, reason="jaxtyping not available")
def test_nested_data():
    inner = Data()
    inner.x = torch.randn((1, 2, 3))
    outer = Data()
    outer.data = inner

    @typecheck
    def forward(x: DataT["data" : DataT["x" : Float[Tensor, "... 3"]]]):
        return x

    assert outer == forward(outer), "Return value does not match"

    with pytest.raises(TypeError):
        forward(torch.randn((1, 2, 3)))
