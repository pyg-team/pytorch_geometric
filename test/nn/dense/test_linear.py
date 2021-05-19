import pytest
import torch
from torch_geometric.nn import Linear
try:
    torch.nn.parameter.UninitializedParameter
    with_uninitialized_parameter = True
except AttributeError:
    with_uninitialized_parameter = False


@pytest.mark.skipif(not with_uninitialized_parameter,
                    reason='No UninitializedParameter')
def test_linear():
    x = torch.randn(3, 4, 16)
    linear = Linear(16, 32)
    out = linear(x)
    assert out.size() == (3, 4, 32)


@pytest.mark.skipif(not with_uninitialized_parameter,
                    reason='No UninitializedParameter')
def test_lazy_linear():
    if torch.__version__ < "1.8":
        return
    x = torch.randn(3, 4, 16)
    linear = Linear(-1, 32)
    out = linear(x)
    assert out.size() == (3, 4, 32)
