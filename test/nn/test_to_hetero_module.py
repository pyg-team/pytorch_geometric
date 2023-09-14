import pytest  # noqa
import torch

from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.to_hetero_module import (
    ToHeteroLinear,
    ToHeteroMessagePassing,
)
from torch_geometric.testing import withPackage


@withPackage('torch>=1.12.0')  # TODO Investigate error
@pytest.mark.parametrize('LinearCls', [torch.nn.Linear, Linear])
def test_to_hetero_linear(LinearCls):
    x_dict = {'1': torch.randn(5, 16), '2': torch.randn(4, 16)}
    x = torch.cat([x_dict['1'], x_dict['2']], dim=0)
    type_vec = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

    module = ToHeteroLinear(LinearCls(16, 32), list(x_dict.keys()))

    out_dict = module(x_dict)
    assert len(out_dict) == 2
    assert out_dict['1'].size() == (5, 32)
    assert out_dict['2'].size() == (4, 32)

    out = module(x, type_vec)
    assert out.size() == (9, 32)

    assert torch.allclose(out_dict['1'], out[0:5])
    assert torch.allclose(out_dict['2'], out[5:9])


def test_to_hetero_message_passing():
    x_dict = {'1': torch.randn(5, 16), '2': torch.randn(4, 16)}
    x = torch.cat([x_dict['1'], x_dict['2']], dim=0)
    node_type = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

    edge_index_dict = {
        ('1', 'to', '2'): torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]]),
        ('2', 'to', '1'): torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]]),
    }
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 5, 6, 7, 8],
        [5, 5, 6, 7, 8, 0, 1, 2, 3, 4],
    ])
    edge_type = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    module = ToHeteroMessagePassing(SAGEConv(16, 32), list(x_dict.keys()),
                                    list(edge_index_dict.keys()))

    out_dict = module(x_dict, edge_index_dict)
    assert len(out_dict) == 2
    assert out_dict['1'].size() == (5, 32)
    assert out_dict['2'].size() == (4, 32)

    out = module(x, edge_index, node_type, edge_type)
    assert out.size() == (9, 32)

    assert torch.allclose(out_dict['1'], out[0:5])
    assert torch.allclose(out_dict['2'], out[5:9])
