# from torch_geometric.testing import onlyFullTest
# import pytest
import torch
from torch import nn

from torch_geometric.contrib.degree_quant import (
    GATConvQuant,
    GCNConvQuant,
    GINConvQuant,
)
# from torch_geometric.contrib.degree_quant.training import (
#     cross_validation_with_val_set,
# )
from torch_geometric.contrib.degree_quant.linear import LinearQuantized
from torch_geometric.contrib.degree_quant.models import (
    GAT,
    GCN,
    GIN,
    ResettableSequential,
)
from torch_geometric.contrib.degree_quant.quantizer import (
    ProbabilisticHighDegreeMask,
    make_quantizers,
)
from torch_geometric.data import Data


def test_layers():

    # device = torch.device('cuda')
    x = torch.randn(8, 3)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ])

    batch = torch.tensor([2])
    prob_mask = torch.tensor([0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0])

    # edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])

    data = Data(x=x, edge_index=edge_index, y=torch.Tensor([1]), batch=batch,
                prob_mask=prob_mask)
    # data = [data, data, data, data]
    data = data
    # dataset = process_dataset(dataset)
    mask = ProbabilisticHighDegreeMask(0.1, 0.8)

    graph = mask(data)

    assert graph.prob_mask is not None

    argsINT8 = {
        "qypte": "INT8",
        "ste": False,
        "momentum": False,
        "percentile": 0.001,
        "dq": True,
        "sample_prop": None,
    }

    GIN_model = GIN(3, 16, num_layers=2, hidden=64, **argsINT8)
    out1 = GIN_model(data)

    GCN_model = GCN(3, 16, num_layers=2, hidden=64, **argsINT8)
    out2 = GCN_model(data)

    GAT_model = GAT(3, 16, num_layers=2, hidden=64, **argsINT8)
    out3 = GAT_model(data)

    lq, mq = make_quantizers(**argsINT8)
    linear = LinearQuantized(3, 16, lq)

    out4 = linear(data.x)

    assert out3.shape == (3, 16)
    assert out2.shape == (3, 16)
    assert out1.shape == (3, 16)
    assert out4.shape == (8, 16)

    assert isinstance(GIN_model, nn.Module) is True
    assert isinstance(GCN_model, nn.Module) is True
    assert isinstance(GAT_model, nn.Module) is True

    n = ResettableSequential(torch.nn.ReLU())
    m1 = GCNConvQuant(3, 16, n, layer_quantizers=lq, mp_quantizers=mq)
    m2 = GATConvQuant(3, 16, n, layer_quantizers=lq, mp_quantizers=mq)
    m3 = GINConvQuant(n, mp_quantizers=mq)

    out5 = m1(data.x, data.edge_index)
    out6 = m2(data.x, data.edge_index)
    out7 = m3(data.x, data.edge_index)

    assert out5.shape == (8, 16)
    assert out6.shape == (8, 16)
    assert out7.shape == (8, 3)
