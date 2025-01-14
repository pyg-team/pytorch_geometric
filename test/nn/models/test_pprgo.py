import pytest
import torch

from torch_geometric.datasets import KarateClub
from torch_geometric.nn.models.pprgo import PPRGo, pprgo_prune_features
from torch_geometric.testing import withPackage


@pytest.mark.parametrize('n_layers', [1, 4])
@pytest.mark.parametrize('dropout', [0.0, 0.2])
def test_pprgo_forward(n_layers, dropout):
    num_nodes = 100
    num_edges = 500
    num_features = 64
    num_classes = 16
    hidden_size = 64

    # Need to ensure edge_index is sorted + full so dim size checks are right
    # edge_index should contain all num_node unique nodes (ie every node is
    # connected to at least one destination, since we truncate by topk)
    edge_index = torch.stack([
        torch.arange(0, num_nodes).repeat(num_edges // num_nodes),
        torch.randint(0, num_nodes, [num_edges])
    ], dim=0)

    # Mimic the behavior of pprgo_prune_features manually
    # i.e., we expect node_embeds to be |V| x d
    node_embeds = torch.rand((num_nodes, num_features))
    node_embeds = node_embeds[edge_index[1], :]

    edge_weight = torch.rand(num_edges)

    model = PPRGo(num_features, num_classes, hidden_size, n_layers, dropout)
    pred = model(node_embeds, edge_index, edge_weight)
    assert pred.size() == (num_nodes, num_classes)


def test_pprgo_karate():
    data = KarateClub()[0]
    num_nodes = data.num_nodes

    data = pprgo_prune_features(data)
    data.edge_weight = torch.ones((data.edge_index.shape[1], ))

    assert data.x.shape[0] == data.edge_index.shape[1]
    num_classes = 16
    model = PPRGo(num_nodes, num_classes, hidden_size=64, n_layers=3,
                  dropout=0.0)
    pred = model(data.x, data.edge_index, data.edge_weight)
    assert pred.shape == (num_nodes, num_classes)


@pytest.mark.parametrize('n_power_iters', [1, 3])
@pytest.mark.parametrize('frac_predict', [1.0, 0.5])
@pytest.mark.parametrize('batch_size', [1, 9999])
@withPackage('torch_sparse')
def test_pprgo_inference(n_power_iters, frac_predict, batch_size):
    data = KarateClub()[0]
    num_nodes = data.num_nodes

    data = pprgo_prune_features(data)
    data.edge_weight = torch.rand(data.edge_index.shape[1])

    num_classes = 16
    model = PPRGo(num_nodes, num_classes, 64, 3, 0.0)
    logits = model.predict_power_iter(data.x, data.edge_index, n_power_iters,
                                      frac_predict, batch_size=batch_size)
    assert torch.all(~torch.isnan(logits))
    assert logits.shape == (data.x.shape[0], num_classes)
