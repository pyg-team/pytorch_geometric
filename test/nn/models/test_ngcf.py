import pytest
import torch

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.models import NGCF


@pytest.mark.parametrize('emb_size', [32, 64])
@pytest.mark.parametrize('lambda_reg', [0, 1e-4])
@pytest.mark.parametrize('node_dropout', [0, .25])
def test_ngcf_ranking(emb_size, lambda_reg, node_dropout):
    num_nodes = 500
    num_edges = 400
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_index, edge_weight = gcn_norm(edge_index=edge_index,
                                       num_nodes=num_nodes)
    edge_label_index = torch.randint(0, num_nodes, (2, 100))

    model = NGCF(num_nodes, emb_size, num_layers=2, node_dropout=node_dropout)
    assert str(model) == f'NGCF(500, {emb_size}, num_layers=2)'

    pred = model(edge_index, edge_label_index, edge_weight)
    assert pred.size() == (100, )

    loss = model.recommendation_loss(
        pos_edge_rank=pred[:50],
        neg_edge_rank=pred[50:],
        node_id=edge_index.unique(),
        lambda_reg=lambda_reg,
    )
    assert loss.dim() == 0 and loss > 0

    out = model.recommend(edge_index, edge_weight, k=2)
    assert out.size() == (500, 2)
    assert out.min() >= 0 and out.max() < 500

    src_index = torch.arange(0, 250)
    dst_index = torch.arange(250, 500)

    out = model.recommend(edge_index, edge_weight, src_index, dst_index, k=2)
    assert out.size() == (250, 2)
    assert out.min() >= 250 and out.max() < 500
