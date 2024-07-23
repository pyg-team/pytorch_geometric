import pytest
import torch

from torch_geometric.nn.models import LightGCN


@pytest.mark.parametrize('embedding_dim', [32, 64])
@pytest.mark.parametrize('with_edge_weight', [False, True])
@pytest.mark.parametrize('lambda_reg', [0, 1e-4])
@pytest.mark.parametrize('alpha', [0, .25, torch.tensor([0.4, 0.3, 0.2])])
def test_lightgcn_ranking(embedding_dim, with_edge_weight, lambda_reg, alpha):
    num_nodes = 500
    num_edges = 400
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weight = torch.rand(num_edges) if with_edge_weight else None
    edge_label_index = torch.randint(0, num_nodes, (2, 100))

    model = LightGCN(num_nodes, embedding_dim, num_layers=2, alpha=alpha)
    assert str(model) == f'LightGCN(500, {embedding_dim}, num_layers=2)'

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


@pytest.mark.parametrize('embedding_dim', [32, 64])
@pytest.mark.parametrize('with_edge_weight', [False, True])
@pytest.mark.parametrize('alpha', [0, .25, torch.tensor([0.4, 0.3, 0.2])])
def test_lightgcn_link_prediction(embedding_dim, with_edge_weight, alpha):
    num_nodes = 500
    num_edges = 400
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weight = torch.rand(num_edges) if with_edge_weight else None
    edge_label_index = torch.randint(0, num_nodes, (2, 100))
    edge_label = torch.randint(0, 2, (edge_label_index.size(1), ))

    model = LightGCN(num_nodes, embedding_dim, num_layers=2, alpha=alpha)
    assert str(model) == f'LightGCN(500, {embedding_dim}, num_layers=2)'

    pred = model(edge_index, edge_label_index, edge_weight)
    assert pred.size() == (100, )

    loss = model.link_pred_loss(pred, edge_label)
    assert loss.dim() == 0 and loss > 0

    prob = model.predict_link(edge_index, edge_label_index, edge_weight,
                              prob=True)
    assert prob.size() == (100, )
    assert prob.min() > 0 and prob.max() < 1

    prob = model.predict_link(edge_index, edge_label_index, edge_weight,
                              prob=False)
    assert prob.size() == (100, )
    assert ((prob == 0) | (prob == 1)).sum() == 100
