import pytest
from itertools import product

import torch
from torch_geometric.nn.models import LightGCN

emb_dims = [32, 64]
lambda_reg = [0, 1e-4]
alpha = [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]]


@pytest.mark.parametrize('embedding_dim,lambda_reg,alpha',
                         product(emb_dims, lambda_reg, alpha))
def test_lightgcn_rankings(embedding_dim, lambda_reg, alpha):
    x = torch.randn(500, embedding_dim)
    edge_index = torch.randint(0, x.size(0), (2, 400))
    edge_label_index = torch.randint(0, x.size(0), (2, 100))
    rec_indices = torch.arange(0, x.size(0))

    alpha_t = torch.FloatTensor(alpha)
    lightGCN = LightGCN(3, x.size(0), embedding_dim, 'ranking', lambda_reg,
                        alpha=alpha_t)

    assert lightGCN.__repr__() == 'LightGCN(num_layers=3, ' \
                                  f'num_nodes={x.size(0)}, ' \
                                  f'embedding_dim={embedding_dim}, ' \
                                  'objective="ranking")'

    rankings = lightGCN(edge_index, edge_label_index)
    assert rankings.size(0) == 100

    loss = lightGCN.loss(pos_ranks=rankings[:-1], neg_ranks=rankings[1:])
    assert loss.dim() == 0

    embeddings = lightGCN.get_embeddings(edge_index)
    recommendations = lightGCN.recommend(edge_index, rec_indices[:200],
                                         rec_indices[200:], topK=2)
    assert recommendations.size() == (200, 2)

    first_indices = recommendations[:, 0]
    second_indices = recommendations[:, 1]

    first_rec = torch.mm(embeddings, embeddings[first_indices].t())
    second_rec = torch.mm(embeddings, embeddings[second_indices].t())
    rec_mismatch = first_rec.diagonal() < second_rec.diagonal()
    assert torch.sum(rec_mismatch).item() == 0


@pytest.mark.parametrize('embedding_dim,lambda_reg,alpha',
                         product(emb_dims, lambda_reg, alpha))
def test_lightgcn_link_predictions(embedding_dim, lambda_reg, alpha):
    x = torch.randn(500, embedding_dim)
    edge_index = torch.randint(0, x.size(0), (2, 400))
    edge_label_index = torch.randint(0, x.size(0), (2, 100))
    edge_label = torch.randint_like(edge_label_index[0], 0, 2)

    alpha_t = torch.FloatTensor(alpha)
    lightGCN = LightGCN(3, x.size(0), embedding_dim, 'link_prediction',
                        lambda_reg, alpha=alpha_t)

    assert lightGCN.__repr__() == 'LightGCN(num_layers=3, '\
                                  f'num_nodes={x.size(0)}, '\
                                  f'embedding_dim={embedding_dim}, '\
                                  'objective="link_prediction")'

    preds = lightGCN(edge_index, edge_label_index)
    assert preds.size(0) == 100

    loss = lightGCN.loss(preds, edge_label)
    assert loss.dim() == 0

    labels = lightGCN.predict_link(edge_index, edge_label_index)
    assert labels.size(0) == 100
