import pytest
import torch

from torch_geometric.nn import TokenGT


@pytest.mark.parametrize("dim_edge", [None, 16])
@pytest.mark.parametrize("include_graph_token", [True, False])
@pytest.mark.parametrize("is_laplacian_node_ids", [True, False])
def test_token_gt(dim_edge, include_graph_token, is_laplacian_node_ids):
    dim_node, d_p = 10, 5
    x = torch.rand(5, dim_node)
    edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 0, 3, 2, 4]])
    if dim_edge is not None:
        edge_attr = torch.rand(5, dim_edge)
    else:
        edge_attr = None
    ptr = torch.tensor([0, 2, 5])
    batch = torch.tensor([0, 0, 1, 1, 1])
    node_ids = torch.rand(5, d_p)

    model = TokenGT(
        dim_node=10,
        dim_edge=dim_edge,
        d_p=d_p,
        d=16,
        num_heads=1,
        num_encoder_layers=1,
        dim_feedforward=16,
        include_graph_token=include_graph_token,
        is_laplacian_node_ids=is_laplacian_node_ids,
    )
    model.reset_params()
    assert str(model) == "TokenGT(16)"

    node_emb, graph_emb = model(x, edge_index, edge_attr, ptr, batch, node_ids)
    assert node_emb.shape == (5, 16)
    if include_graph_token is True:
        assert graph_emb.shape == (2, 16)
    else:
        assert graph_emb is None
