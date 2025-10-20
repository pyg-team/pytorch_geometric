import pytest
import torch

from torch_geometric.nn import TokenGT


@pytest.mark.parametrize("node_id_mode", ["orf", "laplacian"])
@pytest.mark.parametrize("use_two_graphs", [True, False])
def test_token_gt_basic(node_id_mode, use_two_graphs):
    d_p = 4
    embedding_dim = 8

    if use_two_graphs:
        # Two graphs: first has 3 nodes, second has 2 nodes
        num_nodes = 5
        x = torch.randint(1, 3, (num_nodes, 1))
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 1, 4, 3]])
        edge_attr = torch.randint(1, 3, (5, 1))
        ptr = torch.tensor([0, 3, 5])
        batch = torch.tensor([0, 0, 0, 1, 1])
        node_ids = torch.rand(num_nodes,
                              d_p) if node_id_mode == "laplacian" else None
    else:
        # Single graph with 5 nodes
        num_nodes = 5
        x = torch.randint(1, 3, (num_nodes, 1))
        edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 0, 3, 2, 4]])
        edge_attr = torch.randint(1, 3, (5, 1))
        ptr = torch.tensor([0, 5])
        batch = torch.tensor([0, 0, 0, 0, 0])
        node_ids = torch.rand(num_nodes,
                              d_p) if node_id_mode == "laplacian" else None

    model = TokenGT(
        num_atoms=3,
        num_edges=3,
        node_id_mode=node_id_mode,
        d_p=d_p,
        num_encoder_layers=1,
        embedding_dim=embedding_dim,
        ffn_embedding_dim=embedding_dim,
        num_attention_heads=1,
    )

    node_emb, edge_emb, graph_emb = model(x, edge_index, edge_attr, ptr, batch,
                                          node_ids)
    assert node_emb.shape == (num_nodes, embedding_dim)
    assert edge_emb.shape == (5, embedding_dim)
    if use_two_graphs:
        assert graph_emb.shape == (2, embedding_dim)
    else:
        assert graph_emb.shape == (1, embedding_dim)


def test_laplacian_node_ids_required():
    """Test that if node_id_mode is laplacian, node_ids must be provided."""
    model = TokenGT(
        num_atoms=10,
        num_edges=5,
        node_id_mode="laplacian",
        d_p=5,
        num_encoder_layers=1,
        embedding_dim=16,
        ffn_embedding_dim=16,
        num_attention_heads=1,
    )

    x = torch.randint(1, 10, (5, 1))
    edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 0, 3, 2, 4]])
    edge_attr = torch.randint(1, 5, (5, 1))
    ptr = torch.tensor([0, 5])
    batch = torch.tensor([0, 0, 0, 0, 0])

    with pytest.raises(
            AssertionError,
            match="node_ids must be provided when node_id_mode is laplacian",
    ):
        model(x, edge_index, edge_attr, ptr, batch, node_ids=None)


def test_d_p_inconsistency():
    """Test if d_p is inconsistent, an error is raised."""
    model = TokenGT(
        num_atoms=10,
        num_edges=5,
        node_id_mode="laplacian",
        d_p=5,
        num_encoder_layers=1,
        embedding_dim=16,
        ffn_embedding_dim=16,
        num_attention_heads=1,
    )

    x = torch.randint(1, 10, (5, 1))
    edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 0, 3, 2, 4]])
    edge_attr = torch.randint(1, 5, (5, 1))
    ptr = torch.tensor([0, 5])
    batch = torch.tensor([0, 0, 0, 0, 0])

    node_ids_wrong_d_p = torch.rand(5, 3)  # Should be 5, not 3

    with pytest.raises(AssertionError, match="node_ids must have 5 channels"):
        model(x, edge_index, edge_attr, ptr, batch,
              node_ids=node_ids_wrong_d_p)
