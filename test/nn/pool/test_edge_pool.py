import torch
import torch_geometric.nn.pool.edge_pool as edge_pool


def test_compute_new_edges():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    old_to_new_node_idx = torch.tensor([0, 2, 0, 3, 1, 1])
    new_edge_index = edge_pool._compute_new_edges(
        old_to_new_node_idx,
        edge_index,
        n_nodes=4
    )

    goal_edge_index = torch.tensor([[0, 0, 0, 1, 2, 2, 3, 3],
                                    [0, 2, 3, 1, 0, 3, 0, 2]])
    assert torch.all(new_edge_index == goal_edge_index)
