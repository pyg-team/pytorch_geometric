import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import (
    EdgeDeletionPolicy,
    EgoNetPolicy,
    NodeDeletionPolicy,
)


def test_node_deletion_policy():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.randn(4, 8)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    policy = NodeDeletionPolicy()
    subgraphs = policy.graph_to_subgraphs(data)

    # Ensure correct number of subgraphs are produced
    assert len(subgraphs) == 3

    # Ensure each node is deleted once
    deleted_nodes = set()
    for subgraph in subgraphs:
        deleted_nodes.add(subgraph.subgraph_id.item())
        assert subgraph.num_nodes == 3
    assert len(deleted_nodes) == 3


def test_edge_deletion_policy():
    # Create a small undirected graph with 4 nodes and 4 edges
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([1, 2, 3, 4])
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=4)
    policy = EdgeDeletionPolicy()

    # Apply the policy to the graph
    subgraphs = policy.graph_to_subgraphs(data)

    # Check that the policy generates the expected number of subgraphs
    assert len(subgraphs) == 3

    # Check that each subgraph has the expected number of nodes
    for subgraph in subgraphs:
        assert subgraph.num_nodes == 4

    # Check that each subgraph has the expected number of edges
    assert subgraphs[0].num_edges == 4
    assert subgraphs[1].num_edges == 4
    assert subgraphs[2].num_edges == 4

    # Check that the subgraphs have the expected subgraph IDs
    assert subgraphs[0].subgraph_id == 0
    assert subgraphs[1].subgraph_id == 1
    assert subgraphs[2].subgraph_id == 2

    # Check that the subgraphs have the expected subgraph node IDs
    assert torch.all(
        torch.eq(subgraphs[0].subgraph_n_id, torch.tensor([0, 1, 2, 3])))
    assert torch.all(
        torch.eq(subgraphs[1].subgraph_n_id, torch.tensor([0, 1, 2, 3])))
    assert torch.all(
        torch.eq(subgraphs[2].subgraph_n_id, torch.tensor([0, 1, 2, 3])))


@pytest.mark.parametrize('ego_plus', [True, False])
def test_ego_net_policy(ego_plus):
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [2, 2, 4, 4, 6, 6]])
    x = torch.randn(7, 16)
    data = Data(x=x, edge_index=edge_index)

    policy = EgoNetPolicy(num_hops=1, ego_plus=ego_plus)
    subgraphs = policy.graph_to_subgraphs(data)

    assert len(subgraphs) == 7
    for i in range(7):
        subgraph = subgraphs[i]
        assert subgraph.num_nodes == 7
        assert subgraph.subgraph_id == i
