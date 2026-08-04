import copy

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import MultipleVirtualNodes

# modified the tests in test_virtual_node.py


def test_multiple_virtual_nodes():
    print("Test 1: Random assignments")
    assert str(MultipleVirtualNodes(
        n_to_add=3, clustering=False)) == 'MultipleVirtualNodes()'

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[2, 0, 2], [3, 1, 0]])
    edge_weight = torch.rand(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=x.size(0))

    original_data = copy.deepcopy(data)
    # random assignments and uneven split
    data = MultipleVirtualNodes(n_to_add=3, clustering=False)(data)

    assert len(data) == 6

    assert data.x.size() == (7, 16)
    assert torch.allclose(data.x[:4], x)
    assert data.x[4:].abs().sum() == 0

    first_3_col = [row[:3] for row in data.edge_index.tolist()]
    assert first_3_col == [[2, 0, 2],
                           [3, 1,
                            0]]  # check that the original edges are unchanged
    num_total_edges = 11
    assert data.edge_index.size() == (2, num_total_edges)

    def validate_edge_index(edge_index, n_original_nodes, n_added_nodes):
        source_nodes = edge_index[0][n_original_nodes - 1:num_total_edges]
        target_nodes = edge_index[1][n_original_nodes - 1:num_total_edges]
        source_counts = {i: 0 for i in range(n_original_nodes + n_added_nodes)}
        target_counts = {i: 0 for i in range(n_original_nodes + n_added_nodes)}
        original_node_indices = [i for i in range(n_original_nodes)]
        virtual_node_indices = [
            n_original_nodes + i for i in range(n_added_nodes)
        ]

        for i in range(len(source_nodes)):
            source_counts[source_nodes[i]] += 1
            # check that virtual nodes are not connected to each other and original nodes are not connected to each other
            if source_nodes[i] in virtual_node_indices:
                assert target_nodes[i] in original_node_indices
            else:
                assert target_nodes[i] in virtual_node_indices
        for target in target_nodes:
            target_counts[target] += 1

        total_virtual_source_count = 0
        total_virtual_target_count = 0
        # check virtual nodes' edges have been added in the correct way
        for i in range(n_added_nodes):
            assert source_counts[n_original_nodes + i] > 0
            assert source_counts[n_original_nodes + i] < 3
            assert target_counts[n_original_nodes + i] > 0
            assert target_counts[n_original_nodes + i] < 3
            total_virtual_source_count += source_counts[n_original_nodes + i]
            total_virtual_target_count += target_counts[n_original_nodes + i]
        # check original nodes
        for j in range(n_original_nodes):
            assert source_counts[j] == 1
            assert target_counts[j] == 1

        assert total_virtual_source_count == n_original_nodes
        assert total_virtual_target_count == n_original_nodes

    validate_edge_index(data.edge_index.tolist(), 4, 3)

    assert data.edge_weight.size() == (11, )
    assert torch.allclose(data.edge_weight[:3], edge_weight)
    assert data.edge_weight[3:].abs().sum() == 8

    assert data.edge_attr.size() == (11, 8)
    assert torch.allclose(data.edge_attr[:3], edge_attr)
    assert data.edge_attr[3:].abs().sum() == 0

    assert data.num_nodes == 7

    assert data.edge_type.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    print("Test 1 passed\nTest 2: Clustering Assignments")

    # Test 2: clustering assignments
    data = MultipleVirtualNodes(n_to_add=2, clustering=True)(original_data)

    assert len(data) == 6
    assert data.x.size() == (6, 16)
    assert torch.allclose(data.x[:4], x)
    assert data.x[4:].abs().sum() == 0

    first_3_col = [row[:3] for row in data.edge_index.tolist()]
    assert first_3_col == [[2, 0, 2],
                           [3, 1,
                            0]]  # check that the original edges are unchanged
    assert data.edge_index.size() == (2, num_total_edges)
    validate_edge_index(data.edge_index.tolist(), 4, 2)

    assert data.edge_weight.size() == (num_total_edges, )
    assert torch.allclose(data.edge_weight[:3], edge_weight)
    assert data.edge_weight[3:].abs().sum() == 8

    assert data.edge_attr.size() == (num_total_edges, 8)
    assert torch.allclose(data.edge_attr[:3], edge_attr)
    assert data.edge_attr[3:].abs().sum() == 0

    assert data.num_nodes == 6

    assert data.edge_type.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    print(
        "Test 2 passed\nTest 3: Clustering Assignments with an empty cluster. Should add 2 virtual nodes instead of the specified 3"
    )

    # Test 2: clustering assignments
    data = MultipleVirtualNodes(n_to_add=3, clustering=True)(original_data)

    assert len(data) == 6
    assert data.x.size() == (6, 16)
    assert torch.allclose(data.x[:4], x)
    assert data.x[4:].abs().sum() == 0

    first_3_col = [row[:3] for row in data.edge_index.tolist()]
    assert first_3_col == [[2, 0, 2],
                           [3, 1,
                            0]]  # check that the original edges are unchanged
    assert data.edge_index.size() == (2, num_total_edges)
    validate_edge_index(data.edge_index.tolist(), 4, 2)

    assert data.edge_weight.size() == (num_total_edges, )
    assert torch.allclose(data.edge_weight[:3], edge_weight)
    assert data.edge_weight[3:].abs().sum() == 8

    assert data.edge_attr.size() == (num_total_edges, 8)
    assert torch.allclose(data.edge_attr[:3], edge_attr)
    assert data.edge_attr[3:].abs().sum() == 0

    assert data.num_nodes == 6

    assert data.edge_type.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


if __name__ == '__main__':
    test_multiple_virtual_nodes()
    print("All tests passed successfully!")
