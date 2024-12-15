import torch

from torch_geometric.data import Data
from torch_geometric.transforms import MultipleVirtualNodes

# modified the tests in test_virtual_node.py

def test_multiple_virtual_nodes():
    print("Test 1: Random assignments")
    assert str(MultipleVirtualNodes()) == 'MultipleVirtualNodes()'

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[2, 0, 2], [3, 1, 0]])
    edge_weight = torch.rand(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=x.size(0))

    # random assignments and uneven split
    data = MultipleVirtualNodes(n_to_add=3, clustering=False)(data)

    assert len(data) == 6

    assert data.x.size() == (7, 16)
    assert torch.allclose(data.x[:4], x)
    assert data.x[4:].abs().sum() == 0

    first_3_col = [row[:3] for row in data.edge_index.tolist()]
    assert first_3_col == edge_index # check that the original edges are unchanged
    assert data.edge_index.size() == (2, 11)

    virtual_nodes = {4, 5, 6}

    def validate_edge_index(edge_index):
        source_nodes = edge_index[0][3:11]
        target_nodes = edge_index[1][3:11]
        source_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        target_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for source in source_nodes:
            source_counts[source] += 1
        for target in target_nodes:
            target_counts[target] += 1
        assert source_counts[4] + source_counts[5] + source_counts[6] == 4
        assert target_counts[4] + target_counts[5] + target_counts[6] == 4
        for virtual in virtual_nodes:
            assert source_counts[virtual] > 0
            assert source_counts[virtual] < 3
            assert target_counts[virtual] > 0
            assert target_counts[virtual] < 3

    validate_edge_index(data.edge_index.tolist())

    assert data.edge_weight.size() == (11, )
    assert torch.allclose(data.edge_weight[:3], edge_weight)
    assert data.edge_weight[3:].abs().sum() == 8

    assert data.edge_attr.size() == (11, 8)
    assert torch.allclose(data.edge_attr[:3], edge_attr)
    assert data.edge_attr[3:].abs().sum() == 0

    assert data.num_nodes == 7

    assert data.edge_type.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    print("Test 1 passed\nTest 2: Clustering Assignments")

    # clustering assignments and uneven split
    data = MultipleVirtualNodes(n_to_add=3, clustering=True)(data)

    assert len(data) == 6

    assert data.x.size() == (7, 16)
    assert torch.allclose(data.x[:4], x)
    assert data.x[4:].abs().sum() == 0

    first_3_col = [row[:3] for row in data.edge_index.tolist()]
    assert first_3_col == edge_index # check that the original edges are unchanged
    assert data.edge_index.size() == (2, 11)
    validate_edge_index(data.edge_index.tolist())

    assert data.edge_weight.size() == (11, )
    assert torch.allclose(data.edge_weight[:3], edge_weight)
    assert data.edge_weight[3:].abs().sum() == 8

    assert data.edge_attr.size() == (11, 8)
    assert torch.allclose(data.edge_attr[:3], edge_attr)
    assert data.edge_attr[3:].abs().sum() == 0

    assert data.num_nodes == 7

    assert data.edge_type.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]