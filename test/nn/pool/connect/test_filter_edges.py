import torch

from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectOutput


def test_filter_edges():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 1, 3, 2, 2]])
    edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
    batch = torch.tensor([0, 0, 1, 1])

    select_output = SelectOutput(
        node_index=torch.tensor([1, 2]),
        num_nodes=4,
        cluster_index=torch.tensor([0, 1]),
        num_clusters=2,
    )

    connect = FilterEdges()
    assert str(connect) == 'FilterEdges()'

    out1 = connect(select_output, edge_index, edge_attr, batch)
    assert out1.edge_index.tolist() == [[0, 1], [0, 1]]
    assert out1.edge_attr.tolist() == [3, 5]
    assert out1.batch.tolist() == [0, 1]
