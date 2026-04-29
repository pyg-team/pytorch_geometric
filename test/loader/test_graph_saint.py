import torch

from torch_geometric.data import Data
from torch_geometric.loader import (
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
)
from torch_geometric.testing import withPackage


@withPackage('torch_sparse')
def test_graph_saint():
    adj = torch.tensor([
        [+1, +2, +3, +0, +4, +0],
        [+5, +6, +0, +7, +0, +8],
        [+9, +0, 10, +0, 11, +0],
        [+0, 12, +0, 13, +0, 14],
        [15, +0, 16, +0, 17, +0],
        [+0, 18, +0, 19, +0, 20],
    ])

    edge_index = adj.nonzero(as_tuple=False).t()
    edge_id = adj[edge_index[0], edge_index[1]]
    x = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
    ])
    n_id = torch.arange(6)
    data = Data(edge_index=edge_index, x=x, n_id=n_id, edge_id=edge_id,
                num_nodes=6)

    loader = GraphSAINTNodeSampler(data, batch_size=3, num_steps=4,
                                   sample_coverage=10, log=False)

    assert len(loader) == 4
    for sample in loader:
        assert sample.num_nodes <= data.num_nodes
        assert sample.n_id.min() >= 0 and sample.n_id.max() < 6
        assert sample.num_nodes == sample.n_id.numel()
        assert sample.x.tolist() == x[sample.n_id].tolist()
        assert sample.edge_index.min() >= 0
        assert sample.edge_index.max() < sample.num_nodes
        assert sample.edge_id.min() >= 1 and sample.edge_id.max() <= 21
        assert sample.edge_id.numel() == sample.num_edges
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges

    loader = GraphSAINTEdgeSampler(data, batch_size=2, num_steps=4,
                                   sample_coverage=10, log=False)

    assert len(loader) == 4
    for sample in loader:
        assert sample.num_nodes <= data.num_nodes
        assert sample.n_id.min() >= 0 and sample.n_id.max() < 6
        assert sample.num_nodes == sample.n_id.numel()
        assert sample.x.tolist() == x[sample.n_id].tolist()
        assert sample.edge_index.min() >= 0
        assert sample.edge_index.max() < sample.num_nodes
        assert sample.edge_id.min() >= 1 and sample.edge_id.max() <= 21
        assert sample.edge_id.numel() == sample.num_edges
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges

    loader = GraphSAINTRandomWalkSampler(data, batch_size=2, walk_length=1,
                                         num_steps=4, sample_coverage=10,
                                         log=False)

    assert len(loader) == 4
    for sample in loader:
        assert sample.num_nodes <= data.num_nodes
        assert sample.n_id.min() >= 0 and sample.n_id.max() < 6
        assert sample.num_nodes == sample.n_id.numel()
        assert sample.x.tolist() == x[sample.n_id].tolist()
        assert sample.edge_index.min() >= 0
        assert sample.edge_index.max() < sample.num_nodes
        assert sample.edge_id.min() >= 1 and sample.edge_id.max() <= 21
        assert sample.edge_id.numel() == sample.num_edges
        assert sample.node_norm.numel() == sample.num_nodes
        assert sample.edge_norm.numel() == sample.num_edges
