import torch

from torch_geometric import seed_everything
from torch_geometric.datasets import BA2MotifDataset


def test_ba2motif_dataset():
    num_nodes, num_edges, num_graphs = 50, 5, 100
    dataset = BA2MotifDataset(
        num_nodes=num_nodes, num_edges=num_edges, num_graphs=num_graphs
    )
    assert str(dataset) == (
        f"BA2MotifDataset({num_graphs}, "
        f"BAGraph(num_nodes={num_nodes}, num_edges={num_edges}), "
        f"num_graphs={num_graphs} num_house_graphs={int(num_graphs/2)} "
        f"num_cycle_graphs={int(num_graphs/2)})"
    )
    # dataset level checks
    assert len(dataset) == 100
    assert torch.equal(torch.cat([
        torch.zeros(int(num_graphs / 2)).type(torch.long),
        torch.ones(int(num_graphs / 2)).type(torch.long),
    ]), dataset.data.y)
    
    # the ba-graph with house motif attached
    data = dataset[0]
    assert len(data) == 4
    assert data.num_nodes == num_nodes + 5
    assert data.edge_index.min() == 0
    assert data.edge_index.max() == data.num_nodes - 1
    assert data.node_mask.size() == (data.num_nodes, )
    assert data.edge_mask.size() == (data.num_edges, )
    assert data.node_mask.min() == 0 and data.node_mask.max() == 1
    assert data.node_mask.sum() == 5
    assert data.edge_mask.min() == 0 and data.edge_mask.max() == 1
    assert data.edge_mask.sum() == 12

    # the ba-graph with cycle motif attached
    data = dataset[50]
    assert data.num_nodes == num_nodes + 5
    assert data.edge_index.min() == 0
    assert data.edge_index.max() == data.num_nodes - 1
    assert dataset.data.y.min() == 0 and dataset.data.y.max() == 1
    assert data.node_mask.size() == (data.num_nodes, )
    assert data.edge_mask.size() == (data.num_edges, )
    assert data.node_mask.min() == 0 and data.node_mask.max() == 1
    assert data.node_mask.sum() == 5
    assert data.edge_mask.min() == 0 and data.edge_mask.max() == 1
    assert data.edge_mask.sum() == 5


def test_explainer_dataset_reproducibility():
    num_nodes, num_edges, num_graphs = 50, 5, 100

    seed_everything(12345)
    data1 = BA2MotifDataset(num_nodes, num_edges, num_graphs)

    seed_everything(12345)
    data2 = BA2MotifDataset(num_nodes, num_edges, num_graphs)

    # check overall y labels are same
    assert torch.equal(data1.data.y, data2.data.y)
    # check for ba-graph with house motif
    assert torch.equal(data1[0].edge_index, data2[0].edge_index)
    # check for ba-graph with cycle motif
    assert torch.equal(data1[50].edge_index, data2[50].edge_index)
