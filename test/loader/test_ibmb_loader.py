import torch

from torch_geometric.data import Data
from torch_geometric.loader import IBMBBatchLoader, IBMBNodeLoader


def test_graph_ibmb():
    edge_index = torch.tensor([[
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
        4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9,
        9, 10, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14,
        14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20,
        20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24,
        25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29,
        29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33
    ],
                               [
                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13,
                                   17, 19, 21, 31, 0, 1, 2, 3, 7, 13, 17, 19,
                                   21, 30, 0, 1, 2, 3, 7, 8, 9, 13, 27, 28, 32,
                                   0, 1, 2, 3, 7, 12, 13, 0, 4, 6, 10, 0, 5, 6,
                                   10, 16, 0, 4, 5, 6, 16, 0, 1, 2, 3, 7, 0, 2,
                                   8, 30, 32, 33, 2, 9, 33, 0, 4, 5, 10, 0, 11,
                                   0, 3, 12, 0, 1, 2, 3, 13, 33, 14, 32, 33,
                                   15, 32, 33, 5, 6, 16, 0, 1, 17, 18, 32, 33,
                                   0, 1, 19, 33, 20, 32, 33, 0, 1, 21, 22, 32,
                                   33, 23, 25, 27, 29, 32, 33, 24, 25, 27, 31,
                                   23, 24, 25, 31, 26, 29, 33, 2, 23, 24, 27,
                                   33, 2, 28, 31, 33, 23, 26, 29, 32, 33, 1, 8,
                                   30, 32, 33, 0, 24, 25, 28, 31, 32, 33, 2, 8,
                                   14, 15, 18, 20, 22, 23, 29, 30, 31, 32, 33,
                                   8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26,
                                   27, 28, 29, 30, 31, 32, 33
                               ]], dtype=torch.long)

    graph = Data(x=torch.arange(34 * 3).to(torch.float).reshape(34, 3),
                 y=torch.arange(34) % 3, edge_index=edge_index)

    train_indices = torch.tensor(
        [0, 3, 4, 5, 7, 8, 11, 15, 20, 21, 23, 25, 29, 31, 32, 33],
        dtype=torch.long)

    batch_loader = IBMBBatchLoader(
        graph, batch_order='order', num_partitions=4,
        output_indices=train_indices, return_edge_index_type='edge_index',
        batch_expand_ratio=1., metis_output_weight=None, batch_size=1,
        shuffle=False)

    batches1 = []
    for b in batch_loader:
        batches1.append(b)

    assert len(batches1) == 4
    assert sum([b.output_node_mask.sum()
                for b in batches1]) == len(train_indices)

    node_loader = IBMBNodeLoader(graph, batch_order='order',
                                 output_indices=train_indices,
                                 return_edge_index_type='edge_index',
                                 num_auxiliary_node_per_output=4,
                                 num_output_nodes_per_batch=4, batch_size=1,
                                 shuffle=False)

    batches2 = []
    for b in node_loader:
        batches2.append(b)

    assert len(batches2) == 4
    assert sum([b.output_node_mask.sum()
                for b in batches2]) == len(train_indices)
