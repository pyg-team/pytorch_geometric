import networkx as nx
import torch
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.loader import IBMBBatchLoader, IBMBNodeLoader
from torch_geometric.utils import add_remaining_self_loops, to_undirected


def test_graph_ibmb():
    G = nx.karate_club_graph()
    edge_index = torch.tensor(list(G.edges)).t()
    num_nodes = len(G.nodes)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    x = torch.arange(num_nodes * 3).reshape(num_nodes, 3)
    y = torch.arange(num_nodes) % 3

    graph = Data(x=x, y=y, edge_index=edge_index)

    torch.manual_seed(42)
    train_indices = torch.unique(torch.randint(high=33, size=(20, )))

    batch_loader = IBMBBatchLoader(
        graph, batch_order='order', num_partitions=4,
        output_indices=train_indices, return_edge_index_type='edge_index',
        batch_expand_ratio=1., metis_output_weight=None, batch_size=1,
        shuffle=False)
    batches = [b for b in batch_loader]
    assert len(batches) == 4
    assert sum([b.output_node_mask.sum()
                for b in batches]) == len(train_indices)

    node_loader = IBMBNodeLoader(graph, batch_order='order',
                                 output_indices=train_indices,
                                 return_edge_index_type='edge_index',
                                 num_auxiliary_node_per_output=4,
                                 num_output_nodes_per_batch=4, batch_size=1,
                                 shuffle=False)
    batches = [b for b in node_loader]
    assert len(batches) == 4
    assert sum([b.output_node_mask.sum()
                for b in batches]) == len(train_indices)

    batch_loader = IBMBBatchLoader(
        graph, batch_order='order', num_partitions=8,
        output_indices=train_indices, return_edge_index_type='edge_index',
        batch_expand_ratio=1., metis_output_weight=None, batch_size=2,
        shuffle=False)

    batches = [b for b in batch_loader]
    assert len(batches) == 4
    assert sum([b.output_node_mask.sum()
                for b in batches]) == len(train_indices)

    node_loader = IBMBNodeLoader(graph, batch_order='order',
                                 output_indices=train_indices,
                                 return_edge_index_type='edge_index',
                                 num_auxiliary_node_per_output=4,
                                 num_output_nodes_per_batch=2, batch_size=2,
                                 shuffle=False)

    batches = [b for b in node_loader]
    assert len(batches) == 4
    assert sum([b.output_node_mask.sum()
                for b in batches]) == len(train_indices)

    batch_loader = IBMBBatchLoader(
        graph, batch_order='sample', num_partitions=8,
        output_indices=train_indices, return_edge_index_type='adj',
        batch_expand_ratio=1., metis_output_weight=None, batch_size=2,
        shuffle=False)

    batches = [b for b in batch_loader]
    for b in batches:
        assert isinstance(b.edge_index, SparseTensor)

    node_loader = IBMBNodeLoader(graph, batch_order='sample',
                                 output_indices=train_indices,
                                 return_edge_index_type='adj',
                                 num_auxiliary_node_per_output=4,
                                 num_output_nodes_per_batch=2, batch_size=2,
                                 shuffle=False)

    batches = [b for b in node_loader]
    for b in batches:
        assert isinstance(b.edge_index, SparseTensor)
