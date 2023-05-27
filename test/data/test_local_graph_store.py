import pytest
import torch

from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout
from torch_geometric.testing import MyGraphStore, get_random_edge_index
from torch_geometric.data.local_graph_store import LocalGraphStore


def test_local_graph_store():
    
    graph_store = LocalGraphStore()

    edge_index = get_random_edge_index(100, 100, 300)
    edge_ids = torch.tensor([1, 2, 3, 5, 8, 4], dtype=torch.int64) 

    # init_graph by edge_index
    node_num = edge_index[0].size()
    graph_store.put_edge_index(
        edge_index=edge_index,
        edge_type=None, layout='coo', size=(node_num, node_num))

    # put edge_ids into graphstore
    graph_store.set_edge_ids(edge_ids=edge_ids,
        edge_attr=EdgeAttr(edge_type=None, layout='coo', size=(node_num, node_num)))

    # get the edge_index/edge_ids from graphstore
    edge_attrs=graph_store.get_all_edge_attrs()
    for item in edge_attrs:
       edge_idx = graph_store.get_edge_index(item)
       ids = graph_store.get_edge_ids(item)


    # verify.
    assert torch.equal(edge_idx, edge_index)
    assert torch.equal(ids, edge_ids)

