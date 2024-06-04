import pytest
import random
import string
from torch_geometric.data import LargeGraphIndexer, TripletLike, Data, get_features_for_triplets
from torch_geometric.data.large_graph_indexer import NODE_PID, EDGE_RELATION
from typing import List
import torch

# create possible nodes and edges for graph
strkeys = string.ascii_letters + string.digits
NODE_POOL = set([random.sample(strkeys, 10) for i in range(1000)])
EDGE_POOL = set([random.sample(strkeys, 10) for i in range(50)])

def featurize(s: str) -> int:
    return int.from_bytes(s.encode(), 'little')

def sample_triplets(amount: int = 1) -> List[TripletLike]:
    trips = []
    for i in range(amount):
        trips.append(random.choice(NODE_POOL), random.choice(EDGE_POOL), random.choice(NODE_POOL))
    return trips

def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return h.lower(), r, t.lower()


def test_basic_collate():
    graphs = [sample_triplets(1000) for i in range(2)]

    indexer_0 = LargeGraphIndexer.from_triplets(graphs[0], pre_transform=preprocess_triplet)
    indexer_1 = LargeGraphIndexer.from_triplets(graphs[1], pre_transform=preprocess_triplet)

    big_indexer = LargeGraphIndexer.collate([indexer_0, indexer_1])

    assert len(indexer_0._nodes) + len(indexer_1._nodes) - len(indexer_0._nodes.keys() & indexer_1._nodes.keys()) == len(big_indexer._nodes)
    assert len(indexer_0._edges) + len(indexer_1._edges) - len(indexer_0._edges.keys() & indexer_1._edges.keys()) == len(big_indexer._edges)

    assert len(set(big_indexer._nodes.values())) == len(big_indexer._nodes)
    assert len(set(big_indexer._edges.values())) == len(big_indexer._edges)

    for node in (indexer_0._nodes.keys() | indexer_1._nodes.keys()):
        assert big_indexer.node_attr[NODE_PID][big_indexer._nodes[node]] == node


def test_large_graph_index():
    graphs = [sample_triplets(1000) for i in range(100)]

    node_feature_vecs = [featurize(s) for s in NODE_POOL]
    edge_feature_vecs = [featurize(s) for s in EDGE_POOL]

    def encode_graph_from_trips(triplets: List[TripletLike]) -> Data:
        seen_nodes = dict()
        edge_attrs = list()
        edge_idx = []
        for trip in triplets:
            h, r, t = trip
            seen_nodes[h] = len(seen_nodes) if h not in seen_nodes else seen_nodes[h]
            seen_nodes[t] = len(seen_nodes) if t not in seen_nodes else seen_nodes[t]
            edge_attrs.append(edge_feature_vecs[r])
            edge_idx.append((seen_nodes[h], seen_nodes[t]))
        
        x = torch.Tensor([node_feature_vecs[n] for n in seen_nodes.keys()])
        edge_idx = torch.LongTensor(edge_idx).T
        edge_attrs = torch.Tensor(edge_attrs)
        return Data(x=x, edge_index=edge_idx, edge_attr=edge_attrs)
    
    naive_graph_ds = [encode_graph_from_trips(triplets=trips) for trips in graphs]

    indexer = LargeGraphIndexer.collate([LargeGraphIndexer.from_triplets(g) for g in graphs])
    indexer.add_node_feature('x', torch.Tensor(node_feature_vecs))
    indexer.add_edge_feature('edge_attr', torch.Tensor(edge_feature_vecs), map_from_feature=EDGE_RELATION)
    large_graph_ds = [get_features_for_triplets(indexer=indexer, triplets=g, node_feature_name='x', edge_feature_name='edge_attr') for g in graphs]
    assert naive_graph_ds == large_graph_ds

def test_save_load(tmp_path):
    graphs = [sample_triplets(1000)]

    node_feature_vecs = [featurize(s) for s in NODE_POOL]
    edge_feature_vecs = [featurize(s) for s in EDGE_POOL]

    indexer = LargeGraphIndexer.from_triplets(graphs)
    indexer.add_node_feature('x', torch.Tensor(node_feature_vecs))
    indexer.add_edge_feature('edge_attr', torch.Tensor(edge_feature_vecs), map_from_feature=EDGE_RELATION)

    indexer.save(tmp_path)
    assert indexer == LargeGraphIndexer.from_disk(tmp_path)


