import random
import string
from typing import List

import pytest
import torch

from torch_geometric.data import (
    Data,
    LargeGraphIndexer,
    TripletLike,
    get_features_for_triplets,
)
from torch_geometric.data.large_graph_indexer import (
    EDGE_PID,
    EDGE_RELATION,
    NODE_PID,
)
from torch_geometric.typing import WITH_PT20

# create possible nodes and edges for graph
strkeys = string.ascii_letters + string.digits
NODE_POOL = list({"".join(random.sample(strkeys, 10)) for i in range(1000)})
EDGE_POOL = list({"".join(random.sample(strkeys, 10)) for i in range(50)})


def featurize(s: str) -> int:
    return int.from_bytes(s.encode(), 'little')


def sample_triplets(amount: int = 1) -> List[TripletLike]:
    trips = []
    for _ in range(amount):
        h, t = random.sample(NODE_POOL, k=2)
        r = random.sample(EDGE_POOL, k=1)[0]
        trips.append(tuple([h, r, t]))
    return trips


def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return h.lower(), r, t.lower()


def test_basic_collate():
    graphs = [sample_triplets(1000) for i in range(2)]

    indexer_0 = LargeGraphIndexer.from_triplets(
        graphs[0], pre_transform=preprocess_triplet)
    indexer_1 = LargeGraphIndexer.from_triplets(
        graphs[1], pre_transform=preprocess_triplet)

    big_indexer = LargeGraphIndexer.collate([indexer_0, indexer_1])

    assert len(indexer_0._nodes) + len(
        indexer_1._nodes) - len(indexer_0._nodes.keys()
                                & indexer_1._nodes.keys()) == len(
                                    big_indexer._nodes)
    assert len(indexer_0._edges) + len(
        indexer_1._edges) - len(indexer_0._edges.keys()
                                & indexer_1._edges.keys()) == len(
                                    big_indexer._edges)

    assert len(set(big_indexer._nodes.values())) == len(big_indexer._nodes)
    assert len(set(big_indexer._edges.values())) == len(big_indexer._edges)

    for node in (indexer_0._nodes.keys() | indexer_1._nodes.keys()):
        assert big_indexer.node_attr[NODE_PID][
            big_indexer._nodes[node]] == node


def test_large_graph_index():
    graphs = [sample_triplets(1000) for i in range(100)]

    # Preprocessing of trips lowercases nodes but not edges
    node_feature_vecs = {s.lower(): featurize(s.lower()) for s in NODE_POOL}
    edge_feature_vecs = {s: featurize(s) for s in EDGE_POOL}

    def encode_graph_from_trips(triplets: List[TripletLike]) -> Data:
        seen_nodes = dict()
        edge_attrs = list()
        edge_idx = []
        for trip in triplets:
            trip = preprocess_triplet(trip)
            h, r, t = trip
            seen_nodes[h] = len(
                seen_nodes) if h not in seen_nodes else seen_nodes[h]
            seen_nodes[t] = len(
                seen_nodes) if t not in seen_nodes else seen_nodes[t]
            edge_attrs.append(edge_feature_vecs[r])
            edge_idx.append((seen_nodes[h], seen_nodes[t]))

        x = torch.Tensor([node_feature_vecs[n] for n in seen_nodes.keys()])
        edge_idx = torch.LongTensor(edge_idx).T
        edge_attrs = torch.Tensor(edge_attrs)
        return Data(x=x, edge_index=edge_idx, edge_attr=edge_attrs)

    naive_graph_ds = [
        encode_graph_from_trips(triplets=trips) for trips in graphs
    ]

    indexer = LargeGraphIndexer.collate([
        LargeGraphIndexer.from_triplets(g, pre_transform=preprocess_triplet)
        for g in graphs
    ])
    indexer_nodes = indexer.get_unique_node_features()
    indexer_node_vals = torch.Tensor(
        [node_feature_vecs[n] for n in indexer_nodes])
    indexer_edges = indexer.get_unique_edge_features(
        feature_name=EDGE_RELATION)
    indexer_edge_vals = torch.Tensor(
        [edge_feature_vecs[e] for e in indexer_edges])
    indexer.add_node_feature('x', indexer_node_vals)
    indexer.add_edge_feature('edge_attr', indexer_edge_vals,
                             map_from_feature=EDGE_RELATION)
    large_graph_ds = [
        get_features_for_triplets(indexer=indexer, triplets=g,
                                  node_feature_name='x',
                                  edge_feature_name='edge_attr',
                                  pre_transform=preprocess_triplet)
        for g in graphs
    ]

    for ds in large_graph_ds:
        assert NODE_PID in ds
        assert EDGE_PID in ds
        assert "node_idx" in ds
        assert "edge_idx" in ds

    def results_are_close_enough(ground_truth: Data, new_method: Data,
                                 thresh=.99):
        def _sorted_tensors_are_close(tensor1, tensor2):
            return torch.all(
                torch.isclose(tensor1.sort()[0],
                              tensor2.sort()[0]) > thresh)

        def _graphs_are_same(tensor1, tensor2):
            if not WITH_PT20:
                pytest.skip(
                    "This test requires a PyG version with NetworkX as a " +
                    "dependency.")
            import networkx as nx
            return nx.weisfeiler_lehman_graph_hash(nx.Graph(
                tensor1.T)) == nx.weisfeiler_lehman_graph_hash(
                    nx.Graph(tensor2.T))
            return True
        return _sorted_tensors_are_close(
            ground_truth.x, new_method.x) \
            and _sorted_tensors_are_close(
                ground_truth.edge_attr, new_method.edge_attr) \
            and _graphs_are_same(
                ground_truth.edge_index, new_method.edge_index)

    for dsets in zip(naive_graph_ds, large_graph_ds):
        assert results_are_close_enough(*dsets)


def test_save_load(tmp_path):
    graph = sample_triplets(1000)

    node_feature_vecs = {s: featurize(s) for s in NODE_POOL}
    edge_feature_vecs = {s: featurize(s) for s in EDGE_POOL}

    indexer = LargeGraphIndexer.from_triplets(graph)
    indexer_nodes = indexer.get_unique_node_features()
    indexer_node_vals = torch.Tensor(
        [node_feature_vecs[n] for n in indexer_nodes])
    indexer_edges = indexer.get_unique_edge_features(
        feature_name=EDGE_RELATION)
    indexer_edge_vals = torch.Tensor(
        [edge_feature_vecs[e] for e in indexer_edges])
    indexer.add_node_feature('x', indexer_node_vals)
    indexer.add_edge_feature('edge_attr', indexer_edge_vals,
                             map_from_feature=EDGE_RELATION)

    indexer.save(str(tmp_path))
    assert indexer == LargeGraphIndexer.from_disk(str(tmp_path))
