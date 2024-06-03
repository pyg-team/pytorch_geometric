import os
import pickle as pkl
from dataclasses import dataclass
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)
import shutil

import torch
from torch_geometric.data import Data, FeatureStore, GraphStore
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore

# Is there a multiprocessing-friendly implementation of this?

TripletLike = Tuple[Hashable, Hashable, Hashable]


def ordered_set(values: Iterable[Hashable]) -> List[Hashable]:
    return list(dict.fromkeys(values))


# TODO: Refactor Node and Edge funcs and attrs to be accessible via an Enum?

NODE_PID = "pid"

NODE_KEYS = {NODE_PID}

EDGE_PID = "e_pid"
EDGE_HEAD = "h"
EDGE_RELATION = "r"
EDGE_TAIL = "t"
EDGE_INDEX = "edge_idx"

EDGE_KEYS = {EDGE_PID, EDGE_HEAD, EDGE_RELATION, EDGE_TAIL, EDGE_INDEX}


@dataclass
class MappedFeature:
    name: str
    values: Sequence[Any]

    def __eq__(self, value: "MappedFeature") -> bool:
        eq = self.name == value.name
        if isinstance(self.values, torch.Tensor):
            eq &= torch.equal(self.values, value.values)
        else:
            eq &= self.values == value.values
        return eq


class LargeGraphIndexer:
    """For a dataset that consists of mulitiple subgraphs that are assumed to
    be part of a much larger graph, collate the values into a large graph store
    to save resources
    """

    def __init__(
        self,
        nodes: Iterable[Hashable],
        edges: Iterable[TripletLike],
        node_attr: Optional[Dict[str, List[Any]]] = None,
        edge_attr: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        self._nodes: Dict[Hashable, int] = dict()
        self._edges: Dict[TripletLike, int] = dict()

        self._mapped_node_features: Set[str] = set()
        self._mapped_edge_features: Set[str] = set()

        if len(nodes) != len(set(nodes)):
            raise AttributeError("Nodes need to be unique")
        if len(edges) != len(set(edges)):
            raise AttributeError("Edges need to be unique")

        if node_attr is not None:
            # TODO: Validity checks btw nodes and node_attr
            self.node_attr = node_attr
            if NODE_KEYS & set(self.node_attr.keys()) != NODE_KEYS:
                raise AttributeError(
                    f"Invalid node_attr object. Missing {NODE_KEYS - set(self.node_attr.keys())}"
                )
            elif self.node_attr[NODE_PID] != nodes:
                raise AttributeError(f"Nodes provided do not match those in node_attr")
        else:
            self.node_attr = dict()
            self.node_attr[NODE_PID] = nodes

        for i, node in enumerate(self.node_attr[NODE_PID]):
            self._nodes[node] = i

        if edge_attr is not None:
            # TODO: Validity checks btw edges and edge_attr
            self.edge_attr = edge_attr

            if EDGE_KEYS & set(self.edge_attr.keys()) != EDGE_KEYS:
                raise AttributeError(
                    f"Invalid edge_attr object. Missing {EDGE_KEYS - set(self.edge_attr.keys())}"
                )
            elif self.node_attr[EDGE_PID] != edges:
                raise AttributeError(f"Edges provided do not match those in edge_attr")

        else:
            self.edge_attr = dict()
            for default_key in EDGE_KEYS:
                self.edge_attr[default_key] = list()
            self.edge_attr[EDGE_PID] = edges

            for i, tup in enumerate(edges):
                h, r, t = tup
                self.edge_attr[EDGE_HEAD].append(h)
                self.edge_attr[EDGE_RELATION].append(r)
                self.edge_attr[EDGE_TAIL].append(t)
                self.edge_attr[EDGE_INDEX].append((self._nodes[h], self._nodes[t]))

        for i, tup in enumerate(edges):
            self._edges[tup] = i

    @classmethod
    def from_triplets(
        cls,
        triplets: Iterable[TripletLike],
        pre_transform: Optional[Callable[[TripletLike], TripletLike]] = None,
    ) -> "LargeGraphIndexer":
        # NOTE: Right now assumes that all trips can be loaded into memory
        nodes = set()
        edges = set()

        if pre_transform is not None:

            def apply_transform(trips: Iterable[TripletLike]) -> Iterator[TripletLike]:
                for trip in trips:
                    yield pre_transform(trip)

            triplets = apply_transform(triplets)

        for h, r, t in triplets:

            for node in (h, t):
                nodes.add(node)

            edge_idx = (h, r, t)
            edges.add(edge_idx)

        return cls(list(nodes), list(edges))

    @classmethod
    def collate(cls, graphs: Iterable["LargeGraphIndexer"]) -> "LargeGraphIndexer":
        # FIXME Needs to merge node attrs and edge attrs?
        trips = chain.from_iterable([graph.to_triplets() for graph in graphs])
        return cls.from_triplets(trips)

    def get_unique_node_features(self, feature_name: str = NODE_PID) -> List[Hashable]:
        try:
            if feature_name in self._mapped_node_features:
                raise IndexError(f"Only non-mapped features can be retrieved uniquely.")
            return ordered_set(self.get_node_features(feature_name))

        except KeyError:
            raise AttributeError(f"Nodes do not have a feature called {feature_name}")

    def add_node_feature(
        self,
        new_feature_name: str,
        new_feature_vals: Sequence[Any],
        map_from_feature: str = NODE_PID,
    ) -> None:

        if new_feature_name in self.node_attr:
            raise AttributeError("Features cannot be overridden once created")
        if map_from_feature in self._mapped_node_features:
            raise AttributeError(f"{map_from_feature} is already a feature mapping.")

        feature_keys = self.get_unique_node_features(map_from_feature)
        if len(feature_keys) != len(new_feature_vals):
            raise AttributeError(
                f"Expected encodings for {len(feature_keys)} unique features, but got {len(new_feature_vals)} encodings."
            )

        if map_from_feature == NODE_PID:
            self.node_attr[new_feature_name] = new_feature_vals
        else:
            self.node_attr[new_feature_name] = MappedFeature(
                name=map_from_feature, values=new_feature_vals
            )
            self._mapped_node_features.add(new_feature_name)

    def get_node_features(
        self, feature_name: str = NODE_PID, pids: Optional[Iterable[Hashable]] = None
    ) -> List[Any]:
        if feature_name in self._mapped_node_features:
            values = self.node_attr[feature_name].values
        else:
            values = self.node_attr[feature_name]
        if isinstance(values, torch.Tensor):
            idxs = list(
                self.get_node_features_iter(feature_name, pids, index_only=True)
            )
            return values[idxs]
        return list(self.get_node_features_iter(feature_name, pids))

    def get_node_features_iter(
        self,
        feature_name: str = NODE_PID,
        pids: Optional[Iterable[Hashable]] = None,
        index_only: bool = False,
    ) -> Iterator[Any]:
        if pids is None:
            pids = self.node_attr[NODE_PID]

        if feature_name in self._mapped_node_features:
            feature_map_info = self.node_attr[feature_name]
            from_feature_name, to_feature_vals = (
                feature_map_info.name,
                feature_map_info.values,
            )
            from_feature_vals = self.get_unique_node_features(from_feature_name)
            feature_mapping = {k: i for i, k in enumerate(from_feature_vals)}

            for pid in pids:
                idx = self._nodes[pid]
                from_feature_val = self.node_attr[from_feature_name][idx]
                to_feature_idx = feature_mapping[from_feature_val]
                if index_only:
                    yield to_feature_idx
                else:
                    yield to_feature_vals[to_feature_idx]
        else:
            for pid in pids:
                idx = self._nodes[pid]
                if index_only:
                    yield idx
                else:
                    yield self.node_attr[feature_name][idx]

    def get_unique_edge_features(self, feature_name: str = EDGE_PID) -> List[Hashable]:
        try:
            if feature_name in self._mapped_edge_features:
                raise IndexError(f"Only non-mapped features can be retrieved uniquely.")
            return ordered_set(self.get_edge_features(feature_name))
        except KeyError:
            raise AttributeError(f"Edges do not have a feature called {feature_name}")

    def add_edge_feature(
        self,
        new_feature_name: str,
        new_feature_vals: Sequence[Any],
        map_from_feature: str = EDGE_PID,
    ) -> None:

        if new_feature_name in self.edge_attr:
            raise AttributeError("Features cannot be overridden once created")
        if map_from_feature in self._mapped_edge_features:
            raise AttributeError(f"{map_from_feature} is already a feature mapping.")

        feature_keys = self.get_unique_edge_features(map_from_feature)
        if len(feature_keys) != len(new_feature_vals):
            raise AttributeError(
                f"Expected encodings for {len(feature_keys)} unique features, but got {len(new_feature_vals)} encodings."
            )

        if map_from_feature == EDGE_PID:
            self.edge_attr[new_feature_name] = new_feature_vals
        else:
            self.edge_attr[new_feature_name] = MappedFeature(
                name=map_from_feature, values=new_feature_vals
            )
            self._mapped_edge_features.add(new_feature_name)

    def get_edge_features(
        self,
        feature_name: str = EDGE_PID,
        pids: Optional[Iterable[Hashable]] = None,
    ) -> Sequence[Any]:
        if feature_name in self._mapped_edge_features:
            values = self.edge_attr[feature_name].values
        else:
            values = self.edge_attr[feature_name]
        if isinstance(values, torch.Tensor):
            idxs = list(
                self.get_edge_features_iter(feature_name, pids, index_only=True)
            )
            return values[idxs]
        return list(self.get_edge_features_iter(feature_name, pids))

    def get_edge_features_iter(
        self,
        feature_name: str = EDGE_PID,
        pids: Optional[Iterable[TripletLike]] = None,
        index_only: bool = False,
    ) -> Iterator[Any]:
        if pids is None:
            pids = self.edge_attr[EDGE_PID]

        if feature_name in self._mapped_edge_features:
            feature_map_info = self.edge_attr[feature_name]
            from_feature_name, to_feature_vals = (
                feature_map_info.name,
                feature_map_info.values,
            )
            from_feature_vals = self.get_unique_edge_features(from_feature_name)
            feature_mapping = {k: i for i, k in enumerate(from_feature_vals)}

            for pid in pids:
                idx = self._edges[pid]
                from_feature_val = self.edge_attr[from_feature_name][idx]
                to_feature_idx = feature_mapping[from_feature_val]
                if index_only:
                    yield to_feature_idx
                else:
                    yield to_feature_vals[to_feature_idx]
        else:
            for pid in pids:
                idx = self._edges[pid]
                if index_only:
                    yield idx
                else:
                    yield self.edge_attr[feature_name][idx]

    def to_triplets(self) -> Iterator[TripletLike]:
        return iter(self.edge_attr[EDGE_PID])

    def save(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        with open(path + "/edges", "wb") as f:
            pkl.dump(self._edges, f)
        with open(path + "/nodes", "wb") as f:
            pkl.dump(self._nodes, f)

        with open(path + "/mapped_edges", "wb") as f:
            pkl.dump(self._mapped_edge_features, f)
        with open(path + "/mapped_nodes", "wb") as f:
            pkl.dump(self._mapped_node_features, f)

        node_attr_path = path + "/node_attr"
        os.makedirs(node_attr_path, exist_ok=True)
        for attr_name, vals in self.node_attr.items():
            torch.save(vals, node_attr_path + f"/{attr_name}.pt")

        edge_attr_path = path + "/edge_attr"
        os.makedirs(edge_attr_path, exist_ok=True)
        for attr_name, vals in self.edge_attr.items():
            torch.save(vals, edge_attr_path + f"/{attr_name}.pt")

    @classmethod
    def from_disk(cls, path: str) -> "LargeGraphIndexer":
        indexer = cls(list(), list())
        with open(path + "/edges", "rb") as f:
            indexer._edges = pkl.load(f)
        with open(path + "/nodes", "rb") as f:
            indexer._nodes = pkl.load(f)

        with open(path + "/mapped_edges", "rb") as f:
            indexer._mapped_edge_features = pkl.load(f)
        with open(path + "/mapped_nodes", "rb") as f:
            indexer._mapped_node_features = pkl.load(f)

        node_attr_path = path + "/node_attr"
        for fname in os.listdir(node_attr_path):
            full_fname = f"{node_attr_path}/{fname}"
            key = fname.split(".")[0]
            indexer.node_attr[key] = torch.load(full_fname)

        edge_attr_path = path + "/edge_attr"
        for fname in os.listdir(edge_attr_path):
            full_fname = f"{edge_attr_path}/{fname}"
            key = fname.split(".")[0]
            indexer.edge_attr[key] = torch.load(full_fname)

        return indexer

    def __eq__(self, value: "LargeGraphIndexer") -> bool:
        eq = True
        eq &= self._nodes == value._nodes
        eq &= self._edges == value._edges
        eq &= self.node_attr.keys() == value.node_attr.keys()
        eq &= self.edge_attr.keys() == value.edge_attr.keys()
        eq &= self._mapped_node_features == value._mapped_node_features
        eq &= self._mapped_edge_features == value._mapped_edge_features

        for k in self.node_attr:
            eq &= type(self.node_attr[k]) == type(value.node_attr[k])
            if isinstance(self.node_attr[k], torch.Tensor):
                eq &= torch.equal(self.node_attr[k], value.node_attr[k])
            else:
                eq &= self.node_attr[k] == value.node_attr[k]
        for k in self.edge_attr:
            eq &= type(self.edge_attr[k]) == type(value.edge_attr[k])
            if isinstance(self.edge_attr[k], torch.Tensor):
                eq &= torch.equal(self.edge_attr[k], value.edge_attr[k])
            else:
                eq &= self.edge_attr[k] == value.edge_attr[k]
        return eq

    def _to_data_attrs(
        self, node_feature_name: str, edge_feature_name: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.LongTensor, Optional[torch.Tensor]]:
        x = torch.Tensor(self.get_node_features(node_feature_name))
        edge_index = torch.t(torch.LongTensor(self.get_edge_features(EDGE_INDEX)))
        edge_attr = (
            self.get_edge_features(edge_feature_name)
            if edge_feature_name is not None
            else None
        )
        return x, edge_index, edge_attr

    def to_data(
        self, node_feature_name: str, edge_feature_name: Optional[str] = None
    ) -> Data:
        x, edge_index, edge_attr = self._to_data_attrs(
            node_feature_name, edge_feature_name
        )
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_features_for_triplets(
    indexer: LargeGraphIndexer,
    triplets: Iterable[TripletLike],
    node_feature_name: str = "x",
    edge_feature_name: str = "edge_attr",
    pre_transform: Optional[Callable[[TripletLike], TripletLike]] = None,
) -> Data:

    if pre_transform is not None:

        def apply_transform(trips):
            for trip in trips:
                yield pre_transform(tuple(trip))

        # TODO: Make this safe for large amounts of triplets?
        triplets = list(apply_transform(triplets))

    small_graph_indexer = LargeGraphIndexer.from_triplets(
        triplets, pre_transform=pre_transform
    )
    node_keys = small_graph_indexer.get_node_features()
    node_feats = indexer.get_node_features(
        feature_name=node_feature_name, pids=node_keys
    )

    edge_keys = small_graph_indexer.get_edge_features(pids=triplets)
    edge_feats = indexer.get_edge_features(
        feature_name=edge_feature_name, pids=edge_keys
    )

    edge_index = small_graph_indexer.get_edge_features(EDGE_INDEX, triplets)

    x = torch.Tensor(node_feats)
    edge_attr = torch.Tensor(edge_feats)
    edge_index = torch.t(torch.LongTensor(edge_index))
    data_obj = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(node_feats)
    )
    # needed for mappings
    data_obj[NODE_PID] = node_keys
    data_obj[EDGE_PID] = edge_keys
    data_obj["node_idx"] = [indexer._nodes[k] for k in node_keys]
    data_obj["edge_idx"] = [indexer._edges[e] for e in edge_keys]

    return data_obj
