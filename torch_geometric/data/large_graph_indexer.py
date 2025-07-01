import os
import pickle as pkl
import shutil
from dataclasses import dataclass
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.io import fs
from torch_geometric.typing import WITH_PT24

# Could be any hashable type
TripletLike = Tuple[str, str, str]

KnowledgeGraphLike = Iterable[TripletLike]


def ordered_set(values: Iterable[str]) -> List[str]:
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

FeatureValueType = Union[Sequence[Any], Tensor]


@dataclass
class MappedFeature:
    name: str
    values: FeatureValueType

    def __eq__(self, value: "MappedFeature") -> bool:
        eq = self.name == value.name
        if isinstance(self.values, torch.Tensor):
            eq &= torch.equal(self.values, value.values)
        else:
            eq &= self.values == value.values
        return eq


if WITH_PT24:
    torch.serialization.add_safe_globals([MappedFeature])


class LargeGraphIndexer:
    """For a dataset that consists of multiple subgraphs that are assumed to
    be part of a much larger graph, collate the values into a large graph store
    to save resources.
    """
    def __init__(
        self,
        nodes: Iterable[str],
        edges: KnowledgeGraphLike,
        node_attr: Optional[Dict[str, List[Any]]] = None,
        edge_attr: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        r"""Constructs a new index that uniquely catalogs each node and edge
        by id. Not meant to be used directly.

        Args:
            nodes (Iterable[str]): Node ids in the graph.
            edges (KnowledgeGraphLike): Edge ids in the graph.
            node_attr (Optional[Dict[str, List[Any]]], optional): Mapping node
                attribute name and list of their values in order of unique node
                ids. Defaults to None.
            edge_attr (Optional[Dict[str, List[Any]]], optional): Mapping edge
                attribute name and list of their values in order of unique edge
                ids. Defaults to None.
        """
        self._nodes: Dict[str, int] = dict()
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
                    "Invalid node_attr object. Missing " +
                    f"{NODE_KEYS - set(self.node_attr.keys())}")
            elif self.node_attr[NODE_PID] != nodes:
                raise AttributeError(
                    "Nodes provided do not match those in node_attr")
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
                    "Invalid edge_attr object. Missing " +
                    f"{EDGE_KEYS - set(self.edge_attr.keys())}")
            elif self.node_attr[EDGE_PID] != edges:
                raise AttributeError(
                    "Edges provided do not match those in edge_attr")

        else:
            self.edge_attr = dict()
            for default_key in EDGE_KEYS:
                self.edge_attr[default_key] = list()
            self.edge_attr[EDGE_PID] = edges

            for tup in edges:
                h, r, t = tup
                self.edge_attr[EDGE_HEAD].append(h)
                self.edge_attr[EDGE_RELATION].append(r)
                self.edge_attr[EDGE_TAIL].append(t)
                self.edge_attr[EDGE_INDEX].append(
                    (self._nodes[h], self._nodes[t]))

        for i, tup in enumerate(edges):
            self._edges[tup] = i

    @classmethod
    def from_triplets(
        cls,
        triplets: KnowledgeGraphLike,
        pre_transform: Optional[Callable[[TripletLike], TripletLike]] = None,
    ) -> "LargeGraphIndexer":
        r"""Generate a new index from a series of triplets that represent edge
        relations between nodes.
        Formatted like (source_node, edge, dest_node).

        Args:
            triplets (KnowledgeGraphLike): Series of triplets representing
                knowledge graph relations.
            pre_transform (Optional[Callable[[TripletLike], TripletLike]]):
                Optional preprocessing function to apply to triplets.
                Defaults to None.

        Returns:
            LargeGraphIndexer: Index of unique nodes and edges.
        """
        # NOTE: Right now assumes that all trips can be loaded into memory
        nodes = set()
        edges = set()

        if pre_transform is not None:

            def apply_transform(
                    trips: KnowledgeGraphLike) -> Iterator[TripletLike]:
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
    def collate(cls,
                graphs: Iterable["LargeGraphIndexer"]) -> "LargeGraphIndexer":
        r"""Combines a series of large graph indexes into a single large graph
        index.

        Args:
            graphs (Iterable[LargeGraphIndexer]): Indices to be
                combined.

        Returns:
            LargeGraphIndexer: Singular unique index for all nodes and edges
                in input indices.
        """
        # FIXME Needs to merge node attrs and edge attrs?
        trips = chain.from_iterable([graph.to_triplets() for graph in graphs])
        return cls.from_triplets(trips)

    def get_unique_node_features(self,
                                 feature_name: str = NODE_PID) -> List[str]:
        r"""Get all the unique values for a specific node attribute.

        Args:
            feature_name (str, optional): Name of feature to get.
                Defaults to NODE_PID.

        Returns:
            List[str]: List of unique values for the specified feature.
        """
        try:
            if feature_name in self._mapped_node_features:
                raise IndexError(
                    "Only non-mapped features can be retrieved uniquely.")
            return ordered_set(self.get_node_features(feature_name))

        except KeyError as e:
            raise AttributeError(
                f"Nodes do not have a feature called {feature_name}") from e

    def add_node_feature(
        self,
        new_feature_name: str,
        new_feature_vals: FeatureValueType,
        map_from_feature: str = NODE_PID,
    ) -> None:
        r"""Adds a new feature that corresponds to each unique node in
            the graph.

        Args:
            new_feature_name (str): Name to call the new feature.
            new_feature_vals (FeatureValueType): Values to map for that
                new feature.
            map_from_feature (str, optional): Key of feature to map from.
                Size must match the number of feature values.
                Defaults to NODE_PID.
        """
        if new_feature_name in self.node_attr:
            raise AttributeError("Features cannot be overridden once created")
        if map_from_feature in self._mapped_node_features:
            raise AttributeError(
                f"{map_from_feature} is already a feature mapping.")

        feature_keys = self.get_unique_node_features(map_from_feature)
        if len(feature_keys) != len(new_feature_vals):
            raise AttributeError(
                "Expected encodings for {len(feature_keys)} unique features," +
                f" but got {len(new_feature_vals)} encodings.")

        if map_from_feature == NODE_PID:
            self.node_attr[new_feature_name] = new_feature_vals
        else:
            self.node_attr[new_feature_name] = MappedFeature(
                name=map_from_feature, values=new_feature_vals)
            self._mapped_node_features.add(new_feature_name)

    def get_node_features(
        self,
        feature_name: str = NODE_PID,
        pids: Optional[Iterable[str]] = None,
    ) -> List[Any]:
        r"""Get node feature values for a given set of unique node ids.
            Returned values are not necessarily unique.

        Args:
            feature_name (str, optional): Name of feature to fetch. Defaults
                to NODE_PID.
            pids (Optional[Iterable[str]], optional): Node ids to fetch
                for. Defaults to None, which fetches all nodes.

        Returns:
            List[Any]: Node features corresponding to the specified ids.
        """
        if feature_name in self._mapped_node_features:
            values = self.node_attr[feature_name].values
        else:
            values = self.node_attr[feature_name]

        # TODO: torch_geometric.utils.select
        if isinstance(values, torch.Tensor):
            idxs = list(
                self.get_node_features_iter(feature_name, pids,
                                            index_only=True))
            return values[idxs]
        return list(self.get_node_features_iter(feature_name, pids))

    def get_node_features_iter(
        self,
        feature_name: str = NODE_PID,
        pids: Optional[Iterable[str]] = None,
        index_only: bool = False,
    ) -> Iterator[Any]:
        """Iterator version of get_node_features. If index_only is True,
        yields indices instead of values.
        """
        if pids is None:
            pids = self.node_attr[NODE_PID]

        if feature_name in self._mapped_node_features:
            feature_map_info = self.node_attr[feature_name]
            from_feature_name, to_feature_vals = (
                feature_map_info.name,
                feature_map_info.values,
            )
            from_feature_vals = self.get_unique_node_features(
                from_feature_name)
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

    def get_unique_edge_features(self,
                                 feature_name: str = EDGE_PID) -> List[str]:
        r"""Get all the unique values for a specific edge attribute.

        Args:
            feature_name (str, optional): Name of feature to get.
                Defaults to EDGE_PID.

        Returns:
            List[str]: List of unique values for the specified feature.
        """
        try:
            if feature_name in self._mapped_edge_features:
                raise IndexError(
                    "Only non-mapped features can be retrieved uniquely.")
            return ordered_set(self.get_edge_features(feature_name))
        except KeyError as e:
            raise AttributeError(
                f"Edges do not have a feature called {feature_name}") from e

    def add_edge_feature(
        self,
        new_feature_name: str,
        new_feature_vals: FeatureValueType,
        map_from_feature: str = EDGE_PID,
    ) -> None:
        r"""Adds a new feature that corresponds to each unique edge in
        the graph.

        Args:
            new_feature_name (str): Name to call the new feature.
            new_feature_vals (FeatureValueType): Values to map for that new
                feature.
            map_from_feature (str, optional): Key of feature to map from.
                Size must match the number of feature values.
                Defaults to EDGE_PID.
        """
        if new_feature_name in self.edge_attr:
            raise AttributeError("Features cannot be overridden once created")
        if map_from_feature in self._mapped_edge_features:
            raise AttributeError(
                f"{map_from_feature} is already a feature mapping.")

        feature_keys = self.get_unique_edge_features(map_from_feature)
        if len(feature_keys) != len(new_feature_vals):
            raise AttributeError(
                f"Expected encodings for {len(feature_keys)} unique features, "
                + f"but got {len(new_feature_vals)} encodings.")

        if map_from_feature == EDGE_PID:
            self.edge_attr[new_feature_name] = new_feature_vals
        else:
            self.edge_attr[new_feature_name] = MappedFeature(
                name=map_from_feature, values=new_feature_vals)
            self._mapped_edge_features.add(new_feature_name)

    def get_edge_features(
        self,
        feature_name: str = EDGE_PID,
        pids: Optional[Iterable[str]] = None,
    ) -> List[Any]:
        r"""Get edge feature values for a given set of unique edge ids.
            Returned values are not necessarily unique.

        Args:
            feature_name (str, optional): Name of feature to fetch.
                Defaults to EDGE_PID.
            pids (Optional[Iterable[str]], optional): Edge ids to fetch
                for. Defaults to None, which fetches all edges.

        Returns:
            List[Any]: Node features corresponding to the specified ids.
        """
        if feature_name in self._mapped_edge_features:
            values = self.edge_attr[feature_name].values
        else:
            values = self.edge_attr[feature_name]

        # TODO: torch_geometric.utils.select
        if isinstance(values, torch.Tensor):
            idxs = list(
                self.get_edge_features_iter(feature_name, pids,
                                            index_only=True))
            return values[idxs]
        return list(self.get_edge_features_iter(feature_name, pids))

    def get_edge_features_iter(
        self,
        feature_name: str = EDGE_PID,
        pids: Optional[KnowledgeGraphLike] = None,
        index_only: bool = False,
    ) -> Iterator[Any]:
        """Iterator version of get_edge_features. If index_only is True,
        yields indices instead of values.
        """
        if pids is None:
            pids = self.edge_attr[EDGE_PID]

        if feature_name in self._mapped_edge_features:
            feature_map_info = self.edge_attr[feature_name]
            from_feature_name, to_feature_vals = (
                feature_map_info.name,
                feature_map_info.values,
            )
            from_feature_vals = self.get_unique_edge_features(
                from_feature_name)
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
            indexer.node_attr[key] = fs.torch_load(full_fname)

        edge_attr_path = path + "/edge_attr"
        for fname in os.listdir(edge_attr_path):
            full_fname = f"{edge_attr_path}/{fname}"
            key = fname.split(".")[0]
            indexer.edge_attr[key] = fs.torch_load(full_fname)

        return indexer

    def to_data(self, node_feature_name: str,
                edge_feature_name: Optional[str] = None) -> Data:
        """Return a Data object containing all the specified node and
            edge features and the graph.

        Args:
            node_feature_name (str): Feature to use for nodes
            edge_feature_name (Optional[str], optional): Feature to use for
                edges. Defaults to None.

        Returns:
            Data: Data object containing the specified node and
                edge features and the graph.
        """
        x = torch.Tensor(self.get_node_features(node_feature_name))
        node_id = torch.LongTensor(range(len(x)))

        edge_index = torch.t(
            torch.LongTensor(self.get_edge_features(EDGE_INDEX)))

        edge_attr = (self.get_edge_features(edge_feature_name)
                     if edge_feature_name is not None else None)
        edge_id = torch.LongTensor(range(len(edge_attr)))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    edge_id=edge_id, node_id=node_id)

    def __eq__(self, value: "LargeGraphIndexer") -> bool:
        eq = True
        eq &= self._nodes == value._nodes
        eq &= self._edges == value._edges
        eq &= self.node_attr.keys() == value.node_attr.keys()
        eq &= self.edge_attr.keys() == value.edge_attr.keys()
        eq &= self._mapped_node_features == value._mapped_node_features
        eq &= self._mapped_edge_features == value._mapped_edge_features

        for k in self.node_attr:
            eq &= isinstance(self.node_attr[k], type(value.node_attr[k]))
            if isinstance(self.node_attr[k], torch.Tensor):
                eq &= torch.equal(self.node_attr[k], value.node_attr[k])
            else:
                eq &= self.node_attr[k] == value.node_attr[k]
        for k in self.edge_attr:
            eq &= isinstance(self.edge_attr[k], type(value.edge_attr[k]))
            if isinstance(self.edge_attr[k], torch.Tensor):
                eq &= torch.equal(self.edge_attr[k], value.edge_attr[k])
            else:
                eq &= self.edge_attr[k] == value.edge_attr[k]
        return eq


def get_features_for_triplets_groups(
    indexer: LargeGraphIndexer,
    triplet_groups: Iterable[KnowledgeGraphLike],
    node_feature_name: str = "x",
    edge_feature_name: str = "edge_attr",
    pre_transform: Optional[Callable[[TripletLike], TripletLike]] = None,
    verbose: bool = False,
) -> Iterator[Data]:
    """Given an indexer and a series of triplet groups (like a dataset),
    retrieve the specified node and edge features for each triplet from the
    index.

    Args:
        indexer (LargeGraphIndexer): Indexer containing desired features
        triplet_groups (Iterable[KnowledgeGraphLike]): List of lists of
            triplets to fetch features for
        node_feature_name (str, optional): Node feature to fetch.
            Defaults to "x".
        edge_feature_name (str, optional): edge feature to fetch.
            Defaults to "edge_attr".
        pre_transform (Optional[Callable[[TripletLike], TripletLike]]):
            Optional preprocessing to perform on triplets.
            Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Yields:
        Iterator[Data]: For each triplet group, yield a data object containing
            the unique graph and features from the index.
    """
    if pre_transform is not None:

        def apply_transform(trips):
            for trip in trips:
                yield pre_transform(tuple(trip))

        # TODO: Make this safe for large amounts of triplets?
        triplet_groups = (list(apply_transform(triplets))
                          for triplets in triplet_groups)

    node_keys = []
    edge_keys = []
    edge_index = []

    for triplets in tqdm(triplet_groups, disable=not verbose):
        small_graph_indexer = LargeGraphIndexer.from_triplets(
            triplets, pre_transform=pre_transform)

        node_keys.append(small_graph_indexer.get_node_features())
        edge_keys.append(small_graph_indexer.get_edge_features(pids=triplets))
        edge_index.append(
            small_graph_indexer.get_edge_features(EDGE_INDEX, triplets))

    node_feats = indexer.get_node_features(feature_name=node_feature_name,
                                           pids=chain.from_iterable(node_keys))
    edge_feats = indexer.get_edge_features(feature_name=edge_feature_name,
                                           pids=chain.from_iterable(edge_keys))

    last_node_idx, last_edge_idx = 0, 0
    for (nkeys, ekeys, eidx) in zip(node_keys, edge_keys, edge_index):
        nlen, elen = len(nkeys), len(ekeys)
        x = torch.Tensor(node_feats[last_node_idx:last_node_idx + nlen])
        last_node_idx += len(nkeys)

        edge_attr = torch.Tensor(edge_feats[last_edge_idx:last_edge_idx +
                                            elen])
        last_edge_idx += len(ekeys)

        edge_idx = torch.LongTensor(eidx).T

        data_obj = Data(x=x, edge_attr=edge_attr, edge_index=edge_idx)
        data_obj[NODE_PID] = node_keys
        data_obj[EDGE_PID] = edge_keys
        data_obj["node_idx"] = [indexer._nodes[k] for k in nkeys]
        data_obj["edge_idx"] = [indexer._edges[e] for e in ekeys]

        yield data_obj


def get_features_for_triplets(
    indexer: LargeGraphIndexer,
    triplets: KnowledgeGraphLike,
    node_feature_name: str = "x",
    edge_feature_name: str = "edge_attr",
    pre_transform: Optional[Callable[[TripletLike], TripletLike]] = None,
    verbose: bool = False,
) -> Data:
    """For a given set of triplets retrieve a Data object containing the
        unique graph and features from the index.

    Args:
        indexer (LargeGraphIndexer): Indexer containing desired features
        triplets (KnowledgeGraphLike): Triplets to fetch features for
        node_feature_name (str, optional): Feature to use for node features.
            Defaults to "x".
        edge_feature_name (str, optional): Feature to use for edge features.
            Defaults to "edge_attr".
        pre_transform (Optional[Callable[[TripletLike], TripletLike]]):
            Optional preprocessing function for triplets. Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        Data: Data object containing the unique graph and features from the
            index for the given triplets.
    """
    gen = get_features_for_triplets_groups(indexer, [triplets],
                                           node_feature_name,
                                           edge_feature_name, pre_transform,
                                           verbose)
    return next(gen)
