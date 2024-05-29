from typing import (
    Tuple,
    Hashable,
    Dict,
    Iterable,
    Set,
    Optional,
    Any,
    Iterator,
    List,
    Sequence,
)
import pickle as pkl
from dataclasses import dataclass
from itertools import chain

# Is there a multiprocessing-friendly implementation of this?

TripletLike = Tuple[Hashable, Hashable, Hashable]


def ordered_set(values: Iterable[Hashable]) -> List[Hashable]:
    return list(dict.fromkeys(values))


NODE_PID = "pid"

EDGE_PID = "e_pid"
EDGE_HEAD = "h"
EDGE_RELATION = "r"
EDGE_TAIL = "t"


@dataclass
class MappedFeature:
    name: str
    values: Sequence[Any]


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
        else:
            self.node_attr = dict()
            self.node_attr[NODE_PID] = nodes

        for i, node in enumerate(self.node_attr[NODE_PID]):
            self._nodes[node] = i

        if edge_attr is not None:
            # TODO: Validity checks btw edges and edge_attr
            self.edge_attr = edge_attr
            for i, tup in enumerate(edges):
                self._edges[tup] = i
        else:
            self.edge_attr = dict()
            for default_key in [EDGE_HEAD, EDGE_RELATION, EDGE_TAIL, EDGE_PID]:
                self.edge_attr[default_key] = list()

            for i, tup in enumerate(edges):
                h, r, t = tup
                self.edge_attr[EDGE_HEAD].append(h)
                self.edge_attr[EDGE_RELATION].append(r)
                self.edge_attr[EDGE_TAIL].append(t)
                self.edge_attr[EDGE_PID].append(tup)
                self._edges[tup] = i

    @classmethod
    def from_triplets(cls, triplets: Iterable[TripletLike]) -> "LargeGraphIndexer":
        # NOTE: Right now assumes that all trips can be loaded into memory
        nodes = set()
        edges = set()

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
        return list(self.get_node_features_iter(feature_name, pids))

    def get_node_features_iter(
        self, feature_name: str = NODE_PID, pids: Optional[Iterable[Hashable]] = None
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
                yield to_feature_vals[to_feature_idx]
        else:
            for pid in pids:
                idx = self._nodes[pid]
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
        feature_name: str = EDGE_RELATION,
        pids: Optional[Iterable[Hashable]] = None,
    ) -> List[Any]:
        return list(self.get_edge_features_iter(feature_name, pids))

    def get_edge_features_iter(
        self,
        feature_name: str = EDGE_RELATION,
        pids: Optional[Iterable[TripletLike]] = None,
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
                yield to_feature_vals[to_feature_idx]
        else:
            for pid in pids:
                idx = self._edges[pid]
                yield self.edge_attr[feature_name][idx]

    def to_triplets(self) -> Iterator[TripletLike]:
        return iter(self.edge_attr[EDGE_PID])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            pkl.dump(self, f)

    @classmethod
    def from_disk(cls, path: str) -> "LargeGraphIndexer":
        with open(path) as f:
            return pkl.load(f)
