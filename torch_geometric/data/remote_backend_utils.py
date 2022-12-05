# This file defines a set of utilities for remote backends (backends that are
# characterize as Tuple[FeatureStore, GraphStore]). TODO support for
# non-heterogeneous graphs (feature stores with a group_name=None).
from typing import Tuple, Union

from torch_geometric.data import FeatureStore, GraphStore
from torch_geometric.typing import EdgeType, NodeType


# NOTE PyG also supports querying by a relation type `rel` in an edge type
# (src, rel, dst). It may be worth supporting this in remote backends as well.
def _internal_num_nodes(
    feature_store: FeatureStore,
    graph_store: GraphStore,
    query: Union[NodeType, EdgeType],
) -> Union[int, Tuple[int, int]]:
    r"""Returns the number of nodes in the node type or the number of source
    and destination nodes in an edge type by sequentially accessing attributes
    in the feature and graph stores that reveal this number."""
    def _matches_node_type(query: Union[NodeType, EdgeType],
                           node_type: NodeType) -> bool:
        if isinstance(query, (list, tuple)):  # EdgeType
            return query[0] == node_type or query[-1] == node_type
        else:
            return query == node_type

    node_query = isinstance(query, NodeType)

    # TODO: In general, a feature store and graph store should be able to
    # expose methods that allow for easy access to individual attributes,
    # instead of requiring iteration to identify a particular attribute.
    # Implementing this should reduce the iteration below.

    # 1. Check the edges in the GraphStore, for each node type in each edge:
    edge_attrs = graph_store.get_all_edge_attrs()
    _num_nodes = [None] if node_query else [None, None]
    for edge_attr in edge_attrs:
        if edge_attr.size is None:
            continue
        if _matches_node_type(query, edge_attr.edge_type[0]):
            _num_nodes[0] = _num_nodes[0] or edge_attr.size[0]
        if _matches_node_type(query, edge_attr.edge_type[-1]):
            _num_nodes[-1] = _num_nodes[-1] or edge_attr.size[-1]
        if None not in _num_nodes:
            # We have filled out all the nodes:
            return _num_nodes[0] if node_query else tuple(_num_nodes)

    # 2. Check the node types stored in the FeatureStore:
    tensor_attrs = feature_store.get_all_tensor_attrs()
    matching_attrs = [
        attr for attr in tensor_attrs
        if _matches_node_type(query, attr.group_name)
    ]
    if node_query:
        if len(matching_attrs) > 0:
            return feature_store.get_tensor_size(matching_attrs[0])[0]
    else:
        matching_src_attrs = [
            attr for attr in matching_attrs if attr.group_name == query[0]
        ]
        matching_dst_attrs = [
            attr for attr in matching_attrs if attr.group_name == query[-1]
        ]
        if len(matching_src_attrs) > 0 and len(matching_dst_attrs) > 0:
            return (feature_store.get_tensor_size(matching_src_attrs[0])[0],
                    feature_store.get_tensor_size(matching_dst_attrs[0])[0])

    raise ValueError(
        f"Unable to accurately infer the number of nodes corresponding to "
        f"query {query} from feature store {feature_store} and graph store "
        f"{graph_store}. Please consider either adding an edge containing "
        f"the nodes in this query or feature tensors for the nodes in this "
        f"query.")


def num_nodes(
    feature_store: FeatureStore,
    graph_store: GraphStore,
    query: NodeType,
) -> int:
    r"""Returns the number of nodes in a given node type stored in a remote
    backend."""
    return _internal_num_nodes(feature_store, graph_store, query)


def size(
    feature_store: FeatureStore,
    graph_store: GraphStore,
    query: EdgeType,
) -> Tuple[int, int]:
    r"""Returns the size of an edge (number of source nodes, number of
    destination nodes) in an edge stored in a remote backend."""
    return _internal_num_nodes(feature_store, graph_store, query)
