# This file cannot be named `test_utils` as there exists already such file
# in `test/profile/test_utils.py` which would yield a name conflict.

from torch_geometric.loader.utils import edge_type_to_str


def test_edge_type_to_str_for_tuple():
    edge_type = ("SrcNode", "EdgeType", "DstNode")

    assert edge_type_to_str(edge_type) == "SrcNode__EdgeType__DstNode"


def test_edge_type_to_str_for_str():
    edge_type = "SrcNode__EdgeType__DstNode"

    assert edge_type_to_str(edge_type) == edge_type
