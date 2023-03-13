import pytest

from torch_geometric.typing import EdgeTypeStr


def test_edge_type_str():
    edge_type_str = EdgeTypeStr('a__links__b')
    assert isinstance(edge_type_str, str)
    assert edge_type_str == 'a__links__b'
    assert edge_type_str.to_tuple() == ('a', 'links', 'b')

    edge_type_str = EdgeTypeStr('a', 'b')
    assert isinstance(edge_type_str, str)
    assert edge_type_str == 'a__to__b'
    assert edge_type_str.to_tuple() == ('a', 'to', 'b')

    edge_type_str = EdgeTypeStr(('a', 'b'))
    assert isinstance(edge_type_str, str)
    assert edge_type_str == 'a__to__b'
    assert edge_type_str.to_tuple() == ('a', 'to', 'b')

    edge_type_str = EdgeTypeStr('a', 'links', 'b')
    assert isinstance(edge_type_str, str)
    assert edge_type_str == 'a__links__b'
    assert edge_type_str.to_tuple() == ('a', 'links', 'b')

    edge_type_str = EdgeTypeStr(('a', 'links', 'b'))
    assert isinstance(edge_type_str, str)
    assert edge_type_str == 'a__links__b'
    assert edge_type_str.to_tuple() == ('a', 'links', 'b')

    with pytest.raises(ValueError, match="invalid edge type"):
        EdgeTypeStr('a', 'b', 'c', 'd')

    with pytest.raises(ValueError, match="Cannot convert the edge type"):
        EdgeTypeStr('a__b__c__d').to_tuple()
