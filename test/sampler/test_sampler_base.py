import pytest

from torch_geometric.sampler.base import NumNeighbors


def test_homogeneous_num_neighbors():
    with pytest.raises(ValueError, match="'default' must be set to 'None'"):
        num_neighbors = NumNeighbors([25, 10], default=[-1, -1])

    num_neighbors = NumNeighbors([25, 10])
    assert str(num_neighbors) == 'NumNeighbors(values=[25, 10], default=None)'

    assert num_neighbors.get_values() == [25, 10]
    assert num_neighbors.__dict__['_values'] == [25, 10]
    assert num_neighbors.get_values() == [25, 10]  # Test caching.

    assert num_neighbors.get_mapped_values() == [25, 10]
    assert num_neighbors.__dict__['_mapped_values'] == [25, 10]
    assert num_neighbors.get_mapped_values() == [25, 10]  # Test caching.

    assert num_neighbors.num_hops == 2
    assert num_neighbors.__dict__['_num_hops'] == 2
    assert num_neighbors.num_hops == 2  # Test caching.


def test_heterogeneous_num_neighbors_list():
    num_neighbors = NumNeighbors([25, 10])

    values = num_neighbors.get_values([('A', 'B'), ('B', 'A')])
    assert values == {('A', 'B'): [25, 10], ('B', 'A'): [25, 10]}

    values = num_neighbors.get_mapped_values([('A', 'B'), ('B', 'A')])
    assert values == {'A__B': [25, 10], 'B__A': [25, 10]}

    assert num_neighbors.num_hops == 2


def test_heterogeneous_num_neighbors_dict_and_default():
    num_neighbors = NumNeighbors({('A', 'B'): [25, 10]}, default=[-1])
    with pytest.raises(ValueError, match="hops must be the same across all"):
        values = num_neighbors.get_values([('A', 'B'), ('B', 'A')])

    num_neighbors = NumNeighbors({('A', 'B'): [25, 10]}, default=[-1, -1])

    values = num_neighbors.get_values([('A', 'B'), ('B', 'A')])
    assert values == {('A', 'B'): [25, 10], ('B', 'A'): [-1, -1]}

    values = num_neighbors.get_mapped_values([('A', 'B'), ('B', 'A')])
    assert values == {'A__B': [25, 10], 'B__A': [-1, -1]}

    assert num_neighbors.num_hops == 2


def test_num_neighbors_cast():
    num_neighbors = NumNeighbors.cast([25, 10])
    assert isinstance(num_neighbors, NumNeighbors)
    assert num_neighbors.values == [25, 10]
    assert num_neighbors.default is None

    num_neighbors = NumNeighbors.cast({('A', 'B'): [25, 10]}, [-1, -1])
    assert isinstance(num_neighbors, NumNeighbors)
    assert num_neighbors.values == {('A', 'B'): [25, 10]}
    assert num_neighbors.default == [-1, -1]
