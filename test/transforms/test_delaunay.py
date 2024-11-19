import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import Delaunay


def assert_one_point(transform: Delaunay) -> None:
    data = Data(pos=torch.rand(1, 2))
    data = transform(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[], []]


def assert_two_points(transform: Delaunay) -> None:
    data = Data(pos=torch.rand(2, 2))
    data = transform(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]


def assert_three_points(transform: Delaunay) -> None:
    data = Data(pos=torch.rand(3, 2))
    data = transform(data)
    assert len(data) == 2
    assert data.face.tolist() == [[0], [1], [2]]


def assert_four_points(transform: Delaunay) -> None:
    pos = torch.tensor([
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.5, -0.5],
    ])
    data = Data(pos=pos)
    data = transform(data)
    assert len(data) == 2

    # The order of the simplices does not matter, therefore assert sets.
    faces = set(map(tuple, data.face.tolist()))
    assert faces == {(3, 1), (1, 3), (0, 2)}


@withPackage('scipy')
def test_qhull_delaunay() -> None:
    transform = Delaunay()

    assert str(transform) == 'Delaunay()'
    assert_one_point(transform)
    assert_two_points(transform)
    assert_three_points(transform)
    assert_four_points(transform)


@withPackage('torch_delaunay')
def test_shull_delaunay() -> None:
    transform = Delaunay()

    assert str(transform) == 'Delaunay()'
    assert_one_point(transform)
    assert_two_points(transform)
    assert_three_points(transform)
    assert_four_points(transform)
