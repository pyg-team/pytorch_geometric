import os
import os.path as osp
from typing import Callable

from torch_geometric.data import Dataset


def is_full_test() -> bool:
    r"""Whether to run the full but time-consuming test suite."""
    return os.getenv('FULL_TEST', '0') == '1'


def onlyFullTest(func: Callable) -> Callable:
    r"""A decorator to specify that this function belongs to the full test
    suite."""

    import pytest
    return pytest.mark.skipif(not is_full_test(), reason="Fast test run")(func)


def get_dataset(root: str, name: str, *args, **kwargs) -> Dataset:
    if name.lower() in ['karate']:
        from torch_geometric.datasets import KarateClub
        return KarateClub(*args, **kwargs)
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        from torch_geometric.datasets import Planetoid
        path = osp.join(root, 'Planetoid')
        return Planetoid(path, name, *args, **kwargs)
    if name in ['BZR', 'ENZYMES', 'IMDB-BINARY', 'MUTAG']:
        from torch_geometric.datasets import TUDataset
        path = osp.join(root, 'TUDataset')
        return TUDataset(path, name, *args, **kwargs)
    if name in ['ego-facebook', 'soc-Slashdot0811', 'wiki-vote']:
        from torch_geometric.datasets import SNAPDataset
        path = osp.join(root, 'SNAPDataset')
        return SNAPDataset(path, name, *args, **kwargs)
    if name.lower() in ['bashapes']:
        from torch_geometric.datasets import BAShapes
        return BAShapes(*args, **kwargs)
    if name.lower() in ['dblp']:
        from torch_geometric.datasets import DBLP
        path = osp.join(root, 'DBLP')
        return DBLP(path, *args, **kwargs)
    if name in ['DIMACS10/citationCiteseer', 'HB/illc1850']:
        from torch_geometric.datasets import SuiteSparseMatrixCollection
        path = osp.join(root, 'SuiteSparseMatrixCollection')
        g, n = name.split('/')
        return SuiteSparseMatrixCollection(path, g, n, *args, **kwargs)

    raise NotImplementedError


def withDataset(name: str, *args, **kwargs) -> Callable:
    r"""A decorator to load (and re-use) a variety of datasets during
    testing."""
    root = osp.join('/', 'tmp', 'pyg_test_datasets', '2')

    def decorator(func):
        import pytest

        def wrapper():
            return get_dataset(root, name, *args, **kwargs)

        return pytest.mark.parametrize('get_dataset', [wrapper])(func)

    return decorator
