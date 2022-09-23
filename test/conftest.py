import functools
import os.path as osp
import shutil

import pytest

from torch_geometric.data import Dataset


def load_dataset(root: str, name: str, *args, **kwargs) -> Dataset:
    r"""Returns a variety of datasets according to :obj:`name`."""
    if 'karate' in name.lower():
        from torch_geometric.datasets import KarateClub
        return KarateClub(*args, **kwargs)
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        from torch_geometric.datasets import Planetoid
        path = osp.join(root, 'Planetoid', name)
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
    if name in ['citationCiteseer', 'illc1850']:
        from torch_geometric.datasets import SuiteSparseMatrixCollection
        path = osp.join(root, 'SuiteSparseMatrixCollection')
        return SuiteSparseMatrixCollection(path, name=name, *args, **kwargs)
    if 'elliptic' in name.lower():
        from torch_geometric.datasets import EllipticBitcoinDataset
        path = osp.join(root, 'EllipticBitcoinDataset')
        return EllipticBitcoinDataset(path, *args, **kwargs)

    raise NotImplementedError


@pytest.fixture(scope='session')
def get_dataset():
    root = osp.join('/', 'tmp', 'pyg_test_datasets')
    yield functools.partial(load_dataset, root)
    # if osp.exists(root):
    #     shutil.rmtree(root)
