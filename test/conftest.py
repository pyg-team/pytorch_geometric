import functools
import logging
import os.path as osp
from typing import Callable

import pytest
import torch

import torch_geometric.typing
from torch_geometric.data import Dataset
from torch_geometric.io import fs


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
    if name in ['citationCiteseer', 'illc1850']:
        from torch_geometric.datasets import SuiteSparseMatrixCollection
        path = osp.join(root, 'SuiteSparseMatrixCollection')
        return SuiteSparseMatrixCollection(path, name=name, *args, **kwargs)
    if 'elliptic' in name.lower():
        from torch_geometric.datasets import EllipticBitcoinDataset
        path = osp.join(root, 'EllipticBitcoinDataset')
        return EllipticBitcoinDataset(path, *args, **kwargs)
    if name.lower() in ['hetero']:
        from torch_geometric.testing import FakeHeteroDataset
        return FakeHeteroDataset(*args, **kwargs)

    raise ValueError(f"Cannot load dataset with name '{name}'")


@pytest.fixture(scope='session')
def get_dataset() -> Callable:
    # TODO Support memory filesystem on Windows.
    if torch_geometric.typing.WITH_WINDOWS:
        root = osp.join('/', 'tmp', 'pyg_test_datasets')
    else:
        root = 'memory://pyg_test_datasets'

    yield functools.partial(load_dataset, root)

    fs.rm(root)


@pytest.fixture
def enable_extensions():  # Nothing to do.
    yield


@pytest.fixture
def disable_extensions():
    def is_setting(name: str) -> bool:
        if not name.startswith('WITH_'):
            return False
        if name.startswith('WITH_PT') or name.startswith('WITH_WINDOWS'):
            return False
        return True

    settings = dir(torch_geometric.typing)
    settings = [key for key in settings if is_setting(key)]
    state = {key: getattr(torch_geometric.typing, key) for key in settings}

    for key in state.keys():
        setattr(torch_geometric.typing, key, False)
    yield
    for key, value in state.items():
        setattr(torch_geometric.typing, key, value)


@pytest.fixture
def without_extensions(request):
    request.getfixturevalue(request.param)
    return request.param == 'disable_extensions'


@pytest.fixture(scope='function')
def spawn_context():
    torch.multiprocessing.set_start_method('spawn', force=True)
    logging.info("Setting torch.multiprocessing context to 'spawn'")
