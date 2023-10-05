import functools
import os.path as osp
import shutil
from typing import Callable

import pytest
import torch

import torch_geometric.typing
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
    root = osp.join('/', 'tmp', 'pyg_test_datasets')
    yield functools.partial(load_dataset, root)
    if osp.exists(root):
        shutil.rmtree(root)


@pytest.fixture()
def get_tensor_frame() -> Callable:
    import torch_frame

    def _get_tensor_frame(num_rows: int) -> torch_frame.TensorFrame:
        feat_dict = {
            torch_frame.categorical: torch.randint(0, 3, size=(num_rows, 3)),
            torch_frame.numerical: torch.randn(size=(num_rows, 2)),
        }
        col_names_dict = {
            torch_frame.categorical: ['a', 'b', 'c'],
            torch_frame.numerical: ['x', 'y'],
        }
        y = torch.randn(num_rows)

        return torch_frame.TensorFrame(
            feat_dict=feat_dict,
            col_names_dict=col_names_dict,
            y=y,
        )

    return _get_tensor_frame


@pytest.fixture
def disable_extensions():
    prev_state = {
        'WITH_PYG_LIB': torch_geometric.typing.WITH_PYG_LIB,
        'WITH_SAMPLED_OP': torch_geometric.typing.WITH_SAMPLED_OP,
        'WITH_INDEX_SORT': torch_geometric.typing.WITH_INDEX_SORT,
        'WITH_TORCH_SCATTER': torch_geometric.typing.WITH_TORCH_SCATTER,
        'WITH_TORCH_SPARSE': torch_geometric.typing.WITH_TORCH_SPARSE,
    }
    for key in prev_state.keys():
        setattr(torch_geometric.typing, key, False)
    yield
    for key, value in prev_state.items():
        setattr(torch_geometric.typing, key, value)
