import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import Data, OnDiskDataset, download_url, extract_zip
from torch_geometric.data.data import BaseData
from torch_geometric.io import fs
from torch_geometric.utils import from_smiles as _from_smiles


class PCQM4Mv2(OnDiskDataset):
    r"""The PCQM4Mv2 dataset from the `"OGB-LSC: A Large-Scale Challenge for
    Machine Learning on Graphs" <https://arxiv.org/abs/2103.09430>`_ paper.
    :class:`PCQM4Mv2` is a quantum chemistry dataset originally curated under
    the `PubChemQC project
    <https://pubs.acs.org/doi/10.1021/acs.jcim.7b00083>`_.
    The task is to predict the DFT-calculated HOMO-LUMO energy gap of molecules
    given their 2D molecular graphs.

    .. note::
        This dataset uses the :class:`OnDiskDataset` base class to load data
        dynamically from disk.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            If :obj:`"holdout"`, loads the holdout dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        backend (str): The :class:`Database` backend to use.
            (default: :obj:`"sqlite"`)
        from_smiles (callable, optional): A custom function that takes a SMILES
            string and outputs a :obj:`~torch_geometric.data.Data` object.
            If not set, defaults to :meth:`~torch_geometric.utils.from_smiles`.
            (default: :obj:`None`)
    """
    url = ('https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/'
           'pcqm4m-v2.zip')

    split_mapping = {
        'train': 'train',
        'val': 'valid',
        'test': 'test-dev',
        'holdout': 'test-challenge',
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        backend: str = 'sqlite',
        from_smiles: Optional[Callable] = None,
    ) -> None:
        assert split in ['train', 'val', 'test', 'holdout']

        schema = {
            'x': dict(dtype=torch.int64, size=(-1, 9)),
            'edge_index': dict(dtype=torch.int64, size=(2, -1)),
            'edge_attr': dict(dtype=torch.int64, size=(-1, 3)),
            'smiles': str,
            'y': float,
        }

        self.from_smiles = from_smiles or _from_smiles
        super().__init__(root, transform, backend=backend, schema=schema)

        split_idx = fs.torch_load(self.raw_paths[1])
        self._indices = split_idx[self.split_mapping[split]].tolist()

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('pcqm4m-v2', 'raw', 'data.csv.gz'),
            osp.join('pcqm4m-v2', 'split_dict.pt'),
        ]

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        data_list: List[Data] = []
        iterator = enumerate(zip(df['smiles'], df['homolumogap']))
        for i, (smiles, y) in tqdm(iterator, total=len(df)):
            data = self.from_smiles(smiles)
            data.y = y

            data_list.append(data)
            if i + 1 == len(df) or (i + 1) % 1000 == 0:  # Write batch-wise:
                self.extend(data_list)
                data_list = []

    def serialize(self, data: BaseData) -> Dict[str, Any]:
        assert isinstance(data, Data)
        return dict(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
            smiles=data.smiles,
        )

    def deserialize(self, data: Dict[str, Any]) -> Data:
        return Data.from_dict(data)
