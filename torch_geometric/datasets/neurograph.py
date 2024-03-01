import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class NeuroGraphDataset(InMemoryDataset):
    r"""The NeuroGraph benchmark datasets from the
    `"NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics"
    <https://arxiv.org/abs/2306.06202>`_ paper.
    :class:`NeuroGraphDataset` holds a collection of five neuroimaging graph
    learning datasets that span multiple categories of demographics, mental
    states, and cognitive traits.
    See the `documentation
    <https://neurograph.readthedocs.io/en/latest/NeuroGraph.html>`_ and the
    `Github <https://github.com/Anwar-Said/NeuroGraph>`_ for more details.

    +--------------------+---------+----------------------+
    | Dataset            | #Graphs | Task                 |
    +====================+=========+======================+
    | :obj:`HCPTask`     | 7,443   | Graph Classification |
    +--------------------+---------+----------------------+
    | :obj:`HCPGender`   | 1,078   | Graph Classification |
    +--------------------+---------+----------------------+
    | :obj:`HCPAge`      | 1,065   | Graph Classification |
    +--------------------+---------+----------------------+
    | :obj:`HCPFI`       | 1,071   | Graph Regression     |
    +--------------------+---------+----------------------+
    | :obj:`HCPWM`       | 1,078   | Graph Regression     |
    +--------------------+---------+----------------------+

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (one of :obj:`"HCPGender"`,
            :obj:`"HCPTask"`, :obj:`"HCPAge"`, :obj:`"HCPFI"`,
            :obj:`"HCPWM"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = 'https://vanderbilt.box.com/shared/static'
    filenames = {
        'HCPGender': 'r6hlz2arm7yiy6v6981cv2nzq3b0meax.zip',
        'HCPTask': '8wzz4y17wpxg2stip7iybtmymnybwvma.zip',
        'HCPAge': 'lzzks4472czy9f9vc8aikp7pdbknmtfe.zip',
        'HCPWM': 'xtmpa6712fidi94x6kevpsddf9skuoxy.zip',
        'HCPFI': 'g2md9h9snh7jh6eeay02k1kr9m4ido9f.zip',
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in self.filenames.keys()
        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        return 'data.pt'

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = f'{self.url}/{self.filenames[self.name]}'
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        os.rename(
            osp.join(self.raw_dir, self.name, 'processed', f'{self.name}.pt'),
            osp.join(self.raw_dir, 'data.pt'))
        fs.rm(osp.join(self.raw_dir, self.name))

    def process(self) -> None:
        data, slices = torch.load(self.raw_paths[0])

        num_samples = slices['x'].size(0) - 1
        data_list: List[Data] = []
        for i in range(num_samples):
            x = data.x[slices['x'][i]:slices['x'][i + 1]]
            start = slices['edge_index'][i]
            end = slices['edge_index'][i + 1]
            edge_index = data.edge_index[:, start:end]
            sample = Data(x=x, edge_index=edge_index, y=data.y[i])

            if self.pre_filter is not None and not self.pre_filter(sample):
                continue

            if self.pre_transform is not None:
                sample = self.pre_transform(sample)

            data_list.append(sample)

        self.save(data_list, self.processed_paths[0])
