import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)


class EmailEUCore(InMemoryDataset):
    r"""An e-mail communication network of a large European research
    institution, taken from the `"Local Higher-order Graph Clustering"
    <https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>`_ paper.
    Nodes indicate members of the institution.
    An edge between a pair of members indicates that they exchanged at least
    one email.
    Node labels indicate membership to one of the 42 departments.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    urls = [
        'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
        'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz'
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['email-Eu-core.txt', 'email-Eu-core-department-labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for url in self.urls:
            path = download_url(url, self.raw_dir)
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        edge_index = pd.read_csv(self.raw_paths[0], sep=' ', header=None)
        edge_index = torch.from_numpy(edge_index.values).t().contiguous()

        y = pd.read_csv(self.raw_paths[1], sep=' ', header=None, usecols=[1])
        y = torch.from_numpy(y.values).view(-1)

        data = Data(edge_index=edge_index, y=y, num_nodes=y.size(0))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
