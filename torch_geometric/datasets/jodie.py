import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url


class JODIEDataset(InMemoryDataset):
    r"""The temporal graph datasets
    from the `"JODIE: Predicting Dynamic Embedding
    Trajectory in Temporal Interaction Networks"
    <https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Reddit"`,
            :obj:`"Wikipedia"`, :obj:`"MOOC"`, and :obj:`"LastFM"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Reddit
          - 6,509
          - 25,470
          - 172
          - 1
        * - Wikipedia
          - 9,227
          - 157,474
          - 172
          - 2
        * - MOOC
          - 7,144
          - 411,749
          - 4
          - 2
        * - LastFM
          - 1,980
          - 1,293,103
          - 2
          - 1
    """
    url = 'http://snap.stanford.edu/jodie/{}.csv'
    names = ['reddit', 'wikipedia', 'mooc', 'lastfm']

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in self.names

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0], skiprows=1, header=None)

        src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
        dst += int(src.max()) + 1
        t = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
        y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
        msg = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float)

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
