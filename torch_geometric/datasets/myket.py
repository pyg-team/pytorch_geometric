from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url


class MyketDataset(InMemoryDataset):
    r"""Myket Android Application Install dataset
    from the `"Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks"
    <https://arxiv.org/pdf/2308.06862.pdf>`_ paper. The dataset contains a temporal graph of application install interactions in an android application market.

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

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Myket
          - 17,988
          - 694,121
          - 33
          - 1
    """
    url = 'https://raw.githubusercontent.com/erfanloghmani/myket-android-application-market-dataset/main/data_int_index/{}'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return ['myket.csv', 'app_info_sample.npy']

    @property
    def processed_file_names(self) -> str:
        return 'myket.pt'

    def download(self):
        for f in self.raw_file_names:
            download_url(self.url.format(f), self.raw_dir)

    def process(self):
        import numpy as np
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0], skiprows=1, header=None)

        src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
        t = torch.from_numpy(df.iloc[:, 2].values).to(torch.float)

        features = np.load(self.raw_paths[1])
        msg = torch.from_numpy(features[df.iloc[:,
                                                1].values, :]).to(torch.float)
        dst += int(src.max()) + 1

        data = TemporalData(src=src, dst=dst, t=t, msg=msg)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
