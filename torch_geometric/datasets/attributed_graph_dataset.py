import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import scipy.sparse as sp
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.typing import SparseTensor


class AttributedGraphDataset(InMemoryDataset):
    r"""A variety of attributed graph datasets from the
    `"Scaling Attributed Network Embedding to Massive Graphs"
    <https://arxiv.org/abs/2009.00826>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
            :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
            :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
            :obj:`"TWeibo"`, :obj:`"MAG"`).
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
        * - Wiki
          - 2,405
          - 17,981
          - 4,973
          - 17
        * - Cora
          - 2,708
          - 5,429
          - 1,433
          - 7
        * - CiteSeer
          - 3,312
          - 4,715
          - 3,703
          - 6
        * - PubMed
          - 19,717
          - 44,338
          - 500
          - 3
        * - BlogCatalog
          - 5,196
          - 343,486
          - 8,189
          - 6
        * - PPI
          - 56,944
          - 1,612,348
          - 50
          - 121
        * - Flickr
          - 7,575
          - 479,476
          - 12,047
          - 9
        * - Facebook
          - 4,039
          - 88,234
          - 1,283
          - 193
        * - TWeibo
          - 2,320,895
          - 9,840,066
          - 1,657
          - 8
        * - MAG
          - 59,249,719
          - 978,147,253
          - 2,000
          - 100
    """
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    datasets = {
        'wiki': '1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV',
        'cora': '1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey',
        'citeseer': '1d3uQIpHiemWJPgLgTafi70RFYye7hoCp',
        'pubmed': '1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k',
        'blogcatalog': '178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS',
        'ppi': '1dvwRpPT4gGtOcNP_Q-G1TKl9NezYhtez',
        'flickr': '1tZp3EB20fAC27SYWwa-x66_8uGsuU62X',
        'facebook': '12aJWAGCM4IvdGI2fiydDNyWzViEOLZH8',
        'twitter': '1fUYggzZlDrt9JsLsSdRUHiEzQRW1kSA4',
        'tweibo': '1-2xHDPFCsuBuFdQN_7GLleWa8R_t50qU',
        'mag': '1ggraUMrQgdUyA3DjSRzzqMv0jFkU65V5',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['attrs.npz', 'edgelist.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.url.format(self.datasets[self.name])
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        path = osp.join(self.raw_dir, f'{self.name}.attr')
        if self.name == 'mag':
            path = osp.join(self.raw_dir, self.name)
        for name in self.raw_file_names:
            os.rename(osp.join(path, name), osp.join(self.raw_dir, name))
        shutil.rmtree(path)

    def process(self):
        import pandas as pd

        x = sp.load_npz(self.raw_paths[0])
        if x.shape[-1] > 10000 or self.name == 'mag':
            x = SparseTensor.from_scipy(x).to(torch.float)
        else:
            x = torch.from_numpy(x.todense()).to(torch.float)

        df = pd.read_csv(self.raw_paths[1], header=None, sep=None,
                         engine='python')
        edge_index = torch.from_numpy(df.values).t().contiguous()

        with open(self.raw_paths[2], 'r') as f:
            ys = f.read().split('\n')[:-1]
            ys = [[int(y) - 1 for y in row.split()[1:]] for row in ys]
            multilabel = max([len(y) for y in ys]) > 1

        if not multilabel:
            y = torch.tensor(ys).view(-1)
        else:
            num_classes = max([y for row in ys for y in row]) + 1
            y = torch.zeros((len(ys), num_classes), dtype=torch.float)
            for i, row in enumerate(ys):
                for j in row:
                    y[i, j] = 1.

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
