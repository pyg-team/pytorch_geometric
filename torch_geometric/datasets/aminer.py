import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs
from torch_geometric.utils import coalesce


class AMiner(InMemoryDataset):
    r"""The heterogeneous AMiner dataset from the `"metapath2vec: Scalable
    Representation Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/
    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper, consisting of nodes from
    type :obj:`"paper"`, :obj:`"author"` and :obj:`"venue"`.
    Venue categories and author research interests are available as ground
    truth labels for a subset of nodes.

    Args:
        root: Root directory where the dataset should be saved.
        transform: A function/transform that takes in a
            :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access.
        pre_transform: A function/transform that takes in a
            :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk.
        force_reload: Whether to re-process the dataset.
    """

    url = 'https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip?dl=1'
    y_url = 'https://www.dropbox.com/s/nkocx16rpl4ydde/label.zip?dl=1'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'id_author.txt', 'id_conf.txt', 'paper.txt', 'paper_author.txt',
            'paper_conf.txt', 'label'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        fs.rm(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'net_aminer'), self.raw_dir)
        os.unlink(path)
        path = download_url(self.y_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Get author labels.
        path = osp.join(self.raw_dir, 'id_author.txt')
        author = pd.read_csv(path, sep='\t', names=['idx', 'name'],
                             index_col=1)

        path = osp.join(self.raw_dir, 'label',
                        'googlescholar.8area.author.label.txt')
        df = pd.read_csv(path, sep=' ', names=['name', 'y'])
        df = df.join(author, on='name')

        data['author'].y = torch.from_numpy(df['y'].values) - 1
        data['author'].y_index = torch.from_numpy(df['idx'].values)

        # Get venue labels.
        path = osp.join(self.raw_dir, 'id_conf.txt')
        venue = pd.read_csv(path, sep='\t', names=['idx', 'name'], index_col=1)

        path = osp.join(self.raw_dir, 'label',
                        'googlescholar.8area.venue.label.txt')
        df = pd.read_csv(path, sep=' ', names=['name', 'y'])
        df = df.join(venue, on='name')

        data['venue'].y = torch.from_numpy(df['y'].values) - 1
        data['venue'].y_index = torch.from_numpy(df['idx'].values)

        # Get paper<->author connectivity.
        path = osp.join(self.raw_dir, 'paper_author.txt')
        paper_author = pd.read_csv(path, sep='\t', header=None)
        paper_author = torch.from_numpy(paper_author.values)
        paper_author = paper_author.t().contiguous()
        M, N = int(paper_author[0].max() + 1), int(paper_author[1].max() + 1)
        paper_author = coalesce(paper_author, num_nodes=max(M, N))
        data['paper'].num_nodes = M
        data['author'].num_nodes = N
        data['paper', 'written_by', 'author'].edge_index = paper_author
        data['author', 'writes', 'paper'].edge_index = paper_author.flip([0])

        # Get paper<->venue connectivity.
        path = osp.join(self.raw_dir, 'paper_conf.txt')
        paper_venue = pd.read_csv(path, sep='\t', header=None)
        paper_venue = torch.from_numpy(paper_venue.values)
        paper_venue = paper_venue.t().contiguous()
        M, N = int(paper_venue[0].max() + 1), int(paper_venue[1].max() + 1)
        paper_venue = coalesce(paper_venue, num_nodes=max(M, N))
        data['venue'].num_nodes = N
        data['paper', 'published_in', 'venue'].edge_index = paper_venue
        data['venue', 'publishes', 'paper'].edge_index = paper_venue.flip([0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
