import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_tar)


class PolBlogs(InMemoryDataset):
    r"""The Political Blogs dataset from the `"The Political Blogosphere and
    the 2004 US Election: Divided they Blog"
    <https://dl.acm.org/doi/10.1145/1134271.1134277>`_ paper.

    :class:`Polblogs` is a graph with 1,490 vertices (representing political
    blogs) and 19,025 edges (links between blogs).
    The links are automatically extracted from a crawl of the front page of the
    blog.
    Each vertex receives a label indicating the political leaning of the blog:
    liberal or conservative.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://netset.telecom-paris.fr/datasets/polblogs.tar.gz'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adjacency.csv', 'labels.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        import pandas as pd

        edge_index = pd.read_csv(self.raw_paths[0], header=None)
        edge_index = torch.from_numpy(edge_index.values).t().contiguous()

        y = pd.read_csv(self.raw_paths[1], header=None)
        y = torch.from_numpy(y.values).view(-1)

        data = Data(edge_index=edge_index, y=y, num_nodes=y.size(0))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
