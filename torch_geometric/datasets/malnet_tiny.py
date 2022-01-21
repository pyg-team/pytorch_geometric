import glob
import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data, InMemoryDataset, download_url, extract_tar
)
from torch_geometric.utils import remove_isolated_nodes


class MalNetTiny(InMemoryDataset):
    r"""The MalNet Tiny dataset from the
    `"A Large-Scale Database for Graph Representation Learning"
    <https://openreview.net/pdf?id=1xDTDk3XPW>`_ paper.
    :class:`MalNetTiny` contains 5,000 malicious and benign software function
    call graphs across 5 different types. Each graph contains at most 5k nodes.

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
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'http://malnet.cc.gatech.edu/graph-data/malnet-graphs-tiny.tar.gz'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        folders = ['addisplay', 'adware', 'benign', 'downloader', 'trojan']
        return [osp.join('malnet-graphs-tiny', folder) for folder in folders]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = []

        for y, raw_path in enumerate(self.raw_paths):
            raw_path = osp.join(raw_path, os.listdir(raw_path)[0])
            filenames = glob.glob(osp.join(raw_path, '*.edgelist'))

            for filename in filenames:
                with open(filename, 'r') as f:
                    edges = f.read().split('\n')[5:-1]
                edge_index = [[int(s) for s in edge.split()] for edge in edges]
                edge_index = torch.tensor(edge_index).t().contiguous()
                # Remove isolated nodes, including those with only a self-loop
                edge_index = remove_isolated_nodes(edge_index)[0]
                num_nodes = int(edge_index.max()) + 1
                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
