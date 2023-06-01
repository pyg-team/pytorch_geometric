import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class AmazonBook(InMemoryDataset):
    r"""A subset of AmazonBook rating dataset from `"LightGCN: Simplifying
    and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper. This is a heterogeneous
    dataset consisting users (52,643 nodes) and books (91,599 nodes) with 2.9
    million ratings between them. No labels or features are provided. The
    dataset is split into train and test set.

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
    """

    url = ('https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/'
           'master/data/amazon-book/')

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['user_list.txt', 'item_list.txt', 'test.txt', 'train.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names:
            download_url(self.url + f, self.raw_dir)

    def process(self):
        import pandas as pd

        data = HeteroData()
        # Process num_nodes for user and book
        node_types = ['user', 'book']
        for i, node_type in enumerate(node_types):
            df = pd.read_csv(self.raw_paths[i], sep=' ', header=0)
            data[node_type].num_nodes = len(df)

        # Process edge_index for train and test set
        def helper(x, holder):
            x = str(x).split(' ')
            for i in x[1:]:
                edge = [int(x[0]), int(i)]
                holder.append(edge)

        for i in range(2, 4):
            with open(self.raw_paths[i], 'r') as f:
                lines = f.readlines()
                df = pd.DataFrame([line.strip('\n').strip() for line in lines],
                                  columns=['nodes'])

            index = []
            df.nodes.apply(lambda x: helper(x, index))
            index = torch.t(torch.tensor(index))
            name = self.raw_file_names[i].replace('.txt', '_edge_index')
            data['user', 'rates', 'book'][name] = index
            data['book', 'rated_by', 'user'][name] = index.flip([0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
