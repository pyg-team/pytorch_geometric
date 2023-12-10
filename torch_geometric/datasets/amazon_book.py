from typing import Callable, List, Optional

import torch

from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class AmazonBook(InMemoryDataset):
    r"""A subset of the AmazonBook rating dataset from the
    `"LightGCN: Simplifying and Powering Graph Convolution Network for
    Recommendation" <https://arxiv.org/abs/2002.02126>`_ paper.
    This is a heterogeneous dataset consisting of 52,643 users and 91,599 books
    with approximately 2.9 million ratings between them.
    No labels or features are provided.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = ('https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/'
           'master/data/amazon-book')

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
        return ['user_list.txt', 'item_list.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Process number of nodes for each node type:
        node_types = ['user', 'book']
        for path, node_type in zip(self.raw_paths, node_types):
            df = pd.read_csv(path, sep=' ', header=0)
            data[node_type].num_nodes = len(df)

        # Process edge information for training and testing:
        attr_names = ['edge_index', 'edge_label_index']
        for path, attr_name in zip(self.raw_paths[2:], attr_names):
            rows, cols = [], []
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                indices = line.strip().split(' ')
                for dst in indices[1:]:
                    rows.append(int(indices[0]))
                    cols.append(int(dst))
            index = torch.tensor([rows, cols])

            data['user', 'rates', 'book'][attr_name] = index
            if attr_name == 'edge_index':
                data['book', 'rated_by', 'user'][attr_name] = index.flip([0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
