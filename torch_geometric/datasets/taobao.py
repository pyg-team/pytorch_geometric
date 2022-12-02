import os
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

class Taobao(InMemoryDataset):
    r"""Taobao User Behavior is a dataset of user behaviors from Taobao offered
    by Alibaba. The dataset is from the platform Tianchi Alicloud.
    https://tianchi.aliyun.com/dataset/649.
    Taobao is a heterogeous graph for recommendation. Nodes represent users with
    User IDs and items with Item IDs and Category IDs, and edges represent
    different types of user behaviours towards items with timestamps.

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
    url = 'https://alicloud-dev.oss-cn-hangzhou.aliyuncs.com/UserBehavior.csv.zip'

    def __init__(self, root, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['UserBehavior.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        print(self.raw_dir)
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        import pandas as pd

        data = HeteroData()

        df = pd.read_csv(self.raw_paths[0])
        df.columns = ['userId','itemId','categoryId','behaviorType','timestamp']

        # Time representation (YYYY.MM.DD-HH:MM:SS -> Integer)
        # 1511539200 = 2017.11.25-00:00:00 1512316799 = 2017.12.03-23:59:59
        df = df[(df["timestamp"]>=1511539200) & (df["timestamp"]<=1512316799)]

        df = df.drop_duplicates(
            subset=['userId','itemId','categoryId','behaviorType','timestamp'],
            keep='first')

        behavior_dict = {'pv': 0, 'cart': 1, 'buy': 2, 'fav': 3}
        df['behaviorType'] = df['behaviorType'].map(behavior_dict).values
        _, df['userId'] = np.unique(df[['userId']].values, return_inverse=True)
        _, df['itemId'] = np.unique(df[['itemId']].values, return_inverse=True)
        _, df['categoryId'] = np.unique(df[['categoryId']].values, return_inverse=True)

        data['user'].num_nodes = df['userId'].nunique()
        data['item'].num_nodes = df['itemId'].nunique()

        edge_feat, _ = np.unique(
            df[['userId', 'itemId', 'behaviorType', 'timestamp']].values,
            return_index=True, axis=0)
        edge_feat = pd.DataFrame(edge_feat).drop_duplicates(subset=[0, 1], keep='last')
        data['user', '2', 'item'].edge_index = torch.from_numpy(edge_feat[[0, 1]].values).T
        data['user', '2', 'item'].edge_attr = torch.from_numpy(edge_feat[[2, 3]].values).T

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
