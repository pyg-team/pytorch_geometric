import os
import os.path as osp
import pickle
import numpy as np

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url)


class MixHopSyntheticDataset(InMemoryDataset):
    r"""The MixHop synthetic dataset from the `"MixHop: Higher-Order
    Graph Convolutional Architectures via Sparsified Neighborhood Mixing"
    <https://arxiv.org/abs/1905.00067>`_ paper, containing 10
    graphs each with varying degree of homophily, from 0 to 0.9.
    The task is node multi-class classification.
    All graphs have 5000 nodes and 10 equal size classes.
    The feature values of the nodes are sampled from 2D Gaussians.
    In particular, each node class has its own Gaussian.
    For more details, see the `"MixHop Supplementary PDF"
    <http://proceedings.mlr.press/v97/abu-el-haija19a/abu-el-haija19a-supp.pdf>`_

    Args:
        root (string): Root directory where the dataset should be saved.
        homophily(string): The degree of homophily. It must be in
            ['0.0', '0.1', ... '0.9' ].
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

    url = 'https://raw.githubusercontent.com/samihaija/mixhop/master/data/synthetic'
    
    def __init__(self, root, homophily,
                       transform=None, pre_transform=None, pre_filter=None):
        self.homophily = homophily
        assert homophily in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9' ]
        super(MixHopSyntheticDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        path = osp.join(self.processed_dir, self.processed_file_names[0])
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        root_name = f'ind.n5000-h{self.homophily}-c10'
        return [ f'{root_name}.allx',
                 f'{root_name}.ally',
                 f'{root_name}.graph' ]
    
    @property
    def processed_file_names(self):
        return [ f'ind.n5000-h{self.homophily}-c10.pt' ]

    def download(self):
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self):
        filename = osp.join(self.raw_dir, self.raw_file_names[0])
        x = np.load(filename)
        x = torch.Tensor(x)
        
        filename = osp.join(self.raw_dir, self.raw_file_names[1])
        y = np.load(filename)
        # each row is one-hot encoded (c=10 classes)
        # we transform each row into a scalar from 0 to c
        y = np.sum( y * np.arange(10).reshape(1,-1), 1) 
        y = torch.tensor(y,dtype=torch.long)
        
        filename = osp.join(self.raw_dir, self.raw_file_names[2])
        edge_lists = pickle.load(open(filename, 'rb'), encoding='latin1')
        edge_index = []
        for key, value in edge_lists.items():
            if not value:
                # isolated node
                continue
            k = [ key ] * len(value)
            edge_index.append( [ k, value ] )
            
        edge_index = np.concatenate(edge_index, axis=1)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        split = len(x) / 3
        mask = np.array( [ False ] * len(x) )
        data.train_mask = np.copy(mask)
        data.train_mask[ np.arange(split).astype(int) ] = True
        data.val_mask = np.copy(mask)
        data.val_mask[ np.arange(split,2*split).astype(int) ] = True
        data.test_mask = np.copy(mask)
        data.test_mask[ np.arange(2*split,len(x)).astype(int) ] = True
        
        if self.pre_filter:
            self.pre_filter(data)
            
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        data_list = [ data ]
        
        torch.save(self.collate(data_list),
                   osp.join(self.processed_dir, self.processed_file_names[0]))
