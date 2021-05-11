from typing import Optional, Callable

import torch

from torch_geometric.data.data2 import Data


class OGB_MAG(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform

        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset('ogbn-mag', root)
        tmp_data = dataset[0]
        split_idx = dataset.get_idx_split()

        data = Data()
        data['paper'].x = tmp_data.x_dict['paper']
        data['paper'].year = tmp_data.node_year['paper'].squeeze()

        for key in ['author', 'field_of_study', 'institution']:
            data[key].num_nodes = tmp_data.num_nodes_dict[key]

        for (src, _, dst), edge_index in tmp_data.edge_index_dict.items():
            data[src, dst].edge_index = edge_index

        y = tmp_data.y_dict['paper'].squeeze()
        data['paper'].train_y = y[split_idx['train']['paper']].to(torch.long)
        data['paper'].train_idx = split_idx['train']['paper']
        data['paper'].val_y = y[split_idx['valid']['paper']].to(torch.long)
        data['paper'].val_idx = split_idx['valid']['paper']
        data['paper'].test_y = y[split_idx['test']['paper']].to(torch.long)
        data['paper'].test_idx = split_idx['test']['paper']

        if pre_transform is not None:
            raise NotImplementedError

        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.transform is not None:
            raise NotImplementedError
        return self.data

    def __repr__(self):
        return f'{self.__class__.__name__}()'
