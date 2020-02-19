import copy
import os.path as osp

import torch
import torch.utils.data
from torch_sparse import SparseTensor, cat


class ClusterDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_parts, save=True):
        assert len(dataset) == 1
        assert (dataset[0].edge_index is not None)

        self.dataset = dataset
        self.num_parts = num_parts
        self.save = osp.exists(self.dataset.processed_dir) and save

        self.process()

    def process(self):
        filename = f'part_data_{self.num_parts}.pt'

        path = osp.join(self.dataset.processed_dir, filename)
        if self.save and osp.exists(path):
            data, partptr, perm = torch.load(path)
        else:
            data = copy.copy(self.dataset.get(0))

            (row, col), edge_attr = data.edge_index, data.edge_attr
            adj = SparseTensor(row=row, col=col, value=edge_attr,
                               is_sorted=True)
            adj, partptr, perm = adj.partition_kway(self.num_parts)

            for key, item in data:
                if item.size(0) == data.num_nodes:
                    data[key] = item[perm]

            data.edge_index = None
            data.edge_attr = None
            data.adj = adj

            if self.save:
                torch.save((data, partptr, perm), path)

        self.__data__ = data
        self.__perm__ = perm
        self.__partptr__ = partptr

    def __len__(self):
        return self.__partptr__.numel() - 1

    def __getitem__(self, idx):
        start = int(self.__partptr__[idx])
        length = int(self.__partptr__[idx + 1]) - start

        data = copy.copy(self.__data__)
        for key, item in data:
            if item.size(0) == data.num_nodes:
                data[key] = item.narrow(0, start, length)

        data.adj = data.adj.narrow(1, start, length)

        row, col, value = data.adj.coo()
        data.adj = None
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        if self.dataset.transform is not None:
            data = self.dataset.transform(data)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.dataset}, '
                f'num_parts={self.num_parts})')


class ClusterLoader(torch.utils.data.DataLoader):
    def __init__(self, cluster_dataset, batch_size=1, shuffle=False, **kwargs):
        class HelperDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(cluster_dataset)

            def __getitem__(self, idx):
                start = int(cluster_dataset.__partptr__[idx])
                length = int(cluster_dataset.__partptr__[idx + 1]) - start

                data = copy.copy(cluster_dataset.__data__)
                num_nodes = data.num_nodes
                for key, item in data:
                    if item.size(0) == num_nodes:
                        data[key] = item.narrow(0, start, length)

                return data, idx

        def collate(data):
            data_list, parts = [d[0] for d in data], [d[1] for d in data]
            partptr = cluster_dataset.__partptr__

            adj = cat([data.adj for data in data_list], dim=0)

            adj = adj.t()
            adjs = []
            for part in parts:
                start = partptr[part]
                length = partptr[part + 1] - start
                adjs.append(adj.narrow(0, start, length))
            adj = cat(adjs, dim=0).t()
            row, col, value = adj.coo()

            data = cluster_dataset.__data__.__class__()
            data.num_nodes = adj.size(0)
            data.edge_index = torch.stack([row, col], dim=0)
            data.edge_attr = value

            ref = data_list[0]
            keys = ref.keys
            keys.remove('adj')

            for key in keys:
                if ref[key].size(0) != ref.adj.size(0):
                    data[key] = ref[key]
                else:
                    data[key] = torch.cat([d[key] for d in data_list],
                                          dim=ref.__cat_dim__(key, ref[key]))

            if cluster_dataset.dataset.transform is not None:
                data = cluster_dataset.dataset.transform(data)

            return data

        super(ClusterLoader,
              self).__init__(HelperDataset(), batch_size, shuffle,
                             collate_fn=collate, **kwargs)
