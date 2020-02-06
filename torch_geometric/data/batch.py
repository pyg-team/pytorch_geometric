import numpy as np
import torch
import torch_geometric
from torch_sparse import SparseTensor, cat_diag
from torch_geometric.data import Data


class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    __data_class__ = Data
    __private_attrs__ = Data.__private_attrs__

    def __init__(self, batch=None, ptrs=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__ptrs__ = ptrs

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        length = len(data_list)
        assert length > 0

        ptrs = {}
        keys = data_list[0].keys
        for key in keys:
            item = data_list[0][key]
            if isinstance(item, SparseTensor):
                ptrs[key] = torch.zeros((2, length + 1), dtype=torch.long)
            else:
                ptrs[key] = torch.zeros((length + 1, ), dtype=torch.long)

        if data_list[0].num_nodes is not None:
            ptrs['num_nodes'] = torch.zeros((length + 1, ), dtype=torch.long)

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__ptrs__ = ptrs

        assert 'batch' not in keys and '__ptrs__' not in keys
        for key in follow_batch:
            assert f'{key}_batch' not in keys

        for key in keys:
            batch[key] = []

        cumsum = {key: 0 for key in keys}
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                if isinstance(item, np.ndarray):
                    item = torch.from_numpy(item)

                if torch.is_tensor(item):
                    if cumsum[key] != 0:
                        item = item + cumsum[key]
                    cumsum[key] += data.__inc__(key, item)
                    size = item.size(data.__cat_dim__(key, item))
                elif isinstance(item, SparseTensor):
                    size = torch.tensor([item.size(0), item.size(1)])
                elif isinstance(item, list) or isinstance(item, tuple):
                    size = len(item)
                else:
                    size = 1

                batch[key].append(item)
                ptrs[key][..., i + 1] = ptrs[key][..., i] + size

            num_nodes = data.num_nodes
            if num_nodes is not None:
                ptrs['num_nodes'][i + 1] = ptrs['num_nodes'][i] + num_nodes

        for key in keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = cat_diag(batch[key])
            elif key == 'num_nodes':
                batch[key] = sum(batch[key])
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

        if num_nodes is not None:
            ptr = ptrs['num_nodes']
            batch.batch = torch.repeat_interleave(torch.arange(length),
                                                  ptr[1:] - ptr[:-1])

        for key in follow_batch:
            ptr = ptrs[key]
            batch[f'{key}_batch'] = torch.repeat_interleave(
                torch.arange(length), ptr[1:] - ptr[:-1])

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        ptrs = self.__ptrs__

        if ptrs is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using `Batch.from_data_list()`'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        keys.remove('__data_class__')
        keys.remove('__ptrs__')

        data_list = []
        cumsum = {key: 0 for key in keys}
        for i in range(self.num_graphs):
            data = self.__data_class__()

            for key in keys:
                item = self[key]
                start = ptrs[key][..., i]
                length = ptrs[key][..., i + 1] - ptrs[key][..., i]
                if torch.is_tensor(item):
                    dim = data.__cat_dim__(key, item)
                    data[key] = item.narrow(dim, start, length)
                elif isinstance(item, SparseTensor):
                    data[key] = item.__narrow_diag__(start, length)
                elif key == 'num_nodes':
                    data[key] = int(length)
                else:
                    data[key] = item[int(start):int(start + length)]

            for key in keys:
                if torch.is_tensor(item):
                    if cumsum[key] != 0:
                        data[key] = data[key] - cumsum[key]
                    cumsum[key] += data.__inc__(key, data[key])

            data_list.append(data)

        return data_list

    @property
    def ptrs(self):
        return self.__ptrs__

    @property
    def num_graphs(self):
        r"""Returns the number of graphs in the batch."""
        if self.__ptrs__ is not None:
            for key, item in self.__ptrs__.items():
                return item.size(-1) - 1
        elif self.batch is not None:
            return self.batch.max().item() + 1
        else:
            raise ValueError
