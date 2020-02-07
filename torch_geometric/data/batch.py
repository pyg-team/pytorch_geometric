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
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__ptrs__ = {}
        self.__incs__ = {}
        self.__data_class__ = Data
        self.__keys__ = []
        self.__public_keys__ = []

    @property
    def __hidden_keys__(self):
        return self.__data_class__().__hidden_keys__ + [
            '__ptrs__', '__incs__', '__data_class__', '__keys__',
            '__public_keys__'
        ]

    @property
    def __private_keys__(self):
        return self.__data_class__().__private_keys__

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        length = len(data_list)
        assert length > 0

        ref = data_list[0]
        keys = [key for key in ref.__dict__.keys() if ref[key] is not None]
        public_keys = [
            key[2:-2] if key in ref.__private_keys__ else key for key in keys
        ]

        assert 'batch' not in keys
        assert '__ptrs__' not in keys
        assert '__incs__' not in keys
        assert '__data_class__' not in keys
        assert '__keys__' not in keys
        assert '__public_keys__' not in keys
        for key in follow_batch:
            assert f'{key}_batch' not in keys

        batch = Batch()
        ptrs, incs = batch.__ptrs__, batch.__incs__
        batch.__data_class__ = data_list[0].__class__
        batch.__keys__, batch.__public_keys__ = keys, public_keys

        for key, public_key in zip(keys, public_keys):
            item = ref[key]
            if isinstance(item, SparseTensor):
                size = (2, length + 1)
                ptrs[public_key] = torch.zeros(size, dtype=torch.long)
            else:
                size = (length + 1, )
                ptrs[public_key] = torch.zeros(size, dtype=torch.long)

            if (torch.is_tensor(item) or isinstance(item, int)
                    or isinstance(item, float)):
                incs[public_key] = [0]

            batch[key] = []

        batch_indices = []
        for i, data in enumerate(data_list):
            for key, public_key in zip(keys, public_keys):
                item = data[key]

                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(public_key, item))
                elif isinstance(item, SparseTensor):
                    size = torch.tensor([item.size(0), item.size(1)])
                elif isinstance(item, list) or isinstance(item, tuple):
                    size = len(item)
                else:
                    size = 1

                if (torch.is_tensor(item) or isinstance(item, int)
                        or isinstance(item, float)):
                    inc = incs[public_key][-1]
                    item = item + inc if inc != 0 else item
                    inc = inc + data.__inc__(public_key, item)
                    incs[public_key].append(inc)

                batch[key].append(item)
                ptrs[public_key][..., i + 1] = ptrs[public_key][..., i] + size

            num_nodes = data.num_nodes
            if num_nodes is not None:
                idx = torch.full((num_nodes, ), i, dtype=torch.long)
                batch_indices.append(idx)

        for key in keys:
            item = ref[key]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=ref.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = cat_diag(batch[key])
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                batch[key] = batch[key]

        if len(batch_indices) == length:
            batch['batch'] = torch.cat(batch_indices, dim=0)

        for key in follow_batch:
            ptr = ptrs[key]
            batch[f'{key}_batch'] = torch.repeat_interleave(
                torch.arange(length), ptr[1:] - ptr[:-1])

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.__data_class__()
            ptrs, incs = self.__ptrs__, self.__incs__

            for key, public_key in zip(self.__keys__, self.__public_keys__):
                item = self[key]
                start = ptrs[public_key][..., idx]
                length = ptrs[public_key][..., idx + 1] - start

                if torch.is_tensor(item):
                    dim = data.__cat_dim__(key, item)
                    item = item.narrow(dim, start, length)
                    if item.numel() == 1 and not torch.is_floating_point(item):
                        item = item.item()
                elif isinstance(item, SparseTensor):
                    item = item.__narrow_diag__(start, length)
                else:
                    item = item[int(start):int(start + length)]

                if (torch.is_tensor(item) or isinstance(item, int)
                        or isinstance(item, float)):
                    inc = incs[public_key][idx]
                    item = item - inc if inc != 0 else item

                data[key] = item

            return data
        else:
            return super(Batch, self).__getitem__(idx)

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        return [self[idx] for idx in range(self.num_graphs)]

    @property
    def num_nodes(self):
        if torch.is_tensor(self.__num_nodes__):
            return int(self.__num_nodes__.sum())
        else:
            return super(Batch, self).num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_graphs(self):
        r"""Returns the number of graphs in the batch."""
        for key, item in self.__ptrs__.items():
            return item.size(-1) - 1
        if self.batch is not None:
            return self.batch.max().item() + 1
        else:
            raise ValueError
