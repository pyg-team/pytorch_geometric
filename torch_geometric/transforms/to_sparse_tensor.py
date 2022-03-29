from typing import Optional, Union

from torch_sparse import SparseTensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import sort_edge_index


@functional_transform('to_sparse_tensor')
class ToSparseTensor(BaseTransform):
    r"""Converts the :obj:`edge_index` attributes of a homogeneous or
    heterogeneous data object into a (transposed)
    :class:`torch_sparse.SparseTensor` type with key :obj:`adj_t`
    (functional name: :obj:`to_sparse_tensor`).

    .. note::

        In case of composing multiple transforms, it is best to convert the
        :obj:`data` object to a :obj:`SparseTensor` as late as possible, since
        there exist some transforms that are only able to operate on
        :obj:`data.edge_index` for now.

    Args:
        attr: (str, optional): The name of the attribute to add as a value to
            the :class:`~torch_sparse.SparseTensor` object (if present).
            (default: :obj:`edge_weight`)
        remove_edge_index (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
            (default: :obj:`True`)
        fill_cache (bool, optional): If set to :obj:`False`, will not fill the
            underlying :obj:`SparseTensor` cache. (default: :obj:`True`)
    """
    def __init__(self, attr: Optional[str] = 'edge_weight',
                 remove_edge_index: bool = True, fill_cache: bool = True):
        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            keys, values = [], []
            for key, value in store.items():
                if key == 'edge_index':
                    continue

                if store.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)

            store.edge_index, values = sort_edge_index(store.edge_index,
                                                       values,
                                                       sort_by_row=False)

            for key, value in zip(keys, values):
                store[key] = value

            store.adj_t = SparseTensor(
                row=store.edge_index[1], col=store.edge_index[0],
                value=None if self.attr is None or self.attr not in store else
                store[self.attr], sparse_sizes=store.size()[::-1],
                is_sorted=True, trust_data=True)

            if self.remove_edge_index:
                del store['edge_index']
                if self.attr is not None and self.attr in store:
                    del store[self.attr]

            if self.fill_cache:  # Pre-process some important attributes.
                store.adj_t.storage.rowptr()
                store.adj_t.storage.csr2csc()

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
