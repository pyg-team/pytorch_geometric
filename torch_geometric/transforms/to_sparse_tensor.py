from typing import Optional, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    sort_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


@functional_transform('to_sparse_tensor')
class ToSparseTensor(BaseTransform):
    r"""Converts the :obj:`edge_index` attributes of a homogeneous or
    heterogeneous data object into a **transposed**
    :class:`torch_sparse.SparseTensor` or :pytorch:`PyTorch`
    :class:`torch.sparse.Tensor` object with key :obj:`adj_t`
    (functional name: :obj:`to_sparse_tensor`).

    .. note::

        In case of composing multiple transforms, it is best to convert the
        :obj:`data` object via :class:`ToSparseTensor` as late as possible,
        since there exist some transforms that are only able to operate on
        :obj:`data.edge_index` for now.

    Args:
        attr (str, optional): The name of the attribute to add as a value to
            the :class:`~torch_sparse.SparseTensor` or
            :class:`torch.sparse.Tensor` object (if present).
            (default: :obj:`edge_weight`)
        remove_edge_index (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
            (default: :obj:`True`)
        fill_cache (bool, optional): If set to :obj:`True`, will fill the
            underlying :class:`torch_sparse.SparseTensor` cache (if used).
            (default: :obj:`True`)
        layout (torch.layout, optional): Specifies the layout of the returned
            sparse tensor (:obj:`None`, :obj:`torch.sparse_coo` or
            :obj:`torch.sparse_csr`).
            If set to :obj:`None` and the :obj:`torch_sparse` dependency is
            installed, will convert :obj:`edge_index` into a
            :class:`torch_sparse.SparseTensor` object.
            If set to :obj:`None` and the :obj:`torch_sparse` dependency is
            not installed, will convert :obj:`edge_index` into a
            :class:`torch.sparse.Tensor` object with layout
            :obj:`torch.sparse_csr`. (default: :obj:`None`)
    """
    def __init__(
        self,
        attr: Optional[str] = 'edge_weight',
        remove_edge_index: bool = True,
        fill_cache: bool = True,
        layout: Optional[int] = None,
    ):
        if layout not in {None, torch.sparse_coo, torch.sparse_csr}:
            raise ValueError(f"Unexpected sparse tensor layout "
                             f"(got '{layout}')")

        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache
        self.layout = layout

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

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

            store.edge_index, values = sort_edge_index(
                store.edge_index,
                values,
                sort_by_row=False,
            )

            for key, value in zip(keys, values):
                store[key] = value

            layout = self.layout
            size = store.size()[::-1]
            value: Optional[Tensor] = None
            if self.attr is not None and self.attr in store:
                value = store[self.attr]

            if layout is None and torch_geometric.typing.WITH_TORCH_SPARSE:
                store.adj_t = SparseTensor(
                    row=store.edge_index[1],
                    col=store.edge_index[0],
                    value=value,
                    sparse_sizes=size,
                    is_sorted=True,
                    trust_data=True,
                )

            # TODO Multi-dimensional edge attributes only supported for COO.
            elif ((value is not None and value.dim() > 1)
                  or layout == torch.sparse_coo):
                store.adj_t = to_torch_coo_tensor(
                    store.edge_index.flip([0]),
                    edge_attr=value,
                    size=size,
                )

            elif layout is None or layout == torch.sparse_csr:
                store.adj_t = to_torch_csr_tensor(
                    store.edge_index.flip([0]),
                    edge_attr=value,
                    size=size,
                )

            if self.remove_edge_index:
                del store['edge_index']
                if self.attr is not None and self.attr in store:
                    del store[self.attr]

            if self.fill_cache and isinstance(store.adj_t, SparseTensor):
                # Pre-process some important attributes.
                store.adj_t.storage.rowptr()
                store.adj_t.storage.csr2csc()

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(attr={self.attr}, '
                f'layout={self.layout})')
