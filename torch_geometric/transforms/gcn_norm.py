from typing import Union

import torch
from torch_sparse import spmm

from torch_geometric.data import Batch, Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('gcn_norm')
class GCNNorm(BaseTransform):
    r"""Applies the GCN normalization from the `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper (functional name: :obj:`gcn_norm`).

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.
    """
    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def forward(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
        assert 'edge_index' in data or 'adj_t' in data

        if isinstance(data, Batch):
            raise NotImplementedError(
                "Batch processing with sparse tensors is not supported.")
        else:
            if 'edge_index' in data:
                data.edge_index, data.edge_weight = gcn_norm(
                    data.edge_index, data.edge_weight, data.num_nodes,
                    add_self_loops=self.add_self_loops)
            else:
                adj_t = data.adj_t
                row, col, value = adj_t.coo()
                row, col, value = gcn_norm(row, col, value, data.num_nodes,
                                           add_self_loops=self.add_self_loops)
                adj_t = adj_t.__class__(torch.stack([row, col]), value,
                                        adj_t.size(), adj_t.is_sorted())
                data.adj_t = adj_t

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'add_self_loops={self.add_self_loops})')
