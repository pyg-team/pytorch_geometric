import torch
from torch_sparse import SparseTensor, matmul

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import OptTensor


@functional_transform('feature_propagation')
class FeaturePropagation(BaseTransform):
    r"""The feature propagation operator from the
    `"On the Unreasonable Effectiveness of Feature
    propagation in Learning on Graphs with Missing Node Features"
    <https://arxiv.org/abs/2111.12128>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{X},

        \mathbf{X}_{k} = \mathbf{X}^{\prime}_{k}


    where missing node features are inferred by known features via propagation.

    .. note::

        For an example of using :class:`~torch_geometric.transforms.
        FeaturePropagation`, see `examples/feature_propagation.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        feature_propagation.py>`_.

    Args:
        num_iterations (int): The number of propagations. (default: :obj:`40`)
        missing_mask (Tensor, optional): The mask indicating where
            node features are missing, of shape `[num_nodes, channels]`. (default: :obj:`None`)
    """
    def __init__(self, num_iterations: int = 40,
                 missing_mask: OptTensor = None):
        assert missing_mask is None or missing_mask.dtype == torch.bool
        self.num_iterations = num_iterations
        self.missing_mask = missing_mask

    def __call__(self, data: Data) -> Data:

        assert 'edge_index' in data or 'adj_t' in data
        assert self.missing_mask is None or data.x.size(
        ) == self.missing_mask.size()

        if 'edge_index' in data:
            edge_weight = data.edge_attr
            if 'edge_weight' in data:
                edge_weight = data.edge_weight
            edge_index = data.edge_index
            adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
                                 value=edge_weight,
                                 sparse_sizes=data.size()[::-1],
                                 is_sorted=False, trust_data=True)
        else:
            adj_t = data.adj_t

        adj_t = gcn_norm(adj_t)

        x = data.x
        known_feature_mask = None
        if self.missing_mask is not None:
            x[self.missing_mask] = 0
            known_feature_mask = ~self.missing_mask

        for _ in range(self.num_iterations):
            out = matmul(adj_t, x)
            if known_feature_mask is not None:
                # Reset original known features
                out[known_feature_mask] = x[known_feature_mask]
            x = out
        data.x = out

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(num_iterations={self.num_iterations})')
