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
            node features are missing. (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
    """
    def __init__(self, num_iterations: int = 40,
                 missing_mask: OptTensor = None, add_self_loops: bool = True,
                 normalize: bool = True):
        assert missing_mask is None or missing_mask.dtype == torch.bool
        self.num_iterations = num_iterations
        self.missing_mask = missing_mask
        self.normalize = normalize
        self.add_self_loops = add_self_loops

    def __call__(self, data: Data) -> Data:

        assert 'edge_index' in data or 'adj_t' in data

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

        if self.normalize:
            adj_t = gcn_norm(adj_t, add_self_loops=self.add_self_loops)

        x = data.x
        known_feature_mask = None
        missing_mask = self.missing_mask
        if missing_mask is not None:
            out = torch.zeros_like(x)
            known_feature_mask = ~missing_mask
            out[known_feature_mask] = x[known_feature_mask]
        else:
            out = x.clone()

        for _ in range(self.num_iterations):
            # Diffuse current features
            out = matmul(adj_t, out)
            if known_feature_mask is not None:
                # Reset original known features
                out[known_feature_mask] = x[known_feature_mask]
        data.x = out

        return data
