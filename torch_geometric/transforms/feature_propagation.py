from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_torch_sparse_tensor, to_torch_csc_tensor


@functional_transform('feature_propagation')
class FeaturePropagation(BaseTransform):
    r"""The feature propagation operator from the `"On the Unreasonable
    Effectiveness of Feature propagation in Learning on Graphs with Missing
    Node Features" <https://arxiv.org/abs/2111.12128>`_ paper
    (functional name: :obj:`feature_propagation`)

    .. math::
        \mathbf{X}^{(0)} &= (1 - \mathbf{M}) \cdot \mathbf{X}

        \mathbf{X}^{(\ell + 1)} &= \mathbf{X}^{(0)} + \mathbf{M} \cdot
        (\mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \mathbf{X}^{(\ell)})

    where missing node features are inferred by known features via propagation.

    .. code-block:: python

        from torch_geometric.transforms import FeaturePropagation

        transform = FeaturePropagation(missing_mask=torch.isnan(data.x))
        data = transform(data)

    Args:
        missing_mask (torch.Tensor): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{N\times F}` indicating missing
            node features.
        num_iterations (int, optional): The number of propagations.
            (default: :obj:`40`)
    """
    def __init__(self, missing_mask: Tensor, num_iterations: int = 40):
        self.missing_mask = missing_mask
        self.num_iterations = num_iterations

    def forward(self, data: Data) -> Data:
        assert 'edge_index' in data or 'adj_t' in data
        assert data.x.size() == self.missing_mask.size()

        missing_mask = self.missing_mask.to(data.x.device)
        known_mask = ~missing_mask

        if 'edge_index' in data:
            edge_weight = data.edge_attr
            if 'edge_weight' in data:
                edge_weight = data.edge_weight
            edge_index = data.edge_index
            adj_t = to_torch_csc_tensor(edge_index, edge_weight,
                                        size=data.size()).t()
            adj_t, _ = gcn_norm(adj_t, add_self_loops=False)
        elif is_torch_sparse_tensor(data.adj_t):
            adj_t, _ = gcn_norm(data.adj_t, add_self_loops=False)
        else:
            adj_t = gcn_norm(data.adj_t, add_self_loops=False)

        x = data.x.clone()
        x[missing_mask] = 0.

        out = x
        for _ in range(self.num_iterations):
            out = adj_t @ out
            out[known_mask] = x[known_mask]  # Reset.
        data.x = out

        return data

    def __repr__(self) -> str:
        na_values = int(self.missing_mask.sum()) / self.missing_mask.numel()
        return (f'{self.__class__.__name__}('
                f'missing_features={100 * na_values:.1f}%, '
                f'num_iterations={self.num_iterations})')
