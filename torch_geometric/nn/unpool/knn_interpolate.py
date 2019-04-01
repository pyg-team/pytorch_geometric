from torch_geometric.nn import knn
from torch_geometric.utils import scatter_


def knn_interpolate(x_pos, y_pos, x_x, y_x=None, batch_x=None, batch_y=None,
                    k=3, cat=True):
    r"""A simple knn interpolation from `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper. For each point :obj:`y` 
    its interpolated feature are obtained by the formula

    .. math::
        f(y) = \frac{\sum_{i=1}^k w_i(y) f_i}{\sum_{i=1}^k w_i(y)}
        \mbox{, where } w_i(y) = \frac{1}{d(y, x_i)^2}.

    Args:
        x_pos (Tensor): Node position matrix
            :math:`\mathbf{X_pos} \in \mathbb{R}^{N \times d}`.
        y_pos (Tensor): Node position matrix
            :math:`\mathbf{Y_pos} \in \mathbb{R}^{M \times d}`.
        x_x (Tensor): Node feature matrix
            :math:`\mathbf{X_feat} \in \mathbb{R}^{N \times F_x}`.
        y_x (Tensor, optional): Node feature matrix
            :math:`\mathbf{Y_feat} \in \mathbb{R}^{M \times F_y}`.
            (default: :obj:`None`)
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node from :math:`\mathbf{X}` to a specific example.
            (default: :obj:`None`)
        bathc_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node from :math:`\mathbf{Y}` to a specific example.
            (default: :obj:`None`)
        k (int, optional): Number of neighbors. (default: :obj:`3`)
        cat (bool, optional): If set to :obj:`False`, all existing
            features of nodes from :obj:`Y` will be replaced.
            (default: :obj:`True`)

    Returns:
        Tensor: New node feature matrix :math:`\mathbf{Y_feat}`.
     """
    assert x_pos.size(0) == x_x.size(0)
    assert y_pos.size(0) == y_x.size(0)
    assert x_pos.size(0) == batch_x.size(0)
    assert y_pos.size(0) == batch_y.size(0)

    with torch.no_grad():
        y_idx, x_idx = knn(x_pos, y_pos, k, batch_x=batch_x, batch_y=batch_y)
        diff = x_pos[x_idx] - y_pos[y_idx]
        squared_distance = torch.sum(diff * diff, dim=-1, keepdim=True)
        weights = 1. / torch.clamp(squared_distance, 1e-16)
    
    interpolated_features = (scatter_('add', x_x[x_idx] * weights, y_idx)
                             / scatter_('add', weights, y_idx))

    if y_x is not None and cat:
        y_x = torch.cat([y_x, interpolated_features], dim=-1)
    else:
        y_x = interpolated_features

    return y_x
