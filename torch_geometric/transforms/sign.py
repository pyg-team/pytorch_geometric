import torch
from torch_sparse import SparseTensor


class SIGN(object):
    r"""The Scalable Inception Graph Neural Network module (SIGN) from the
    `"SIGN: Scalable Inception Graph Neural Networks"
    <https://arxiv.org/abs/2004.11198>`_ paper, which precomputes the fixed
    representations

    .. math::
        \mathbf{X}^{(i)} = \left( \left( p \mathbf{I} + (1-p) \mathbf{D}
        \right)^{-\alpha} \left( \mathbf{A} + r \mathbf{I} \right)
        \left( q \mathbf{I} + (1-q) \mathbf{D}
        \right)^{-\beta} \right)^i \mathbf{X}

    for :math:`i \in \{ 1, \ldots, K \}` and saves them in
    :obj:`data.x1`, :obj:`data.x2`, ...
    
    Notice that :math:`p=0`, :math:`q=0`, :math:`r=0`, :math:`\alpha=1/2`
    and :math:`\beta=1/2` (the default values) gives the normalized adjacency matrix
    :math:`\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}`
    whereas :math:`p=1/2`, :math:`q=1/2`, :math:`r=1`, :math:`\alpha=1/2`
    and :math:`\beta=1/2` gives the normalized adjacency matrix with self-loops
    :math:`\hat{\mathbf{D}}^{-1/2}\hat{\mathbf{A}}
    \hat{\mathbf{D}}^{-1/2}` (up to a constant of proportionality)
    from the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.
    
    The continuum of propagation matrices given by varying :math:`\alpha`
    and :math:`\beta` was introduced in the
    `"Semi-Supervised Learning with Heterophily"
    <https://arxiv.org/abs/1412.3100>`_ paper.
    
    Since it is unclear which propagation matrix to use, it is usually
    adviced to concatenate different representations given by
    different propagation matrices, i.e. different choices of 
    :math:`p`, :math:`q`, :math:`r`, as done in the
    `"SIGN: Scalable Inception Graph Neural Networks"
    <https://arxiv.org/abs/2004.11198>`_ paper. The present implementation
    extends this possibility to different :math:`\alpha` and :math:`\beta`.
    
    .. note::

        Since intermediate node representations are pre-computed, this operator
        is able to scale well to large graphs via classic mini-batching.
        For an example of using SIGN, see `examples/sign.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        sign.py>`_.
    
    .. warning::
        
        If :math:`r \ne 0`, the diagonal elements of :math:`\mathbf{A}`
        are overwritten and all set to :math:`r`.
        
    Args:
        K (int): The number of hops/layer.
        p (float): Parameter :math:`p` in the formula above. (default: :obj:`0`)
        q (float): Parameter :math:`q` in the formula above. (default: :obj:`0`)
        r (float): Parameter :math:`r` in the formula above. (default: :obj:`0`)
        alpha (float): Parameter :math:`\alpha` in the formula above. (default: :obj:`0.5`)
        beta (float): Parameter :math:`\beta` in the formula above. (default: :obj:`0.5`)
        
    """
    def __init__(self, K, p=0., q=0., r=0., alpha=0.5, beta=0.5):
        self.K = K
        self.p = p
        self.q = q
        self.r = r
        self.alpha = alpha
        self.beta = beta

    def __call__(self, data):
        assert data.edge_index is not None
        row, col = data.edge_index
        adj_t = SparseTensor(row=row, col=col,
                             sparse_sizes=(data.num_nodes, data.num_nodes))
        deg = adj_t.sum(dim=1).to(torch.float)
        if self.r != 0:
            adj_t = adj_t.fill_value(self.r)
            adj_t = SparseTensor.fill_diag(adj_t, self.r)
        deg_l = (1-self.p)*deg + self.p 
        deg_inv_sqrt_l = deg_l.pow(-self.alpha)
        deg_inv_sqrt_l[deg_inv_sqrt_l == float('inf')] = 0
        deg_r = (1-self.q)*deg + self.q 
        deg_inv_sqrt_r = deg_r.pow(-self.beta)
        deg_inv_sqrt_r[deg_inv_sqrt_r == float('inf')] = 0
        adj_t = deg_inv_sqrt_l.view(-1, 1) * adj_t * deg_inv_sqrt_r.view(1, -1)
        
        assert data.x is not None
        xs = data.x
        for i in range(1, self.K + 1):
            xs = adj_t @ xs
            data[f'x{i}'] = xs

        return data

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
