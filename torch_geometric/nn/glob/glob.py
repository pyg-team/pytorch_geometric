from typing import Optional

from torch_scatter import scatter


def global_add_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def global_mean_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def global_max_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')
