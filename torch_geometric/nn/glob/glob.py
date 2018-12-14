from torch_geometric.utils import scatter_


def global_add_pool(x, batch, size=None):
    r"""Returns batch-wise graph-level-outputs by adding node features across
    the node node dimension:

    .. math::
        \mathbf{r}_i = \sum_{n, b_n = i}^N \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific graph.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """

    size = batch[-1].item() + 1 if size is None else size
    return scatter_('add', x, batch, dim_size=size)


def global_mean_pool(x, batch, size=None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension:

    .. math::
        \mathbf{r}_i = \mathrm{mean}_{n, b_n = i}^N \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific graph.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """

    size = batch[-1].item() + 1 if size is None else size
    return scatter_('mean', x, batch, dim_size=size)


def global_max_pool(x, batch, size=None):
    r"""Returns batch-wise graph-level-outputs by taking the maximum node
    features across the node dimension:

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n, b_n = i}^N \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific graph.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """

    size = batch[-1].item() + 1 if size is None else size
    return scatter_('max', x, batch, dim_size=size)
