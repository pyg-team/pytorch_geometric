import torch
from torch_geometric.utils import to_batch


def global_sort_pool(x, batch, k):
    r"""The global pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are first sorted individually and then  sorted in
    descending order based on their last features. The first :math:`k` nodes
    form the output of the layer.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        k (int): The number of nodes to hold for each graph.

    :rtype: :class:`Tensor`
    """
    x, _ = x.sort(dim=-1)

    fill_value = x.min().item() - 1
    batch_x, num_nodes = to_batch(x, batch, fill_value)
    B, N, D = batch_x.size()

    _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)

    batch_x = batch_x[:, :k].contiguous()
    batch_x[batch_x == fill_value] = 0
    x = batch_x.view(B, k * D)

    return x
