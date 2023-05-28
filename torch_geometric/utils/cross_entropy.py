from torch import Tensor


def sparse_cross_entropy(inputs: Tensor, edge_label_index: Tensor) -> Tensor:
    r"""A sparse-label variant of :func:`torch.nn.functional.cross_entropy`.
    In particular, the binary target matrix is solely given by sparse indices
    :obj:`edge_label_index`.

    Args:
        inputs (torch.Tensor): The predicted unnormalized logits of shape
            :obj:`[batch_size, num_classes]`.
        edge_index (torch.Tensor): The sparse ground-truth indices of
            shape :obj:`[2, num_labels]`.

    :rtype: :class:`torch.Tensor`

    Example:

        >>> inputs = torch.randn(2, 3)
        >>> edge_label_index = torch.tensor([[0, 0, 1],
        ...                                  [0, 1, 2]])
        >>> sparse_cross_entropy(inputs, edge_label_index)
        tensor(1.2919)
    """
    assert inputs.dim() == 2
    logsumexp = inputs.logsumexp(dim=-1)
    values = inputs[edge_label_index[0], edge_label_index[1]]
    out = -values + logsumexp[edge_label_index[0]]
    return out.sum() / inputs.size(0)
