import torch_scatter


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (:obj:`"add"`, :obj:`"mean"`,
    :obj:`"max"`).

    Args:
        name (string): The aggregation to use (one of :obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_geometric.utils import scatter_
        src = torch.Tensor([2, 3, -2, 1, 1])
        index = torch.tensor([0, 1, 0, 1, 2])
        output = scatter_("add", src, index)
        print(output)

    .. testoutput::

       tensor([0., 4., 1.])
    """

    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name is 'max' else 0

    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name is 'max':
        out[out == fill_value] = 0

    return out
