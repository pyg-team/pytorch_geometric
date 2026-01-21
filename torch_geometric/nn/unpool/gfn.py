import torch


class GFNUnpooling(torch.nn.Module):
    r"""The Graph Feedforward Network unpooling layer from
    `"GFN: A graph feedforward network for resolution-invariant
    reduced operator learning in multifidelity applications"
    <https://doi.org/10.1016/j.cma.2024.117458>`_.

    The GFN unpooling equation is given by:

        .. math::
            :nowrap:

            \begin{equation*}
            \begin{aligned}
            \tilde{W}_{i_{\mathcal{M}_{n}}j} &= \underset{\forall
            k_{\mathcal{M}_{o}} \text{ s.t } k_{\mathcal{M}_{o}}
            {\leftarrow}\!{\backslash}\!{\rightarrow}
            i_{\mathcal{M}_{n}}}{\operatorname{mean}}
            {W}_{k_{\mathcal{M}_{o}}j}, \\
            \tilde{b}_{i_{\mathcal{M}_{n}}} &= \underset{\forall
            k_{\mathcal{M}_{o}} \text{ s.t } k_{\mathcal{M}_{o}}
            {\leftarrow}\!{\backslash}\!{\rightarrow}
            i_{\mathcal{M}_{n}}}{\operatorname{mean}}
            {b}_{k_{\mathcal{M}_{o}}}.
            \end{aligned}
            \end{equation*}

        where:

        - :math:`\mathcal{M}_{o}` is the original output graph,
        - :math:`\mathcal{M}_{n}` is the new output graph,
        - :math:`W` and :math:`b` are the weights and biases
          associated to the original graph,
        - :math:`\tilde{W}` and :math:`\tilde{b}` are the new
          weights and biases associated to the new graph,
        - :math:`i_{\mathcal{M}_o} {\leftarrow}\!{\backslash}\!
          {\rightarrow} j_{\mathcal{M}_n}` indicates that either
          node :math:`i` in graph :math:`\mathcal{M}_o` is the
          nearest neighbor of node :math:`j` in graph
          :math:`\mathcal{M}_n` or vice versa.

    Args:
        in_size (int): Size of the input vector.
        pos_y (torch.tensor): Original output graph (node position matrix)
            :math:`\in \mathbb{R}^{M^{\prime} \times d}`.
        **kwargs (optional): Additional arguments of :class:`gfn.GFN`.
    """
    def __init__(self, in_size: int, pos_y: torch.Tensor, **kwargs):
        try:
            import gfn  # noqa
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "GFNUnpooling requires `gfn` to be installed. "
                "Please install it via `pip install gfn-layer`.") from e
        super().__init__()
        self._gfn = gfn.GFN(in_features=in_size, out_features=pos_y, **kwargs)

    def forward(self, x: torch.Tensor, pos_y: torch.Tensor,
                batch_x: torch.Tensor = None, batch_y: torch.Tensor = None):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            pos_y (torch.Tensor): New output graph (new node position matrix)
                :math:`\in \mathbb{R}^{M \times d}`.
            batch_x (torch.Tensor, optional): Batch vector
                :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^{N}`, which
                assigns each node from :math:`\mathbf{X}` to a specific
                example. (default: :obj:`None`)
            batch_y (torch.Tensor, optional): Batch vector
                :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^{M}`, which
                assigns each node from :math:`\mathbf{Y}` to a specific
                example. (default: :obj:`None`)

        :rtype: :class:`torch.Tensor`
        """
        out = torch.empty((batch_y.shape[0]), *x.shape[1:],
                          dtype=self._gfn.weight.dtype,
                          device=self._gfn.weight.device)
        for batch_label in batch_x.unique():
            mask = batch_y == batch_label
            pos = pos_y[mask, ...]
            x_ = x[batch_x == batch_label, ...]
            out[mask] = self._gfn(x_.T, out_graph=pos).T
        return out
