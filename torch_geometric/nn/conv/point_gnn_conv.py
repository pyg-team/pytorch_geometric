import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from ..inits import reset


class PointGNNConv(MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper

    .. math::

        \Delta \textrm{pos}_i &= h_{\mathbf{\Theta}}(\mathbf{x}_i)

        \mathbf{e}_{j,i} &= f_{\mathbf{\Theta}}(\textrm{pos}_j -
        \textrm{pos}_i + \Delta \textrm{pos}_i, \mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= g_{\mathbf{\Theta}}(\max_{j \in
        \mathcal{N}(i)} \mathbf{e}_{j,i}) + \mathbf{x}_i

    The relative position is used in the message passing step to introduce
    global translation invariance.
    To also counter shifts in the local neighborhood of the center node, the
    authors propose to utilize an alignment offset.
    The graph should be statically constructed using radius-based cutoff.

    Args:
        mlp_h (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}`
            that maps node features of size :math:`F_{in}` to three-dimensional
            coordination offsets :math:`\Delta \textrm{pos}_i`.
        mlp_f (torch.nn.Module): A neural network :math:`f_{\mathbf{\Theta}}`
            that computes :math:`\mathbf{e}_{j,i}` from the features of
            neighbors of size :math:`F_{in}` and the three-dimensional vector
            :math:`\textrm{pos_j} - \textrm{pos_i} + \Delta \textrm{pos}_i`.
        mlp_g (torch.nn.Module): A neural network :math:`g_{\mathbf{\Theta}}`
            that maps the aggregated edge features back to :math:`F_{in}`.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          positions :math:`(|\mathcal{V}|, 3)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
        - **output:** node features :math:`(|\mathcal{V}|, F_{in})`
    """
    def __init__(
        self,
        mlp_h: torch.nn.Module,
        mlp_f: torch.nn.Module,
        mlp_g: torch.nn.Module,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.mlp_h = mlp_h
        self.mlp_f = mlp_f
        self.mlp_g = mlp_g

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_f)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        """"""
        # propagate_type: (x: Tensor, pos: Tensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)
        out = self.mlp_g(out)
        return x + out

    def message(self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor,
                x_j: Tensor) -> Tensor:
        delta = self.mlp_h(x_i)
        e = torch.cat([pos_j - pos_i + delta, x_j], dim=-1)
        return self.mlp_f(e)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  mlp_h={self.mlp_h},\n'
                f'  mlp_f={self.mlp_f},\n'
                f'  mlp_g={self.mlp_g},\n'
                f')')
