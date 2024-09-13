from typing import Tuple

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor


class BipartitePooling(torch.nn.Module):
    r"""The bipartite pooling operator from the `"DeepTreeGANv2: Iterative
    Pooling of Point Clouds" <https://arxiv.org/abs/2312.00042>`_ paper.

    The Pooling layer constructs a dense bipartite graph between the input
    nodes and the "Seed" nodes that are trainable parameters of the layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (int): Number of seed nodes.
        gnn (torch.nn.Module): A graph neural network layer that
                implements the bipartite messages passing methode, such as
                :class:`torch_geometric.nn.conv.GATv2Conv`,
                :class:`torch_geometric.nn.conv.GATConv`,
                :class:`torch_geometric.nn.conv.GINConv`,
                :class:`torch_geometric.nn.conv.GeneralConv`,
                :class:`torch_geometric.nn.conv.GraphConv`,
                :class:`torch_geometric.nn.conv.MFConv`,
                :class:`torch_geometric.nn.conv.SAGEConv`,
                :class:`torch_geometric.nn.conv.WLConvContinuous`.
                (Recommended: :class:`torch_geometric.nn.conv.GATv2Conv`
                with `add_self_loops=False`.)

    Shapes:
        - **inputs:**
                node features :math:`(|\mathcal{V}|, F_{in})`,
                batch :math:`(|\mathcal{V}|)`
        - **outputs:**
            node features (`ratio`, :math:`F_{out}`), batch (`ratio`,)
    """
    def __init__(
        self,
        in_channels: int,
        ratio: int,
        gnn: torch.nn.Module,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio

        self.seed_nodes = torch.nn.Parameter(
            torch.empty(size=(self.ratio, self.in_channels)))
        self.gnn = gnn

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()
        self.seed_nodes.data.normal_()

    def forward(
        self,
        x: Tensor,
        batch: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.

            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
        """
        if batch is None:
            batch = torch.zeros((x.size(0)), dtype=torch.long).to(x.device)
        batch_size = batch.max() + 1

        x_aggrs = self.seed_nodes.repeat(batch_size, 1)

        source_graph_size = len(x)

        source = torch.arange(source_graph_size, device=x.device,
                              dtype=torch.long).repeat_interleave(self.ratio)

        target = torch.arange(self.ratio, device=x.device,
                              dtype=torch.long).repeat(source_graph_size)
        target += batch.repeat_interleave(self.ratio) * self.ratio

        out = self.gnn(
            x=(x, x_aggrs),
            edge_index=torch.vstack([source, target]),
            # size=(len(x), self.ratio * int(batch_size)),
        )

        new_batchidx = torch.arange(batch_size, dtype=torch.long,
                                    device=x.device).repeat_interleave(
                                        self.ratio)

        return (out, new_batchidx)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.gnn.__class__.__name__}, '
                f'{self.in_channels}, {self.ratio})')
