from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor


class DisenConv(MessagePassing):
    r"""The graph convolutional operator from the `"Disentangled graph convolutional networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
      in_channels (int): The number of input channels.
      out_channels (int): The number of channels of the output feature.
      K (int): The number of channels in the output.
      T (int): The number of iterations. Defaults to 5
      tao (float): The time decay parameter.
      separate_channels (List[int]): This is a list of integers, which specifies
                the number of channels for each factor.
      weight_initializer (Optional[str]): The initializer for the weights of the linear projector.
      bias_initializer (Optional[str]): Initializer for the bias vector.


    Shapes:
        - **input:**
            node features :math:`(|\mathcal{V}|, F_{in})`,
            edge indices :math:`(2, |\mathcal{E}|)`,
            batch_size :math:`B`
        - **output:** node features :math:`(B, F_{out})`
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            K: int = None,
            T: int = 5,
            tao: float = 1.0,
            separate_channels: List[int] = None,
            weight_initializer: Optional[str] = None,
            bias_initializer: Optional[str] = None,
            **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        assert K > 0
        assert T > 0
        assert tao > 0

        self.T = T
        self.tao = tao
        self.in_channels = in_channels
        self.out_channels = out_channels

        if separate_channels is None:
            if K is None:
                raise ValueError(
                    "`K` must be assigned when `separate_channels` is None"
                )
            self.separate_channels = [out_channels // K] * K
            self.separate_channels[-1] += out_channels % K
            self.K = K
        else:
            assert torch.all(torch.tensor(separate_channels) > 0)
            assert sum(separate_channels) == out_channels
            self.separate_channels = separate_channels
            self.K = len(separate_channels)

        self.lin_projector = Linear(
            in_channels,
            out_channels,
            bias=True,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )

    def reset_parameters(self):
        self.lin_projector.reset_parameters()

    def get_projector(
            self, separate=False
    ) -> Tuple[Union[Tuple[Tensor], Tensor], Union[Tuple[Tensor], Tensor]]:
        projector = self.lin_projector.weight
        projector_bias = self.lin_projector.bias
        if separate:
            projector = torch.split(projector, self.separate_channels, dim=0)
            projector_bias = torch.split(projector_bias, self.separate_channels, dim=0)
        return projector, projector_bias

    def project2aspects(self, features: Tensor) -> List[Tensor]:
        # Z (sum_{i=1}^{i=B}|N(ui)|+B,out_channels)
        Z = self.lin_projector(features)
        Z = torch.split(Z, self.separate_channels, dim=-1)  # tuple of tensors
        zs = [ck / ck.norm(dim=1, keepdim=True) for ck in Z]
        return zs

    def update_p(
            self,
            zs: List[Tensor],
            cs: List[Tensor],
            dst_nodes: Union[Tensor, List[int]],
            src_nodes: Union[Tensor, List[int]],
    ) -> Tensor:
        assert len(dst_nodes) == len(src_nodes)
        pout = []
        for k in range(self.K):
            pout.append(torch.sum(cs[k][dst_nodes] * zs[k][src_nodes], dim=-1).div_(self.tao))
        pout = torch.stack(pout)
        pout = F.softmax(pout, dim=0)
        return pout

    def forward(
            self, x: Tensor, edge_index: Union[SparseTensor, Tensor], batch_size: int = None
    ) -> Tensor:

        if isinstance(edge_index, Tensor):
            src_nodes, dst_nodes = edge_index
        elif isinstance(edge_index, SparseTensor):
            dst_nodes, src_nodes, _ = edge_index.coo()
        else:
            raise TypeError(f"{type(edge_index)} not a valid graph type")
        batch_size = batch_size if batch_size is not None else len(torch.unique(dst_nodes))

        Z = self.project2aspects(x)
        # target nodes always placed first
        C = [chunk[:batch_size].clone() for chunk in Z]

        for i in range(self.T):
            p = self.update_p(Z, C, dst_nodes, src_nodes)
            for k in range(self.K):
                graph = (
                    edge_index[:batch_size].set_value(p[k], layout="coo")
                    if isinstance(edge_index, SparseTensor)
                    else edge_index
                )
                ck = (
                        self.propagate(graph, x=Z[k], edge_weight=p[k])[:batch_size]
                        + Z[k][:batch_size]
                )
                C[k] = ck / ck.norm(dim=-1, keepdim=True)

        out = torch.cat([C[k] for k in range(self.K)], dim=1)
        return out

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # noqa
        return matmul(adj_t, x, reduce=self.aggr)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:  # noqa
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, [{self.K}]disentangled_channels={self.separate_channels})"
        )
