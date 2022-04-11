from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Size, OptPairTensor


class DisenConv(MessagePassing):
    r"""The graph convolutional operator from the `"Disentangled graph
    convolutional networks"
    <http://proceedings.mlr.press/v97/ma19a.html>`_ paper。
    Given the feature vectors of centarl node :obj:`u` and its neighbors
    :math:`\mathcal{N}(u)`, this layer outputs the central node feature
    vector disentangled into :obj:`K` factors:

    .. math::
        \mathbf{y}_{u}=\left[\mathbf{c}_{1}, \mathbf{c}_{2},
        \ldots, \mathbf{c}_{K}\right].

    DisenConv scans the input feature vectors from :obj:`K`
    perspectives:

    .. math::
        \mathbf{z}_{i, k} \leftarrow \sigma\left(\mathbf{W}_{k}^{\top}
        \mathbf{x}_{i}+\mathbf{b}_{k}\right), i \in\{u\} \cup \mathcal{N}(u)

    where :math:`\mathbf{W}_{k}` and :math:`\mathbf{b}_{k}` project the
    features :math:`\mathbf{x}_{i}` of node :math:`i` into the k-th
    aspect.

    The :math:`\mathbf{c}_{k}` is initialized with :math:`\mathbf{z}_{u, k}`
    Later, the following process is iterated for :obj:`T` times.

    First, the propagation weight from :math:`v\in \mathcal{N}(u)`
    is computed via:

    .. math::
        &p_{v, k} \leftarrow \mathbf{z}_{v, k}^{\top}
        \mathbf{c}_{k} / \tau, \forall k=1,2, \ldots, K

        &{\left[p_{v, 1} \ldots p_{v, K}\right] \leftarrow
        \operatorname{softmax}\left(\left[p_{v, 1}
        \ldots p_{v, K}\right]\right)}

    Then, the message passing is conducted for :obj:`K` factors:

    .. math::
        &\mathbf{c}_{k} \leftarrow \mathbf{Z}_{u, k}+
        \sum_{v\in\mathcal{N}(u)} p_{v, k} \mathbf{Z}_{v, k}

        &\mathbf{c}_{k} \leftarrow \mathbf{c}_{k} / \left\|
        \mathbf{c}_{k}\right\|_{2} \forall k=1,2, \ldots, K .


    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of channels of the output feature.
        K (int): The number of channels in the output.
        T (int): The number of iterations. Defaults to 5
        tao (float): The time decay parameter.
        separate_channels (List[int]): This is a list of integers, which
            specifies the number of channels for each factor.
            If set to :obj:`None`, the channels will be automatically
            split as even as possible.

    Shapes:
        - **input:**
            node features :math:`(|\mathcal{V}|, F_{in})`
            if you want the embeddings of all nodes in :math:`\mathcal{V}` ;
            :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
            if you only want the embeddings of the target nodes in
            :math:`\mathcal{V_t}`
            edge indices :math:`(2, |\mathcal{E}|)`,
        - **output:**
            node features :math:`(|\mathcal{V}|, F_{out})`
            if you want the embeddings of all nodes in :math:`\mathcal{V}` ;
            :math:`(|\mathcal{V_t}|, F_{out})`
            if you only want the embeddings of the target nodes in
            :math:`\mathcal{V_t}` .

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            K: int = None,
            T: int = 5,
            tao: float = 1.0,
            separate_channels: List[int] = None,
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
                    "You have to choose a `K` when `separate_channels` is None"
                )
            self.separate_channels = [out_channels // K] * K
            self.separate_channels[-1] += out_channels % K
            self.K = K
        else:
            assert torch.all(torch.tensor(separate_channels) > 0)
            assert sum(separate_channels) == out_channels
            self.separate_channels = separate_channels
            self.K = len(separate_channels)

        self.lin_projector = Linear(in_channels, out_channels, bias=True)

    def reset_parameters(self):
        self.lin_projector.reset_parameters()

    def get_projector(
            self, separate: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tuple[Tensor], Tuple[Tensor]]]:
        projector = self.lin_projector.weight
        projector_bias = self.lin_projector.bias
        if separate:
            projector = torch.split(projector, self.separate_channels, dim=0)
            projector_bias = torch.split(projector_bias,
                                         self.separate_channels, dim=0)
        return projector, projector_bias

    def project2aspects(self, features: Tensor) -> List[Tensor]:
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
        pout = [
            torch.sum(cs[k][dst_nodes] *
                      zs[k][src_nodes], dim=-1).div_(self.tao)
            for k in range(self.K)
        ]
        pout = F.softmax(torch.stack(pout), dim=0)
        return pout

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Union[SparseTensor, Tensor],
            size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(edge_index, Tensor):
            src_nodes, dst_nodes = edge_index
        elif isinstance(edge_index, SparseTensor):
            dst_nodes, src_nodes, _ = edge_index.coo()
        else:
            raise TypeError(f"{type(edge_index)} not a valid graph type")

        if isinstance(x, Tensor):
            x_src, x_dst = x, x
        else:  # x_src includes the features of central and neighboring nodes
            x_src, x_dst = x[0], x[1]

        z_src = self.project2aspects(x_src)
        if x_dst is not None:
            z_dst = self.project2aspects(x_dst)
        else:
            unique_dst_nodes = torch.unique(dst_nodes, sorted=True)
            z_dst = [chunk[unique_dst_nodes] for chunk in z_src]

        c_dst = [ck.clone() for ck in z_dst]
        size = (z_src[0].shape[0], z_dst[0].shape[0]) if size is None else size

        for i in range(self.T):
            p = self.update_p(z_src, c_dst, dst_nodes, src_nodes)
            for k in range(self.K):
                graph = (
                    edge_index.set_value(p[k], layout="coo")
                    if isinstance(edge_index, SparseTensor)
                    else edge_index
                )
                ck = self.propagate(
                    graph, x=(z_src[k], None), edge_weight=p[k], size=size
                )
                ck = ck + z_dst[k]
                c_dst[k] = ck / ck.norm(dim=-1, keepdim=True)

        out = torch.cat(c_dst, dim=1)
        return out

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, "
            f"[{self.K}]disentangled_channels={self.separate_channels})"
        )
