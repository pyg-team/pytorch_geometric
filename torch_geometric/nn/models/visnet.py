import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad

from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import scatter


class CosineCutoff(nn.Module):
    r"""Appies a cosine cutoff to the input distances.

    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}

    Args:
        cutoff (float): A scalar that determines the point
            at which the cutoff is applied.
    """

    def __init__(self, cutoff: float):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        r"""Applies a cosine cutoff to the input distances.

        Args:
            distances (torch.Tensor): A tensor of distances.

        Returns:
            cutoffs (torch.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    r"""Applies exponential normal smearing to the input distances.

    .. math::
        \text{smeared\_dist} = \text{CosineCutoff}(\text{dist})
        * e^{-\beta * (e^{\alpha * (-\text{dist})} - \text{means})^2}

    Args:
        cutoff (float): A scalar that determines the point
            at which the cutoff is applied.
        num_rbf (int): The number of radial basis functions.
        trainable (bool): If True, the means and betas of the RBFs
            are trainable parameters.
    """

    def __init__(
        self, cutoff: float = 5.0, num_rbf: int = 128, trainable: bool = True
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        r"""Initializes the means and betas
        for the radial basis functions.

        Returns:
            means, betas (Tuple[torch.Tensor, torch.Tensor]): The
                initialized means and betas.
        """
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        r"""Resets the means and betas to their initial values."""
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        r"""Applies the exponential normal smearing
        to the input distance.

        Args:
            dist (torch.Tensor): A tensor of distances.

        Returns:
            smeared_dist (torch.Tensor): The smeared distances.
        """
        dist = dist.unsqueeze(-1)
        smeared_dist = self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )
        return smeared_dist


class Sphere(nn.Module):
    r"""Computes spherical harmonics of the input data.

    This module computes the spherical harmonics up
    to a given degree `lmax` for the input tensor of 3D vectors.
    The vectors are assumed to be given in Cartesian coordinates.
    See `Wikipedia
    <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics>`_
    for mathematical details.

    Args:
        lmax (int): The maximum degree of the spherical harmonics.
    """

    def __init__(self, lmax: int = 2):
        super(Sphere, self).__init__()
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        r"""Computes the spherical harmonics of the input tensor.

        Args:
            edge_vec (torch.Tensor): A tensor of 3D vectors.

        Returns:
            edge_sh (torch.Tensor): The spherical harmonics
                of the input tensor.
        """
        edge_sh = self._spherical_harmonics(
            self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        )
        return edge_sh

    @staticmethod
    def _spherical_harmonics(
        lmax: int, x: Tensor, y: Tensor, z: Tensor
    ) -> Tensor:
        r"""Computes the spherical harmonics
        up to degree `lmax` of the input vectors.

        Args:
            lmax (int): The maximum degree of the spherical harmonics.
            x (torch.Tensor): The x coordinates of the vectors.
            y (torch.Tensor): The y coordinates of the vectors.
            z (torch.Tensor): The z coordinates of the vectors.

        Returns:
            sh (torch.Tensor): The spherical harmonics of the input vectors.
        """
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                ],
                dim=-1,
            )


class VecLayerNorm(nn.Module):
    r"""Applies layer normalization to the input data.

    This module applies a custom layer normalization to a tensor of vectors.
    The normalization can either be "max_min" normalization,
    or no normalization.

    Args:
        hidden_channels (int): The number of hidden channels in the input.
        trainable (bool): If True, the normalization weights
            are trainable parameters.
        norm_type (str): The type of normalization to apply.
            Can be "max_min" or "none".
    """

    def __init__(
        self, hidden_channels: int, trainable: bool, norm_type: str = "max_min"
    ):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the normalization weights to their initial values."""
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec: Tensor) -> Tensor:
        r"""Applies no normalization to the input tensor.

        Args:
            vec (torch.Tensor): The input tensor.

        Returns:
            vec (torch.Tensor): The same input tensor.
        """
        return vec

    def max_min_norm(self, vec: Tensor) -> Tensor:
        r"""Applies max-min normalization to the input tensor.

        .. math::
            \text{dist} = ||\text{vec}||_2
            \text{direct} = \frac{\text{vec}}{\text{dist}}
            \text{max\_val} = \max(\text{dist})
            \text{min\_val} = \min(\text{dist})
            \text{delta} = \text{max\_val} - \text{min\_val}
            \text{dist} = \frac{\text{dist} - \text{min\_val}}{\text{delta}}
            \text{normed\_vec} = \max(0, \text{dist}) \cdot \text{direct}

        Args:
            vec (torch.Tensor): The input tensor.

        Returns:
            normed_vec (torch.Tensor): The normalized tensor.
        """
        dist = torch.norm(vec, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, vec: Tensor) -> Tensor:
        r"""Applies the layer normalization to the input tensor.

        Args:
            vec (torch.Tensor): The input tensor.

        Returns:
            normed_vec (torch.Tensor): The normalized tensor.
        """
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class Distance(nn.Module):
    r"""Computes the pairwise distances between atoms in a molecule.

    This module computes the pairwise distances between atoms in a molecule,
    represented by their positions `pos`.
    The distances are computed only between points
    that are within a certain cutoff radius.

    Args:
        cutoff (float): The cutoff radius beyond
            which distances are not computed.
        max_num_neighbors (int): The maximum number of neighbors
            considered for each point.
        loop (bool): Whether self-loops are included.
    """

    def __init__(
        self, cutoff: float, max_num_neighbors: int = 32, loop: bool = True
    ):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(
        self, pos: Tensor, batch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes the pairwise distances between atoms in the molecule.

        Args:
            pos (torch.Tensor): The positions of the atoms
                in the molecule.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            edge_index (torch.Tensor): The indices of the edges
                in the graph.
            edge_weight (torch.Tensor): The distances
                between connected nodes.
            edge_vec (torch.Tensor): The vector differences
                between connected nodes.
        """
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    r"""The `NeighborEmbedding` module from the
    `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        num_rbf (int): The number of radial basis functions.
        cutoff (float): The cutoff distance.
        max_z (int): The maximum atomic numbers.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        max_z: int = 100,
    ):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        r"""Computes the neighborhood embedding of the nodes in the graph.

        Args:
            z (torch.Tensor): The atomic numbers.
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The indices of the edges.
            edge_weight (torch.Tensor): The weights of the edges.
            edge_attr (torch.Tensor): The edge features.

        Returns:
            x_neighbors (torch.Tensor): The neighborhood embeddings
                of the nodes.
        """
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class EdgeEmbedding(MessagePassing):
    r"""The `EdgeEmbedding` module
    from the `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        num_rbf (int):
            The number of radial basis functions.
        hidden_channels (int):
            The number of hidden channels in the node embeddings.
    """

    def __init__(self, num_rbf: int, hidden_channels: int):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)

    def forward(
        self, edge_index: Tensor, edge_attr: Tensor, x: Tensor
    ) -> Tensor:
        r"""Computes the edge embeddings of the graph.

        Args:
            edge_index (torch.Tensor): The indices of the edges.
            edge_attr (torch.Tensor): The edge features.
            x (torch.Tensor): The node features.

        Returns:
            out_edge_attr (torch.Tensor): The edge embeddings.
        """
        out_edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out_edge_attr

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return (x_i + x_j) * self.edge_proj(edge_attr)

    def aggregate(self, features: Tensor, index: Tensor) -> Tensor:
        return features


class ViS_MP(MessagePassing):
    r"""The message passing module without vertex geometric features
    of the equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        num_heads (int): The number of attention heads.
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        cutoff (float): The cutoff distance.
        vecnorm_type (str): The type of normalization
            to apply to the vectors.
        trainable_vecnorm (bool): Whether the normalization weights
            are trainable.
        last_layer (bool): Whether this is the last layer
            in the model.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: str,
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ):
        super(ViS_MP, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False
        )

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False
            )
            self.w_trg_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False
            )

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor):
        r"""Computes the component of 'vec' orthogonal to 'd_ij'.

        Args:
            vec (torch.Tensor): The input vector.
            d_ij (torch.Tensor): The reference vector.

        Returns:
            vec_rej (torch.Tensor): The component of 'vec'
                orthogonal to 'd_ij'.
        """
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)

        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Computes the residual scalar and vector features
        of the nodes and scalar featues of the edges.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            vec (torch.Tensor):The vector features of the nodes.
            edge_index (torch.Tensor): The indices of the edges.
            r_ij (torch.Tensor): The distances between connected nodes.
            f_ij (torch.Tensor): The scalar features of the edges.
            d_ij (torch.Tensor): The unit vectors of the edges

        Returns:
            dx (torch.Tensor): The residual scalar features
                of the nodes.
            dvec (torch.Tensor): The residual vector features
                of the nodes.
            df_ij (torch.Tensor, optional): The residual scalar features
                of the edges, or None if this is the last layer.
        """
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )

        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1)

        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(
                edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij
            )
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Tensor,
        dv: Tensor,
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(
            self.act(self.s_proj(v_j)), self.hidden_channels, dim=1
        )
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(
        self,
        vec_i: Tensor,
        vec_j: Tensor,
        d_ij: Tensor,
        f_ij: Tensor,
    ) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class ViS_MP_Vertex(MessagePassing):
    r"""The message passing module with vertex geometric features
    of the equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        num_heads (int): The number of attention heads.
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        cutoff (float): The cutoff distance.
        vecnorm_type (str): The type of normalization
            to apply to the vectors.
        trainable_vecnorm (bool): Whether the normalization weights
            are trainable.
        last_layer (bool): Whether this is the last layer
            in the model.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: str,
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ):
        super(ViS_MP_Vertex, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False
        )

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.w_src_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False
            )
            self.w_trg_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False
            )
            self.t_src_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False
            )
            self.t_trg_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False
            )

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor):
        r"""Computes the component of 'vec' orthogonal to 'd_ij'.

        Args:
            vec (torch.Tensor): The input vector.
            d_ij (torch.Tensor): The reference vector.

        Returns:
            vec_rej (torch.Tensor): The component of 'vec'
                orthogonal to 'd_ij'.
        """
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)

        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)
            nn.init.xavier_uniform_(self.t_src_proj.weight)
            nn.init.xavier_uniform_(self.t_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Computes the residual scalar and vector features
        of the nodes and scalar featues of the edges.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            vec (torch.Tensor):The vector features of the nodes.
            edge_index (torch.Tensor): The indices of the edges.
            r_ij (torch.Tensor): The distances between connected nodes.
            f_ij (torch.Tensor): The scalar features of the edges.
            d_ij (torch.Tensor): The unit vectors of the edges

        Returns:
            dx (torch.Tensor): The residual scalar features
                of the nodes.
            dvec (torch.Tensor): The residual vector features
                of the nodes.
            df_ij (torch.Tensor, optional): The residual scalar features
                of the edges, or None if this is the last layer.
        """
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )

        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1)

        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(
                edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij
            )
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Tensor,
        dv: Tensor,
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(
            self.act(self.s_proj(v_j)), self.hidden_channels, dim=1
        )
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(
        self,
        vec_i: Tensor,
        vec_j: Tensor,
        d_ij: Tensor,
        f_ij: Tensor,
    ) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)

        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)

        f1, f2 = torch.split(
            self.act(self.f_proj(f_ij)), self.hidden_channels, dim=-1
        )

        return f1 * w_dot + f2 * t_dot

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class ViSNetBlock(nn.Module):
    r"""The representation module of the equivariant vector-scalar
    interactive graph neural network (ViSNet) from the
    `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        lmax (int): The maximum degree
            of the spherical harmonics.
        vecnorm_type (str): The type of normalization
            to apply to the vectors.
        trainable_vecnorm (bool):  Whether the normalization weights
            are trainable.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the network.
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        num_rbf (int): The number of radial basis functions.
        trainable_rbf (bool):Whether the radial basis function
            parameters are trainable.
        max_z (int): The maximum atomic numbers.
        cutoff (float): The cutoff distance.
        max_num_neighbors (int):The maximum number of neighbors
            considered for each atom.
        vertex (bool): Whether to use vertex geometric features.
    """

    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: str = "none",
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 9,
        hidden_channels: int = 256,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
    ):
        super(ViSNetBlock, self).__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance = Distance(
            cutoff, max_num_neighbors=max_num_neighbors, loop=True
        )
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(
            cutoff, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff, max_z
        )
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
        )
        vis_mp_class = ViS_MP if not vertex else ViS_MP_Vertex
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(
            vis_mp_class(last_layer=True, **vis_mp_kwargs)
        )

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(self, z, pos, batch):
        r"""Computes the scalar and vector features of the nodes.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            x (torch.Tensor): The scalar features of the nodes.
            vec (torch.Tensor): The vector features of the nodes.
        """
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(
            edge_vec[mask], dim=1
        ).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(
            x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), device=x.device
        )
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec
            )
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](
            x, vec, edge_index, edge_weight, edge_attr, edge_vec
        )
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec


class GatedEquivariantBlock(nn.Module):
    r"""Applies a gated equivariant operation
    to scalar features and vector features from the
    `"Equivariant message passing for the prediction
    of tensorial properties and molecular spectra"
    <https://arxiv.org/abs/2102.03150>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        out_channels (int): The number of output channels.
        intermediate_channels (int or None): The number of channels
            in the intermediate layer,
            or None to use the same number as 'hidden_channels'.
        scalar_activation (bool): Whether to apply
            a scalar activation function to the output node features.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Applies a gated equivariant operation
        to node features and vector features.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.

        Returns:
            x (torch.Tensor): The updated scalar features of the nodes.
            v (torch.Tensor): The updated vector features of the nodes.
        """
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantScalar(nn.Module):
    r"""Computes final scalar outputs based on
    node features and vector features.

    Args:
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
    """

    def __init__(self, hidden_channels: int):
        super(EquivariantScalar, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    1,
                    scalar_activation=False,
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tensor:
        r"""Computes the final scalar outputs.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.

        Returns:
            out (torch.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0


class Atomref(nn.Module):
    r"""Adds atom reference values to atomic energies.

    Args:
        atomref (torch.Tensor, optional):  A tensor of atom reference values,
            or None if not provided.
        max_z (int): The maximum atomic numbers.
    """

    def __init__(self, atomref: Optional[Tensor] = None, max_z: int = 100):
        super(Atomref, self).__init__()
        if atomref is None:
            atomref = torch.zeros(max_z, 1)
        else:
            atomref = torch.as_tensor(atomref)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(len(atomref), 1)
        self.atomref.weight.data.copy_(atomref)

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        r"""Adds atom reference values to atomic energies.

        Args:
            x (torch.Tensor): The atomic energies.
            z (torch.Tensor): The atomic numbers.

        Returns:
            x (torch.Tensor): The updated atomic energies.
        """
        return x + self.atomref(z)


class ViSNet(nn.Module):
    r"""A PyTorch module that implements
    the equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing geometric representations for molecules
    with equivariant vector-scalar interactive message passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        lmax (int): The maximum degree
            of the spherical harmonics.
        vecnorm_type (str): The type of normalization
            to apply to the vectors.
        trainable_vecnorm (bool): Whether the normalization weights
            are trainable.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of layers in the network.
        hidden_channels (int): The number of hidden channels
            in the node embeddings.
        num_rbf (int): The number of radial basis functions.
        trainable_rbf (bool): Whether the radial basis function
            parameters are trainable.
        max_z (int): The maximum atomic numbers.
        cutoff (float): The cutoff distance.
        max_num_neighbors (int): The maximum number of neighbors
            considered for each atom.
        vertex (bool): Whether to use vertex geometric features.
        atomref (torch.Tensor, optional): A tensor of atom reference values,
            or None if not provided.
        reduce_op (str): The type of reduction operation to apply
            ("add", "mean").
        mean (float, optional): The mean of the output distribution,
            or 0 if not provided.
        std (float, optional): The standard deviation
            of the output distribution, or 1 if not provided.
        derivative (bool): Whether to compute the derivative of the output
            with respect to the positions.
    """

    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: str = "none",
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        atomref: Optional[Tensor] = None,
        reduce_op: str = "add",
        mean: Optional[float] = None,
        std: Optional[float] = None,
        derivative: bool = False,
    ):
        super(ViSNet, self).__init__()

        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )

        self.output_model = EquivariantScalar(hidden_channels=hidden_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = (
            torch.scalar_tensor(0)
            if mean is None
            else torch.scalar_tensor(mean)
        )
        self.register_buffer("mean", mean)
        std = (
            torch.scalar_tensor(1) if std is None else torch.scalar_tensor(std)
        )
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces)
        for a batch of molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            y (torch.Tensor): The energies or properties for each molecule.
            dy (torch.Tensor, optional): The negative derivative of energies.
        """
        if self.derivative:
            pos.requires_grad_(True)

        x, v = self.representation_model(z, pos, batch)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean

        if self.derivative:
            grad_outputs = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction."
                )
            return y, -dy
        return y, None
