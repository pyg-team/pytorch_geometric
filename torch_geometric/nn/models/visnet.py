import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter

from torch_geometric.nn import MessagePassing, radius_graph


class CosineCutoff(nn.Module):

    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
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

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class Sphere(nn.Module):

    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.lmax = lmax

    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(
            self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh

    @staticmethod
    def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

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
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec):
        return vec

    def rms_norm(self, vec):
        dist = torch.norm(vec, dim=1)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)

    def max_min_norm(self, vec):
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

    def forward(self, vec):
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
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  loop=self.loop, max_num_neighbors=self.max_num_neighbors)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
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

    def message(self, x_j, W):
        return x_j * W


class EdgeEmbedding(MessagePassing):

    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)

    def forward(self, edge_index, edge_attr, x):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_proj(edge_attr)

    def aggregate(self, features, index):
        return features


class ViS_MP(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
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
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)

        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
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

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1,
                                                  self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1,
                                                  self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1)
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
                edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)),
                             self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class ViS_MP_Vertex(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
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
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)

        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.w_src_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False)
            self.t_src_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False)
            self.t_trg_proj = nn.Linear(
                hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
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

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1,
                                                  self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1,
                                                  self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1)
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
                edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)),
                             self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(self, vec_i, vec_j, d_ij, f_ij):

        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)

        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)

        f1, f2 = torch.split(self.act(self.f_proj(f_ij)),
                             self.hidden_channels, dim=-1)

        return f1 * w_dot + f2 * t_dot

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class ViSNetBlock(nn.Module):

    def __init__(
        self,
        lmax=1,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=9,
        hidden_channels=256,
        num_rbf=32,
        trainable_rbf=False,
        max_z=100,
        cutoff=5.0,
        max_num_neighbors=32,
        vertex=False,
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
            cutoff, max_num_neighbors=max_num_neighbors, loop=True)
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(
            cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff, max_z)
        self.edge_embedding = EdgeEmbedding(
            num_rbf, hidden_channels)

        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm
        )
        vis_mp_class = ViS_MP if not vertex else ViS_MP_Vertex
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(
            last_layer=True, **vis_mp_kwargs))

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(self, z, pos, batch):

        # Embedding Layers
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / \
            torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) -
                          1, x.size(1), device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        # ViS-MP Layers
        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec,
                                             edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec


class GatedEquivariantBlock(nn.Module):

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantScalar(nn.Module):
    def __init__(self, hidden_channels):
        super(EquivariantScalar, self).__init__()
        self.output_network = nn.ModuleList([
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
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0


class Atomref(nn.Module):

    def __init__(self, atomref=None, max_z=100):
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
        self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, x, z):
        return x + self.atomref(z)


class ViSNet(nn.Module):
    def __init__(
        self,
        lmax=1,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=6,
        hidden_channels=128,
        num_rbf=32,
        trainable_rbf=False,
        max_z=100,
        cutoff=5.0,
        max_num_neighbors=32,
        vertex=False,
        atomref=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
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

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
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

        if self.derivative:
            pos.requires_grad_(True)

        x, v = self.representation_model(z, pos, batch)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        out = out + self.mean

        if self.derivative:
            grad_outputs = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction.")
            return out, -dy
        return out, None
