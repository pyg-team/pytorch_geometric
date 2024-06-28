import math
from functools import partial
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhd
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_sparse.matmul import spmm_add, spmm_max, spmm_mean

from torch_geometric.nn import GCNConv, LGConv
from torch_geometric.nn.models import MLP
from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph
from torch_geometric.utils import to_edge_index

USE_CUSTOM_MLP = True

########################
###### NodeLabel #######
########################

MINIMUM_SIGNATURE_DIM = 64


class NodeLabel(torch.nn.Module):
    def __init__(self, dim: int = 1024, signature_sampling="torchhd",
                 minimum_degree_onehot: int = -1):
        super().__init__()
        self.dim = dim
        self.signature_sampling = signature_sampling
        self.cached_two_hop_adj = None
        self.minimum_degree_onehot = minimum_degree_onehot

    def forward(self, edges: Tensor, adj_t: SparseTensor,
                node_weight: Tensor = None, cache_mode=None):
        if cache_mode is not None:
            return self.propagation_only_cache(edges, adj_t, node_weight,
                                               cache_mode)
        else:
            return self.propagation_only(edges, adj_t, node_weight)

    def get_random_node_vectors(self, adj_t: SparseTensor,
                                node_weight) -> Tensor:
        num_nodes = adj_t.size(0)
        device = adj_t.device()
        if self.minimum_degree_onehot > 0:
            degree = adj_t.sum(dim=1)
            nodes_to_one_hot = degree >= self.minimum_degree_onehot
            one_hot_dim = nodes_to_one_hot.sum()
            if one_hot_dim + MINIMUM_SIGNATURE_DIM > self.dim:
                raise ValueError(
                    f"There are {int(one_hot_dim)} nodes with degree higher than {self.minimum_degree_onehot}, select a higher threshold to choose fewer nodes as hub"
                )
            embedding = torch.zeros(num_nodes, self.dim, device=device)
            if one_hot_dim > 0:
                one_hot_embedding = F.one_hot(torch.arange(
                    0, one_hot_dim)).float().to(device)
                embedding[nodes_to_one_hot, :one_hot_dim] = one_hot_embedding
        else:
            embedding = torch.zeros(num_nodes, self.dim, device=device)
            nodes_to_one_hot = torch.zeros(num_nodes, dtype=torch.bool,
                                           device=device)
            one_hot_dim = 0
        rand_dim = self.dim - one_hot_dim

        if self.signature_sampling == "torchhd":
            scale = math.sqrt(1 / rand_dim)
            node_vectors = torchhd.random(num_nodes - one_hot_dim, rand_dim,
                                          device=device)
            node_vectors.mul_(scale)  # make them unit vectors
        elif self.signature_sampling == "gaussian":
            node_vectors = F.normalize(
                torch.nn.init.normal_(
                    torch.empty((num_nodes - one_hot_dim, rand_dim),
                                dtype=torch.float32, device=device)))
        elif self.signature_sampling == "onehot":
            embedding = torch.zeros(num_nodes, num_nodes, device=device)
            node_vectors = F.one_hot(torch.arange(
                0, num_nodes)).float().to(device)

        embedding[~nodes_to_one_hot, one_hot_dim:] = node_vectors

        if node_weight is not None:
            node_weight = node_weight.unsqueeze(
                1
            )  # Note: not sqrt here because it can cause problem for MLP when output is negative
            # thus, it requires the MLP to approximate one more sqrt?
            embedding.mul_(node_weight)
        return embedding

    def propagation_only(self, edges: Tensor, adj_t: SparseTensor,
                         node_weight=None):
        adj_t, new_edges, subset_nodes = subgraph(edges, adj_t, 2)
        node_weight = node_weight[
            subset_nodes] if node_weight is not None else None
        x = self.get_random_node_vectors(adj_t, node_weight=node_weight)
        subset = new_edges.view(-1)  # flatten the target nodes [row, col]

        subset_unique, inverse_indices = torch.unique(subset,
                                                      return_inverse=True)
        one_hop_x_subgraph_nodes = matmul(adj_t, x)
        one_hop_x = one_hop_x_subgraph_nodes[subset]
        two_hop_x = matmul(adj_t[subset_unique],
                           one_hop_x_subgraph_nodes)[inverse_indices]
        degree_one_hop = adj_t.sum(dim=1)

        one_hop_x = one_hop_x.view(2, new_edges.size(1), -1)
        two_hop_x = two_hop_x.view(2, new_edges.size(1), -1)

        count_1_1 = dot_product(one_hop_x[0, :, :], one_hop_x[1, :, :])
        count_1_2 = dot_product(one_hop_x[0, :, :], two_hop_x[1, :, :])
        count_2_1 = dot_product(two_hop_x[0, :, :], one_hop_x[1, :, :])
        count_2_2 = dot_product(
            (two_hop_x[0, :, :] -
             degree_one_hop[new_edges[0]].view(-1, 1) * x[new_edges[0]]),
            (two_hop_x[1, :, :] -
             degree_one_hop[new_edges[1]].view(-1, 1) * x[new_edges[1]]))

        count_self_1_2 = dot_product(one_hop_x[0, :, :], two_hop_x[0, :, :])
        count_self_2_1 = dot_product(one_hop_x[1, :, :], two_hop_x[1, :, :])
        degree_u = degree_one_hop[new_edges[0]]
        degree_v = degree_one_hop[new_edges[1]]
        return count_1_1, count_1_2, count_2_1, count_2_2, count_self_1_2, count_self_2_1, degree_u, degree_v

    def propagation_only_cache(self, edges: Tensor, adj_t: SparseTensor,
                               node_weight=None, cache_mode=None):
        if cache_mode == 'build':
            # get the 2-hop subgraph of the target edges
            x = self.get_random_node_vectors(adj_t, node_weight=node_weight)

            degree_one_hop = adj_t.sum(dim=1)

            one_hop_x = matmul(adj_t, x)
            two_iter_x = matmul(adj_t, one_hop_x)

            # caching
            self.cached_x = x
            self.cached_degree_one_hop = degree_one_hop

            self.cached_one_hop_x = one_hop_x
            self.cached_two_iter_x = two_iter_x
            return
        if cache_mode == 'delete':
            del self.cached_x
            del self.cached_degree_one_hop
            del self.cached_one_hop_x
            del self.cached_two_iter_x
            return
        if cache_mode == 'use':
            # loading
            x = self.cached_x
            degree_one_hop = self.cached_degree_one_hop

            one_hop_x = self.cached_one_hop_x
            two_iter_x = self.cached_two_iter_x
        count_1_1 = dot_product(one_hop_x[edges[0]], one_hop_x[edges[1]])
        count_1_2 = dot_product(one_hop_x[edges[0]], two_iter_x[edges[1]])
        count_2_1 = dot_product(two_iter_x[edges[0]], one_hop_x[edges[1]])
        count_2_2 = dot_product(two_iter_x[edges[0]]-degree_one_hop[edges[0]].view(-1,1)*x[edges[0]],\
                                two_iter_x[edges[1]]-degree_one_hop[edges[1]].view(-1,1)*x[edges[1]])

        count_self_1_2 = dot_product(one_hop_x[edges[0]], two_iter_x[edges[0]])
        count_self_2_1 = dot_product(one_hop_x[edges[1]], two_iter_x[edges[1]])

        degree_u = degree_one_hop[edges[0]]
        degree_v = degree_one_hop[edges[1]]
        return count_1_1, count_1_2, count_2_1, count_2_2, count_self_1_2, count_self_2_1, degree_u, degree_v


def subgraph(edges: Tensor, adj_t: SparseTensor, k: int = 2):
    row, col = edges
    nodes = torch.cat((row, col), dim=-1)
    edge_index, _ = to_edge_index(adj_t)
    subset, new_edge_index, inv, edge_mask = pyg_k_hop_subgraph(
        nodes, k, edge_index=edge_index, num_nodes=adj_t.size(0),
        relabel_nodes=True)
    # subset[inv] = nodes. The new node id is based on `subset`'s order.
    # inv means the new idx (in subset) of the old nodes in `nodes`
    new_adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1],
                             sparse_sizes=(subset.size(0), subset.size(0)))
    new_edges = inv.view(2, -1)
    return new_adj_t, new_edges, subset


def dotproduct_naive(tensor1, tensor2):
    return (tensor1 * tensor2).sum(dim=-1)


def dotproduct_bmm(tensor1, tensor2):
    return torch.bmm(tensor1.unsqueeze(1), tensor2.unsqueeze(2)).view(-1)


dot_product = dotproduct_naive

########################
######### MLP ##########
########################


class CustomMLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
        tailnormactdrop=False,
        affine=True,
    ):
        super(CustomMLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.tailnormactdrop = tailnormactdrop
        self.affine = affine  # the affine in batchnorm
        self.layers = []
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
            if tailnormactdrop:
                self.__build_normactdrop(self.layers, output_dim,
                                         dropout_ratio)
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.__build_normactdrop(self.layers, hidden_dim, dropout_ratio)
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.__build_normactdrop(self.layers, hidden_dim,
                                         dropout_ratio)
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            if tailnormactdrop:
                self.__build_normactdrop(self.layers, hidden_dim,
                                         dropout_ratio)
        self.layers = nn.Sequential(*self.layers)

    def __build_normactdrop(self, layers, dim, dropout):
        if self.norm_type == "batch":
            layers.append(nn.BatchNorm1d(dim, affine=self.affine))
        elif self.norm_type == "layer":
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.Dropout(dropout, inplace=True))
        layers.append(nn.ReLU(inplace=True))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, feats, adj_t=None):
        return self.layers(feats)


########################
######### GNN ##########
########################


# Addpted from NCNC
class PureConv(nn.Module):
    aggr: Final[str]

    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1 + adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x


class MPLP_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, xdropout, use_feature=True, jk=False, gcn_name='gcn',
                 embedding=None):
        super(MPLP_GCN, self).__init__()

        self.use_feature = use_feature
        self.embedding = embedding
        self.dropout = dropout
        self.xdropout = xdropout
        self.input_size = 0
        self.jk = jk
        if jk:
            self.register_parameter("jkparams",
                                    nn.Parameter(torch.randn((num_layers, ))))
        if self.use_feature:
            self.input_size += in_channels
        if self.embedding is not None:
            self.input_size += embedding.embedding_dim
        self.convs = torch.nn.ModuleList()

        if self.input_size > 0:
            if gcn_name == 'gcn':
                conv_func = partial(GCNConv, cached=False)
            elif 'pure' in gcn_name:
                conv_func = LGConv
            self.xemb = nn.Sequential(nn.Dropout(xdropout))
            if ("pure" in gcn_name or num_layers == 0):
                self.xemb.append(nn.Linear(self.input_size, hidden_channels))
                self.xemb.append(
                    nn.Dropout(dropout, inplace=True) if dropout >
                    1e-6 else nn.Identity())
                self.input_size = hidden_channels
            self.convs.append(conv_func(self.input_size, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(conv_func(hidden_channels, hidden_channels))
            self.convs.append(conv_func(hidden_channels, out_channels))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, adj_t):
        if self.input_size > 0:
            xs = []
            if self.use_feature:
                xs.append(x)
            if self.embedding is not None:
                xs.append(self.embedding.weight)
            x = torch.cat(xs, dim=1)
            x = self.xemb(x)
            jkx = []
            for conv in self.convs:
                x = conv(x, adj_t)
                if self.jk:
                    jkx.append(x)
            if self.jk:  # JumpingKnowledge Connection
                jkx = torch.stack(jkx, dim=0)
                sftmax = self.jkparams.reshape(-1, 1, 1)
                x = torch.sum(jkx * sftmax, dim=0)
        return x


########################
######### MPLP #########
########################


class MPLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, feat_dropout,
                 label_dropout, num_hops=2, signature_sampling='torchhd',
                 use_degree='none', signature_dim=1024,
                 minimum_degree_onehot=-1, batchnorm_affine=True,
                 feature_combine="hadamard"):
        super(MPLP, self).__init__()

        self.in_channels = in_channels
        self.feat_dropout = feat_dropout
        self.label_dropout = label_dropout
        self.num_hops = num_hops
        self.signature_sampling = signature_sampling
        self.use_degree = use_degree
        self.feature_combine = feature_combine
        if self.use_degree == 'mlp':
            if USE_CUSTOM_MLP:
                self.node_weight_encode = CustomMLP(2, in_channels + 1, 32, 1,
                                                    feat_dropout,
                                                    norm_type="batch",
                                                    affine=batchnorm_affine)
            else:
                self.node_weight_encode = MLP(
                    num_layers=2, in_channels=in_channels + 1,
                    hidden_channels=32, out_channels=1,
                    dropout=self.label_dropout, act='relu', norm="BatchNorm",
                    norm_kwargs={"affine": batchnorm_affine})
        struct_dim = 8
        self.nodelabel = NodeLabel(signature_dim,
                                   signature_sampling=self.signature_sampling,
                                   minimum_degree_onehot=minimum_degree_onehot)
        if USE_CUSTOM_MLP:
            self.struct_encode = CustomMLP(1, struct_dim, struct_dim,
                                           struct_dim, self.label_dropout,
                                           "batch", tailnormactdrop=True,
                                           affine=batchnorm_affine)
        else:
            self.struct_encode = MLP(num_layers=1, in_channels=struct_dim,
                                     hidden_channels=struct_dim,
                                     out_channels=struct_dim,
                                     dropout=self.label_dropout, act='relu',
                                     plain_last=False, norm="BatchNorm",
                                     norm_kwargs={"affine": batchnorm_affine})

        dense_dim = struct_dim + in_channels
        if in_channels > 0:
            if feature_combine == "hadamard":
                feat_encode_input_dim = in_channels
            elif feature_combine == "plus_minus":
                feat_encode_input_dim = in_channels * 2
            if USE_CUSTOM_MLP:
                self.feat_encode = CustomMLP(2, feat_encode_input_dim,
                                             in_channels, in_channels,
                                             self.feat_dropout, "batch",
                                             tailnormactdrop=True,
                                             affine=batchnorm_affine)
            else:
                self.feat_encode = MLP(
                    num_layers=1, in_channels=feat_encode_input_dim,
                    hidden_channels=in_channels, out_channels=in_channels,
                    dropout=self.label_dropout, act='relu', plain_last=False,
                    norm="BatchNorm", norm_kwargs={"affine": batchnorm_affine})
        self.classifier = nn.Linear(dense_dim, 1)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, adj, edges, cache_mode=None):
        """Args:
        x: [N, in_channels] node embedding after GNN
        adj: [N, N] adjacency matrix
        edges: [2, E] target edges
        fast_inference: bool. If True, only caching the message-passing without calculating the structural features
        """
        if cache_mode in ["use", "delete"]:
            # no need to compute node_weight
            node_weight = None
        elif self.use_degree == 'none':
            node_weight = None
        elif self.use_degree == 'mlp':  # 'mlp' for now
            xs = []
            if self.in_channels > 0:
                xs.append(x)
            degree = adj.sum(dim=1).view(-1, 1).to(adj.device())
            xs.append(degree)
            node_weight_feat = torch.cat(xs, dim=1)
            node_weight = self.node_weight_encode(node_weight_feat).squeeze(
                -1) + 1  # like residual, can be learned as 0 if needed
        else:
            # AA or RA
            degree = adj.sum(dim=1).view(-1, 1).to(adj.device()).squeeze(
                -1) + 1  # degree at least 1. then log(degree) > 0.
            if self.use_degree == 'AA':
                node_weight = torch.sqrt(torch.reciprocal(torch.log(degree)))
            elif self.use_degree == 'RA':
                node_weight = torch.sqrt(torch.reciprocal(degree))
            node_weight = torch.nan_to_num(node_weight, nan=0.0, posinf=0.0,
                                           neginf=0.0)

        if cache_mode in ["build", "delete"]:
            propped = self.nodelabel(edges, adj, node_weight=node_weight,
                                     cache_mode=cache_mode)
            return
        else:
            propped = self.nodelabel(edges, adj, node_weight=node_weight,
                                     cache_mode=cache_mode)
        propped_stack = torch.stack([*propped], dim=1)
        out = self.struct_encode(propped_stack)

        if self.in_channels > 0:
            x_i = x[edges[0]]
            x_j = x[edges[1]]
            if self.feature_combine == "hadamard":
                x_ij = x_i * x_j
            elif self.feature_combine == "plus_minus":
                x_ij = torch.cat([x_i + x_j, torch.abs(x_i - x_j)], dim=1)
            x_ij = self.feat_encode(x_ij)
            x = torch.cat([x_ij, out], dim=1)
        else:
            x = out
        logit = self.classifier(x)
        return logit

    def precompute(self, adj):
        self(None, adj, None, cache_mode="build")
        return self
