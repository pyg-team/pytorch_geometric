from typing import List, Optional, Union

import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Batch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_sort_pool
from torch_geometric.utils import subgraph


class GCNConvWithDropout(torch.nn.Module):
    r"""
    A GCNConv followed by a Dropout.
    """
    def __init__(self, in_feats, out_feats, dropout=0.3, normalize=True,
                 bias=True):
        super(GCNConvWithDropout, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.gcn = GCNConv(in_feats, out_feats, normalize=normalize, bias=bias)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        out = self.gcn(x, edge_index)
        return out


class Discriminator(torch.nn.Module):
    r"""
    Description
    -----------
    A discriminator used to let the network to discrimate
    between positive (neighborhood of center node) and
    negative (any neighborhood in graph) samplings.

    Parameters
    ----------
    feat_dim : int
        The number of channels of node features.
    """
    def __init__(self, feat_dim: int):
        super(Discriminator, self).__init__()
        self.affine = torch.nn.Bilinear(feat_dim, feat_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.affine.weight)
        torch.nn.init.zeros_(self.affine.bias)

    def forward(self, h_x: Tensor, h_pos: Tensor, h_neg: Tensor,
                bias_pos: Optional[Tensor] = None,
                bias_neg: Optional[Tensor] = None):
        r"""
        Parameters
        ----------
        h_x : torch.Tensor
            Node features, shape: :obj:`(num_nodes, feat_dim)`
        h_pos : torch.Tensor
            The node features of positive samples
            It has the same shape as :obj:`h_x`
        h_neg : torch.Tensor
            The node features of negative samples
            It has the same shape as :obj:`h_x`
        bias_pos : torch.Tensor
            Bias parameter vector for positive scores
            shape: :obj:`(num_nodes)`
        bias_neg : torch.Tensor
            Bias parameter vector for negative scores
            shape: :obj:`(num_nodes)`

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            The output scores with shape (2 * num_nodes,), (num_nodes,)
        """
        score_pos = self.affine(h_pos, h_x).squeeze()
        score_neg = self.affine(h_neg, h_x).squeeze()
        if bias_pos is not None:
            score_pos = score_pos + bias_pos
        if bias_neg is not None:
            score_neg = score_neg + bias_neg

        logits = torch.cat((score_pos, score_neg), 0)

        return logits, score_pos


class DenseLayer(torch.nn.Module):
    r"""
    Description
    -----------
    Dense layer with a linear layer and an activation function
    """
    def __init__(self, in_dim: int, out_dim: int, act: str = "prelu",
                 bias=True):
        super(DenseLayer, self).__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.act_type = act.lower()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            torch.nn.init.zeros_(self.lin.bias)
        if self.act_type == "prelu":
            self.act = torch.nn.PReLU()
        else:
            self.act = torch.relu

    def forward(self, x):
        x = self.lin(x)
        return self.act(x)


class IndexSelect(torch.nn.Module):
    r"""
    Description
    -----------
    The index selection layer used by VIPool

    Parameters
    ----------
    pool_ratio : float
        The pooling ratio (for keeping nodes). For example,
        if `pool_ratio=0.8`, 80% nodes will be preserved.
    hidden_dim : int
        The number of channels in node features.
    act : str, optional
        The activation function type.
        Default: :obj:`'prelu'`
    dist : int, optional
        DO NOT USE THIS PARAMETER
    """
    def __init__(self, pool_ratio: float, hidden_dim: int, act: str = "prelu",
                 dist: int = 1):
        super(IndexSelect, self).__init__()
        self.pool_ratio = pool_ratio
        self.dist = dist
        self.dense = DenseLayer(hidden_dim, hidden_dim, act)
        self.discriminator = Discriminator(hidden_dim)
        self.gcn = GCNConvWithDropout(hidden_dim, hidden_dim)

    def forward(self, graph: Batch, h_pos: Tensor, h_neg: Tensor,
                bias_pos: Optional[Tensor] = None,
                bias_neg: Optional[Tensor] = None):
        r"""
        Description
        -----------
        Perform index selection

        Parameters
        ----------
        graph : torch_geometric.data.Batch
            Input graph.
        h_pos : torch.Tensor
            The node features of positive samples
            It has the same shape as :obj:`h_x`
        h_neg : torch.Tensor
            The node features of negative samples
            It has the same shape as :obj:`h_x`
        bias_pos : torch.Tensor
            Bias parameter vector for positive scores
            shape: :obj:`(num_nodes)`
        bias_neg : torch.Tensor
            Bias parameter vector for negative scores
            shape: :obj:`(num_nodes)`
        """
        # compute scores
        h_pos = self.dense(h_pos)
        h_neg = self.dense(h_neg)
        embed = self.gcn(h_pos, graph.edge_index)
        h_center = torch.sigmoid(embed)

        logit, logit_pos = self.discriminator(h_center, h_pos, h_neg, bias_pos,
                                              bias_neg)
        scores = torch.sigmoid(logit_pos)

        # sort scores
        scores, idx = torch.sort(scores, descending=True)

        # select top-k
        num_nodes = graph.num_nodes
        num_select_nodes = int(self.pool_ratio * num_nodes)
        size_list = [num_select_nodes, num_nodes - num_select_nodes]
        select_scores, _ = torch.split(scores, size_list, dim=0)
        select_idx, non_select_idx = torch.split(idx, size_list, dim=0)

        return logit, select_scores, select_idx, non_select_idx, embed


class GraphPool(torch.nn.Module):
    r"""
    Description
    -----------
    The pooling module for graph

    Parameters
    ----------
    hidden_dim : int
        The number of channels of node features.
    use_gcn : bool, optional
        Whether use gcn in down sampling process.
        default: :obj:`False`
    """
    def __init__(self, hidden_dim: int, use_gcn=False):
        super(GraphPool, self).__init__()
        self.use_gcn = use_gcn
        if use_gcn:
            self.down_sample_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)

    def forward(self, graph: Batch, feat: Tensor, select_idx: Tensor,
                non_select_idx: Optional[Tensor] = None,
                scores: Optional[Tensor] = None, pool_graph=False):
        r"""
        Description
        -----------
        Perform graph pooling.

        Parameters
        ----------
        graph : torch_geometric.data.Batch
            The input graph
        feat : torch.Tensor
            The input node feature
        select_idx : torch.Tensor
            The index in fine graph of node from
            coarse graph, this is obtained from
            previous graph pooling layers.
        non_select_idx : torch.Tensor, optional
            The index that not included in output graph.
            default: :obj:`None`
        scores : torch.Tensor, optional
            Scores for nodes used for pooling and scaling.
            default: :obj:`None`
        pool_graph : bool, optional
            Whether perform graph pooling on graph topology.
            default: :obj:`False`
        """
        if self.use_gcn:
            feat = self.down_sample_gcn(feat, graph.edge_index)

        feat = feat[select_idx]
        if scores is not None:
            feat = feat * scores.unsqueeze(-1)

        if pool_graph:
            sub_graph = graph.clone()
            sub_graph.num_nodes = select_idx.shape[0]
            sub_graph.x = sub_graph.x[select_idx]
            sub_graph.edge_index = subgraph(select_idx, graph.edge_index,
                                            relabel_nodes=True)[0]
            sub_graph.batch = graph.batch[select_idx]
            sub_graph.ptr = None
            return feat, sub_graph
        else:
            return feat


class GraphUnpool(torch.nn.Module):
    r"""
    Description
    -----------
    The unpooling module for graph

    Parameters
    ----------
    hidden_dim : int
        The number of channels of node features.
    """
    def __init__(self, hidden_dim: int):
        super(GraphUnpool, self).__init__()
        self.up_sample_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)

    def forward(self, graph: Batch, feat: Tensor, select_idx: Tensor):
        r"""
        Description
        -----------
        Perform graph unpooling

        Parameters
        ----------
        graph : torch_geometric.data.Batch
            The input graph
        feat : torch.Tensor
            The input node feature
        select_idx : torch.Tensor
            The index in fine graph of node from
            coarse graph, this is obtained from
            previous graph pooling layers.
        """
        fine_feat = torch.zeros((graph.num_nodes, feat.size(-1)),
                                device=feat.device)
        fine_feat[select_idx] = feat
        fine_feat = self.up_sample_gcn(fine_feat, graph.edge_index)
        return fine_feat


class GraphCrossModule(torch.nn.Module):
    r"""
    Description
    -----------
    The Graph Cross Module used by Graph Cross Networks.
    This module only contains graph cross layers.

    Parameters
    ----------
    pool_ratios : Union[float, List[float]]
        The pooling ratios (for keeping nodes) for each layer.
        For example, if `pool_ratio=0.8`, 80% nodes will be preserved.
        If a single float number is given, all pooling layers will have the
        same pooling ratio.
    in_dim : int
        The number of input node feature channels.
    out_dim : int
        The number of output node feature channels.
    hidden_dim : int
        The number of hidden node feature channels.
    cross_weight : float, optional
        The weight parameter used in graph cross layers
        Default: :obj:`1.0`
    fuse_weight : float, optional
        The weight parameter used at the end of GXN for channel fusion.
        Default: :obj:`1.0`
    """
    def __init__(self, pool_ratios: Union[float, List[float]], in_dim: int,
                 out_dim: int, hidden_dim: int, cross_weight: float = 1.,
                 fuse_weight: float = 1., dist: int = 1,
                 num_cross_layers: int = 2):
        super().__init__()
        if isinstance(pool_ratios, float):
            pool_ratios = (pool_ratios, pool_ratios)
        self.cross_weight = cross_weight
        self.fuse_weight = fuse_weight
        self.num_cross_layers = num_cross_layers

        # build network
        self.start_gcn_scale1 = GCNConvWithDropout(in_dim, hidden_dim)
        self.start_gcn_scale2 = GCNConvWithDropout(hidden_dim, hidden_dim)
        self.end_gcn = GCNConvWithDropout(2 * hidden_dim, out_dim)

        self.index_select_scale1 = IndexSelect(pool_ratios[0], hidden_dim,
                                               act="prelu", dist=dist)
        self.index_select_scale2 = IndexSelect(pool_ratios[1], hidden_dim,
                                               act="prelu", dist=dist)
        self.start_pool_s12 = GraphPool(hidden_dim)
        self.start_pool_s23 = GraphPool(hidden_dim)
        self.end_unpool_s21 = GraphUnpool(hidden_dim)
        self.end_unpool_s32 = GraphUnpool(hidden_dim)

        self.s1_l1_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)
        self.s1_l2_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)
        self.s1_l3_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)

        self.s2_l1_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)
        self.s2_l2_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)
        self.s2_l3_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)

        self.s3_l1_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)
        self.s3_l2_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)
        self.s3_l3_gcn = GCNConvWithDropout(hidden_dim, hidden_dim)

        if num_cross_layers >= 1:
            self.pool_s12_1 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s21_1 = GraphUnpool(hidden_dim)
            self.pool_s23_1 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s32_1 = GraphUnpool(hidden_dim)
        if num_cross_layers >= 2:
            self.pool_s12_2 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s21_2 = GraphUnpool(hidden_dim)
            self.pool_s23_2 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s32_2 = GraphUnpool(hidden_dim)

    def forward(self, graph):
        # start of scale-1
        graph_scale1 = graph
        feat_scale1 = self.start_gcn_scale1(graph.x, graph.edge_index)
        feat_origin = feat_scale1
        # negative samples
        feat_scale1_neg = feat_scale1[torch.randperm(feat_scale1.size(0))]
        logit_s1, scores_s1, select_idx_s1, non_select_idx_s1, feat_down_s1 = \
            self.index_select_scale1(graph_scale1,
                                     feat_scale1, feat_scale1_neg)
        feat_scale2, graph_scale2 = self.start_pool_s12(
            graph_scale1, feat_scale1, select_idx_s1, non_select_idx_s1,
            scores_s1, pool_graph=True)

        # start of scale-2
        feat_scale2 = self.start_gcn_scale2(feat_scale2,
                                            graph_scale2.edge_index)
        # negative samples
        feat_scale2_neg = feat_scale2[torch.randperm(feat_scale2.size(0))]
        logit_s2, scores_s2, select_idx_s2, non_select_idx_s2, feat_down_s2 = \
            self.index_select_scale2(graph_scale2,
                                     feat_scale2, feat_scale2_neg)
        feat_scale3, graph_scale3 = self.start_pool_s23(
            graph_scale2, feat_scale2, select_idx_s2, non_select_idx_s2,
            scores_s2, pool_graph=True)

        # layer-1
        res_s1_0, res_s2_0, res_s3_0 = feat_scale1, feat_scale2, feat_scale3

        feat_scale1 = F.relu(
            self.s1_l1_gcn(feat_scale1, graph_scale1.edge_index))
        feat_scale2 = F.relu(
            self.s2_l1_gcn(feat_scale2, graph_scale2.edge_index))
        feat_scale3 = F.relu(
            self.s3_l1_gcn(feat_scale3, graph_scale3.edge_index))

        if self.num_cross_layers >= 1:
            feat_s12_fu = self.pool_s12_1(graph_scale1, feat_scale1,
                                          select_idx_s1, non_select_idx_s1,
                                          scores_s1)
            feat_s21_fu = self.unpool_s21_1(graph_scale1, feat_scale2,
                                            select_idx_s1)
            feat_s23_fu = self.pool_s23_1(graph_scale2, feat_scale2,
                                          select_idx_s2, non_select_idx_s2,
                                          scores_s2)
            feat_s32_fu = self.unpool_s32_1(graph_scale2, feat_scale3,
                                            select_idx_s2)

            feat_scale1 = feat_scale1 + \
                self.cross_weight * feat_s21_fu + res_s1_0
            feat_scale2 = feat_scale2 + self.cross_weight * (
                feat_s12_fu + feat_s32_fu) / 2 + res_s2_0
            feat_scale3 = feat_scale3 + \
                self.cross_weight * feat_s23_fu + res_s3_0

        # layer-2
        feat_scale1 = F.relu(
            self.s1_l2_gcn(feat_scale1, graph_scale1.edge_index))
        feat_scale2 = F.relu(
            self.s2_l2_gcn(feat_scale2, graph_scale2.edge_index))
        feat_scale3 = F.relu(
            self.s3_l2_gcn(feat_scale3, graph_scale3.edge_index))

        if self.num_cross_layers >= 2:
            feat_s12_fu = self.pool_s12_2(graph_scale1, feat_scale1,
                                          select_idx_s1, non_select_idx_s1,
                                          scores_s1)
            feat_s21_fu = self.unpool_s21_2(graph_scale1, feat_scale2,
                                            select_idx_s1)
            feat_s23_fu = self.pool_s23_2(graph_scale2, feat_scale2,
                                          select_idx_s2, non_select_idx_s2,
                                          scores_s2)
            feat_s32_fu = self.unpool_s32_2(graph_scale2, feat_scale3,
                                            select_idx_s2)

            cross_weight = self.cross_weight * 0.05
            feat_scale1 = feat_scale1 + cross_weight * feat_s21_fu
            feat_scale2 = feat_scale2 + cross_weight * (feat_s12_fu +
                                                        feat_s32_fu) / 2
            feat_scale3 = feat_scale3 + cross_weight * feat_s23_fu

        # layer-3
        feat_scale1 = F.relu(
            self.s1_l3_gcn(feat_scale1, graph_scale1.edge_index))
        feat_scale2 = F.relu(
            self.s2_l3_gcn(feat_scale2, graph_scale2.edge_index))
        feat_scale3 = F.relu(
            self.s3_l3_gcn(feat_scale3, graph_scale3.edge_index))

        # final layers
        feat_s3_out = self.end_unpool_s32(graph_scale2, feat_scale3,
                                          select_idx_s2) + feat_down_s2
        feat_s2_out = self.end_unpool_s21(graph_scale1,
                                          feat_scale2 + feat_s3_out,
                                          select_idx_s1)
        feat_agg = feat_scale1 + self.fuse_weight * feat_s2_out + \
            self.fuse_weight * feat_down_s1
        feat_agg = torch.cat((feat_agg, feat_origin), dim=1)
        feat_agg = self.end_gcn(feat_agg, graph_scale1.edge_index)

        return feat_agg, logit_s1, logit_s2


class GraphCrossNet(torch.nn.Module):
    r""" Graph Cross Network from the paper `"Graph
    Cross Networks with Vertex Infomax Pooling"
    <https://arxiv.org/abs/2010.01804>`
    published at NeurIPS 2020.

    Parameters
    ----------
    in_dim : int
        The number of input node feature channels.
    out_dim : int
        The number of output node feature channels.
    edge_feat_dim : int, optional
        The number of input edge feature channels. Edge feature
        will be passed to a Linear layer and concatenated to
        input node features. Default: :obj:`0`
    hidden_dim : int, optional
        The number of hidden node feature channels.
        Default: :obj:`96`
    pool_ratios : Union[float, List[float]], optional
        The pooling ratios (for keeping nodes) for each layer.
        For example, if `pool_ratio=0.8`, 80% nodes will be preserved.
        If a single float number is given, all pooling layers will have the
        same pooling ratio.
        Default: :obj:`[0.9, 0.7]`
    readout_nodes : int, optional
        Number of nodes perserved in the final sort pool operation.
        Default: :obj:`30`
    conv1d_dims : List[int], optional
        The number of kernels of Conv1d operations.
        Default: :obj:`[16, 32]`
    conv1d_kws : List[int], optional
        The kernel size of Conv1d.
        Default: :obj:`[5]`
    cross_weight : float, optional
        The weight parameter used in graph cross layers
        Default: :obj:`1.0`
    fuse_weight : float, optional
        The weight parameter used at the end of GXN for channel fusion.
        Default: :obj:`1.0`
    """
    def __init__(self, in_dim: int, out_dim: int, edge_feat_dim: int = 0,
                 hidden_dim: int = 96, pool_ratios: Union[List[float],
                                                          float] = [0.9, 0.7],
                 readout_nodes: int = 30, conv1d_dims: List[int] = [16, 32],
                 conv1d_kws: List[int] = [5], cross_weight: float = 1.,
                 fuse_weight: float = 1., dist: int = 1):
        super(GraphCrossNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.readout_nodes = readout_nodes
        conv1d_kws = [hidden_dim] + conv1d_kws

        if edge_feat_dim > 0:
            self.in_dim += hidden_dim
            self.e2l_lin = torch.nn.Linear(edge_feat_dim, hidden_dim)
        else:
            self.e2l_lin = None

        self.gxn = GraphCrossModule(pool_ratios, in_dim=self.in_dim,
                                    out_dim=hidden_dim,
                                    hidden_dim=hidden_dim // 2,
                                    cross_weight=cross_weight,
                                    fuse_weight=fuse_weight, dist=dist)

        # final updates
        self.final_conv1 = torch.nn.Conv1d(1, conv1d_dims[0],
                                           kernel_size=conv1d_kws[0],
                                           stride=conv1d_kws[0])
        self.final_maxpool = torch.nn.MaxPool1d(2, 2)
        self.final_conv2 = torch.nn.Conv1d(conv1d_dims[0], conv1d_dims[1],
                                           kernel_size=conv1d_kws[1], stride=1)
        self.final_dense_dim = int((readout_nodes - 2) / 2 + 1)
        self.final_dense_dim = (self.final_dense_dim - conv1d_kws[1] +
                                1) * conv1d_dims[1]

        if self.out_dim > 0:
            self.out_lin = torch.nn.Linear(self.final_dense_dim, out_dim)

        self.init_weights()

    def init_weights(self):
        if self.e2l_lin is not None:
            torch.nn.init.xavier_normal_(self.e2l_lin.weight)
        torch.nn.init.xavier_normal_(self.final_conv1.weight)
        torch.nn.init.xavier_normal_(self.final_conv2.weight)
        if self.out_dim > 0:
            torch.nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, graph):
        node_feat = graph.x
        num_batch = graph.num_graphs

        node_feat, logits1, logits2 = self.gxn(graph)
        batch_sortpool_feats = global_sort_pool(node_feat, graph.batch,
                                                self.readout_nodes)

        # final updates
        to_conv1d = batch_sortpool_feats.unsqueeze(1)
        conv1d_result = F.relu(self.final_conv1(to_conv1d))
        conv1d_result = self.final_maxpool(conv1d_result)
        conv1d_result = F.relu(self.final_conv2(conv1d_result))

        to_dense = conv1d_result.view(num_batch, -1)
        if self.out_dim > 0:
            out = F.relu(self.out_lin(to_dense))
        else:
            out = to_dense

        return out, logits1, logits2
