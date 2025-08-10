import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from ...nn.conv import MessagePassing
from ...nn.dense.linear import Linear
from ...nn.inits import glorot, zeros
from ...typing import Adj, OptTensor, Tuple
from ...utils import get_ppr, is_sparse, scatter, softmax
from .basic_gnn import GCN


class LPFormer(nn.Module):
    r"""The LPFormer model from the
    `"LPFormer: An Adaptive Graph Transformer for Link Prediction"
    <https://arxiv.org/abs/2310.11009>`_ paper.

    .. note::

        For an example of using LPFormer, see
        `examples/lpformer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        lpformer.py>`_.

    Args:
        in_channels (int): Size of input dimension
        hidden_channels (int): Size of hidden dimension
        num_gnn_layers (int, optional): Number of GNN layers
            (default: :obj:`2`)
        gnn_dropout(float, optional): Dropout used for GNN
            (default: :obj:`0.1`)
        num_transformer_layers (int, optional): Number of Transformer layers
            (default: :obj:`1`)
        num_heads (int, optional): Number of heads to use in MHA
            (default: :obj:`1`)
        transformer_dropout (float, optional): Dropout used for Transformer
            (default: :obj:`0.1`)
        ppr_thresholds (list): PPR thresholds for different types of nodes.
            Types include (in order) common neighbors, 1-Hop nodes
            (that aren't CNs), and all other nodes.
            (default: :obj:`[0, 1e-4, 1e-2]`)
        gcn_cache (bool, optional): Whether to cache edge indices
            during message passing. (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_gnn_layers: int = 2,
        gnn_dropout: float = 0.1,
        num_transformer_layers: int = 1,
        num_heads: int = 1,
        transformer_dropout: float = 0.1,
        ppr_thresholds: list = None,
        gcn_cache=False,
    ):
        super().__init__()

        # Default thresholds
        if ppr_thresholds is None:
            ppr_thresholds = [0, 1e-4, 1e-2]

        if len(ppr_thresholds) == 3:
            self.thresh_cn = ppr_thresholds[0]
            self.thresh_1hop = ppr_thresholds[1]
            self.thresh_non1hop = ppr_thresholds[2]
        else:
            raise ValueError(
                "Argument 'ppr_thresholds' must only be length 3!")

        self.in_dim = in_channels
        self.hid_dim = hidden_channels
        self.gnn_drop = gnn_dropout
        self.trans_drop = transformer_dropout

        self.gnn = GCN(in_channels, hidden_channels, num_gnn_layers,
                       dropout=gnn_dropout, norm="layer_norm",
                       cached=gcn_cache)
        self.gnn_norm = nn.LayerNorm(hidden_channels)

        # Create Transformer Layers
        self.att_layers = nn.ModuleList()
        for il in range(num_transformer_layers):
            if il == 0:
                node_dim = None
                self.out_dim = self.hid_dim * 2 if num_transformer_layers > 1 \
                    else self.hid_dim
            elif il == self.num_layers - 1:
                node_dim = self.hid_dim
            else:
                self.out_dim = node_dim = self.hid_dim

            self.att_layers.append(
                LPAttLayer(self.hid_dim, self.out_dim, node_dim, num_heads,
                           self.trans_drop))

        self.elementwise_lin = MLP(self.hid_dim, self.hid_dim, self.hid_dim)

        # Relative Positional Encodings
        self.ppr_encoder_cn = MLP(2, self.hid_dim, self.hid_dim)
        self.ppr_encoder_onehop = MLP(2, self.hid_dim, self.hid_dim)
        self.ppr_encoder_non1hop = MLP(2, self.hid_dim, self.hid_dim)

        # thresh=1 implies ignoring some set of nodes
        # Also allows us to be more efficient later
        if self.thresh_non1hop == 1 and self.thresh_1hop == 1:
            self.mask = "cn"
        elif self.thresh_non1hop == 1 and self.thresh_1hop < 1:
            self.mask = "1-hop"
        else:
            self.mask = "all"

        # 4 is for counts of diff nodes
        pairwise_dim = self.hid_dim * num_heads + 4
        self.pairwise_lin = MLP(pairwise_dim, pairwise_dim, self.hid_dim)

        self.score_func = MLP(self.hid_dim * 2, self.hid_dim * 2, 1, norm=None)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.hid_dim}, num_gnn_layers={self.gnn.num_layers}, '
                f'num_transformer_layers={len(self.att_layers)})')

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()
        self.gnn_norm.reset_parameters()
        self.elementwise_lin.reset_parameters()
        self.pairwise_lin.reset_parameters()
        self.ppr_encoder_cn.reset_parameters()
        self.ppr_encoder_onehop.reset_parameters()
        self.ppr_encoder_non1hop.reset_parameters()
        self.score_func.reset_parameters()
        for i in range(len(self.att_layers)):
            self.att_layers[i].reset_parameters()

    def forward(
        self,
        batch: Tensor,
        x: Tensor,
        edge_index: Adj,
        ppr_matrix: Tensor,
    ) -> Tensor:
        r"""Forward Pass of LPFormer.

        Returns raw logits for each link

        Args:
            batch (Tensor): The batch vector.
                Denotes which node pairs to predict.
            x (Tensor): Input node features
            edge_index (torch.Tensor, SparseTensor): The edge indices.
                Either in COO or SparseTensor format
            ppr_matrix (Tensor): PPR matrix
        """
        batch = batch.to(x.device)

        X_node = self.propagate(x, edge_index)
        x_i, x_j = X_node[batch[0]], X_node[batch[1]]
        elementwise_edge_feats = self.elementwise_lin(x_i * x_j)

        # Ensure in sparse format
        # Need as native torch.sparse for later computations
        # (necessary operations are not supported by PyG SparseTensor)
        if not edge_index.is_sparse:
            num_nodes = ppr_matrix.size(1)
            vals = torch.ones(len(edge_index[0]), device=edge_index.device)
            edge_index = torch.sparse_coo_tensor(edge_index, vals,
                                                 [num_nodes, num_nodes])
        # Checks if SparseTensor, if so the convert
        if is_sparse(edge_index) and not edge_index.is_sparse:
            edge_index = edge_index.to_torch_sparse_coo_tensor()

        # Ensure {0, 1}
        edge_index = edge_index.coalesce().bool().int()

        pairwise_feats = self.calc_pairwise(batch, X_node, edge_index,
                                            ppr_matrix)
        combined_feats = torch.cat((elementwise_edge_feats, pairwise_feats),
                                   dim=-1)

        logits = self.score_func(combined_feats)
        return logits

    def propagate(self, x: Tensor, adj: Adj) -> Tensor:
        """Propagate via GNN.

        Args:
            x (Tensor): Node features
            adj (torch.Tensor, SparseTensor): Adjacency matrix
        """
        x = F.dropout(x, p=self.gnn_drop, training=self.training)
        X_node = self.gnn(x, adj)
        X_node = self.gnn_norm(X_node)

        return X_node

    def calc_pairwise(self, batch: Tensor, X_node: Tensor, adj_mask: Tensor,
                      ppr_matrix: Tensor) -> Tensor:
        r"""Calculate the pairwise features for the node pairs.

        Args:
            batch (Tensor): The batch vector.
                Denotes which node pairs to predict.
            X_node (Tensor): Node representations
            adj_mask (Tensor): Mask of adjacency matrix used for computing the
                different node types.
            ppr_matrix (Tensor): PPR matrix
        """
        k_i, k_j = X_node[batch[0]], X_node[batch[1]]
        pairwise_feats = torch.cat((k_i, k_j), dim=-1)

        cn_info, onehop_info, non1hop_info = self.compute_node_mask(
            batch, adj_mask, ppr_matrix)

        all_mask = cn_info[0]
        if onehop_info is not None:
            all_mask = torch.cat((all_mask, onehop_info[0]), dim=-1)
        if non1hop_info is not None:
            all_mask = torch.cat((all_mask, non1hop_info[0]), dim=-1)

        pes = self.get_pos_encodings(cn_info[1:], onehop_info[1:],
                                     non1hop_info[1:])

        for lay in range(len(self.att_layers)):
            pairwise_feats = self.att_layers[lay](all_mask, pairwise_feats,
                                                  X_node, pes)

        num_cns, num_1hop, num_non1hop, num_neigh = self.get_structure_cnts(
            batch, cn_info, onehop_info, non1hop_info)

        pairwise_feats = torch.cat(
            (pairwise_feats, num_cns, num_1hop, num_non1hop, num_neigh),
            dim=-1)

        pairwise_feats = self.pairwise_lin(pairwise_feats)
        return pairwise_feats

    def get_pos_encodings(
            self, cn_ppr: Tuple[Tensor, Tensor],
            onehop_ppr: Optional[Tuple[Tensor, Tensor]] = None,
            non1hop_ppr: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        r"""Calculate the PPR-based relative positional encodings.

        Due to thresholds, sometimes we don't have 1-hop or >1-hop nodes.
        In those cases, the value of onehop_ppr and/or non1hop_ppr should
        be `None`.

        Args:
            cn_ppr (tuple, optional): PPR scores of CNs.
            onehop_ppr (tuple, optional): PPR scores of 1-Hop.
                (default: :obj:`None`)
            non1hop_ppr (tuple, optional): PPR scores of >1-Hop.
                (default: :obj:`None`)
        """
        cn_a = self.ppr_encoder_cn(torch.stack((cn_ppr[0], cn_ppr[1])).t())
        cn_b = self.ppr_encoder_cn(torch.stack((cn_ppr[1], cn_ppr[0])).t())
        cn_pe = cn_a + cn_b

        if onehop_ppr is None:
            return cn_pe

        onehop_a = self.ppr_encoder_onehop(
            torch.stack((onehop_ppr[0], onehop_ppr[1])).t())
        onehop_b = self.ppr_encoder_onehop(
            torch.stack((onehop_ppr[1], onehop_ppr[0])).t())
        onehop_pe = onehop_a + onehop_b

        if non1hop_ppr is None:
            return torch.cat((cn_pe, onehop_pe), dim=0)

        non1hop_a = self.ppr_encoder_non1hop(
            torch.stack((non1hop_ppr[0], non1hop_ppr[1])).t())
        non1hop_b = self.ppr_encoder_non1hop(
            torch.stack((non1hop_ppr[1], non1hop_ppr[0])).t())
        non1hop_pe = non1hop_a + non1hop_b

        return torch.cat((cn_pe, onehop_pe, non1hop_pe), dim=0)

    def compute_node_mask(
            self, batch: Tensor, adj: Tensor, ppr_matrix: Tensor
    ) -> Tuple[Tuple, Optional[Tuple], Optional[Tuple]]:
        r"""Get mask based on type of node.

        When mask_type is not "cn", also return the ppr vals for both
        the source and target.

        Args:
            batch (Tensor): The batch vector.
                Denotes which node pairs to predict.
            adj (SparseTensor): Adjacency matrix
            ppr_matrix (Tensor): PPR matrix
        """
        src_adj = torch.index_select(adj, 0, batch[0])
        tgt_adj = torch.index_select(adj, 0, batch[1])

        if self.mask == "cn":
            # 1 when CN, 0 otherwise
            pair_adj = src_adj * tgt_adj
        else:
            # Equals: {0: ">1-Hop", 1: "1-Hop (Non-CN)", 2: "CN"}
            pair_adj = src_adj + tgt_adj

        pair_ix, node_type, src_ppr, tgt_ppr = self.get_ppr_vals(
            batch, pair_adj, ppr_matrix)

        cn_filt_cond = (src_ppr >= self.thresh_cn) & (tgt_ppr
                                                      >= self.thresh_cn)
        onehop_filt_cond = (src_ppr >= self.thresh_1hop) & (
            tgt_ppr >= self.thresh_1hop)

        if self.mask != "cn":
            filt_cond = torch.where(node_type == 1, onehop_filt_cond,
                                    cn_filt_cond)
        else:
            filt_cond = torch.where(node_type == 0, onehop_filt_cond,
                                    cn_filt_cond)

        pair_ix, node_type = pair_ix[:, filt_cond], node_type[filt_cond]
        src_ppr, tgt_ppr = src_ppr[filt_cond], tgt_ppr[filt_cond]

        # >1-Hop mask is gotten separately
        if self.mask == "all":
            non1hop_ix, non1hop_sppr, non1hop_tppr = self.get_non_1hop_ppr(
                batch, adj, ppr_matrix)

        # Dropout
        if self.training and self.trans_drop > 0:
            pair_ix, src_ppr, tgt_ppr, node_type = self.drop_pairwise(
                pair_ix, src_ppr, tgt_ppr, node_type)
            if self.mask == "all":
                non1hop_ix, non1hop_sppr, non1hop_tppr, _ = self.drop_pairwise(
                    non1hop_ix, non1hop_sppr, non1hop_tppr)

        # Separate out CN and 1-Hop
        if self.mask != "cn":
            cn_ind = node_type == 2
            cn_ix = pair_ix[:, cn_ind]
            cn_src_ppr = src_ppr[cn_ind]
            cn_tgt_ppr = tgt_ppr[cn_ind]

            one_hop_ind = node_type == 1
            onehop_ix = pair_ix[:, one_hop_ind]
            onehop_src_ppr = src_ppr[one_hop_ind]
            onehop_tgt_ppr = tgt_ppr[one_hop_ind]

        if self.mask == "cn":
            return (pair_ix, src_ppr, tgt_ppr), None, None
        elif self.mask == "1-hop":
            return (cn_ix, cn_src_ppr, cn_tgt_ppr), (onehop_ix, onehop_src_ppr,
                                                     onehop_tgt_ppr), None
        else:
            return (cn_ix, cn_src_ppr,
                    cn_tgt_ppr), (onehop_ix, onehop_src_ppr,
                                  onehop_tgt_ppr), (non1hop_ix, non1hop_sppr,
                                                    non1hop_tppr)

    def get_ppr_vals(
            self, batch: Tensor, pair_diff_adj: Tensor,
            ppr_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Get the src and tgt ppr vals.

        Returns the: link the node belongs to, type of node
        (e.g., CN), PPR relative to src, PPR relative to tgt.

        Args:
            batch (Tensor): The batch vector.
                Denotes which node pairs to predict.
            pair_diff_adj (SparseTensor): Combination of rows in
                adjacency for src and tgt nodes (e.g., X1 + X2)
            ppr_matrix (Tensor): PPR matrix
        """
        # Additional terms for also choosing scores when ppr=0
        # Multiplication removes any values for nodes not in batch
        # Addition then adds offset to ensure we select when ppr=0
        # All selected scores are +1 higher than their true val
        src_ppr_adj = torch.index_select(
            ppr_matrix, 0, batch[0]) * pair_diff_adj + pair_diff_adj
        tgt_ppr_adj = torch.index_select(
            ppr_matrix, 0, batch[1]) * pair_diff_adj + pair_diff_adj

        # Can now convert ppr scores to dense
        ppr_ix = src_ppr_adj.coalesce().indices()
        src_ppr = src_ppr_adj.coalesce().values()
        tgt_ppr = tgt_ppr_adj.coalesce().values()

        # TODO: Needed due to a bug in recent torch versions
        # see here for more - https://github.com/pytorch/pytorch/issues/114529
        # note that if one is 0 so is the other
        zero_vals = (src_ppr != 0)
        src_ppr = src_ppr[zero_vals]
        tgt_ppr = tgt_ppr[tgt_ppr != 0]
        ppr_ix = ppr_ix[:, zero_vals]

        pair_diff_adj = pair_diff_adj.coalesce().values()
        node_type = pair_diff_adj[src_ppr != 0]

        # Remove additional +1 from each ppr val
        src_ppr = (src_ppr - node_type) / node_type
        tgt_ppr = (tgt_ppr - node_type) / node_type

        return ppr_ix, node_type, src_ppr, tgt_ppr

    def drop_pairwise(
        self,
        pair_ix: Tensor,
        src_ppr: Optional[Tensor] = None,
        tgt_ppr: Optional[Tensor] = None,
        node_indicator: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Perform dropout on pairwise information
        by randomly dropping a percentage of nodes.

        Done before performing attention for efficiency

        Args:
            pair_ix (Tensor): Link node belongs to
            src_ppr (Tensor, optional): PPR relative to src
                (default: :obj:`None`)
            tgt_ppr (Tensor, optional): PPR relative to tgt
                (default: :obj:`None`)
            node_indicator (Tensor, optional): Type of node (e.g., CN)
                (default: :obj:`None`)
        """
        num_indices = math.ceil(pair_ix.size(1) * (1 - self.trans_drop))
        indices = torch.randperm(pair_ix.size(1))[:num_indices]
        pair_ix = pair_ix[:, indices]

        if src_ppr is not None:
            src_ppr = src_ppr[indices]
        if tgt_ppr is not None:
            tgt_ppr = tgt_ppr[indices]
        if node_indicator is not None:
            node_indicator = node_indicator[indices]

        return pair_ix, src_ppr, tgt_ppr, node_indicator

    def get_structure_cnts(
        self,
        batch: Tensor,
        cn_info: Tuple[Tensor, Tensor],
        onehop_info: Tuple[Tensor, Tensor],
        non1hop_info: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Counts for CNs, 1-Hop, and >1-Hop that satisfy PPR threshold.

        Also include total # of neighbors

        Args:
            batch (Tensor): The batch vector.
                Denotes which node pairs to predict.
            cn_info (tuple): Information of CN nodes
                Contains (ID of node, src ppr, tgt ppr)
            onehop_info (tuple): Information of 1-Hop nodes.
                Contains (ID of node, src ppr, tgt ppr)
            non1hop_info (tuple): Information of >1-Hop nodes.
                Contains (ID of node, src ppr, tgt ppr)
        """
        num_cns = self.get_num_ppr_thresh(batch, cn_info[0], cn_info[1],
                                          cn_info[2], self.thresh_cn)
        num_1hop = self.get_num_ppr_thresh(batch, onehop_info[0],
                                           onehop_info[1], onehop_info[2],
                                           self.thresh_1hop)

        # TOTAL num of 1-hop neighbors union
        num_ppr_ones = self.get_num_ppr_thresh(batch, onehop_info[0],
                                               onehop_info[1], onehop_info[2],
                                               thresh=0)
        num_neighbors = num_cns + num_ppr_ones

        # Process for >1-hop is different which is why we use get_count below
        if non1hop_info is None:
            return num_cns, num_1hop, 0, num_neighbors
        else:
            num_non1hop = self.get_count(non1hop_info[0], batch)
            return num_cns, num_1hop, num_non1hop, num_neighbors

    def get_num_ppr_thresh(self, batch: Tensor, node_mask: Tensor,
                           src_ppr: Tensor, tgt_ppr: Tensor,
                           thresh: float) -> Tensor:
        """Get # of nodes `v` where `ppr(a, v) >= eta` & `ppr(b, v) >= eta`.

        Args:
            batch (Tensor): The batch vector.
                Denotes which node pairs to predict.
            node_mask (Tensor): IDs of nodes
            src_ppr (Tensor): PPR relative to src node
            tgt_ppr (Tensor): PPR relative to tgt node
            thresh (float): PPR threshold for nodes (`eta`)
        """
        weight = torch.ones(node_mask.size(1), device=node_mask.device)

        ppr_above_thresh = (src_ppr >= thresh) & (tgt_ppr >= thresh)
        num_ppr = scatter(ppr_above_thresh.float() * weight,
                          node_mask[0].long(), dim=0, dim_size=batch.size(1),
                          reduce="sum")
        num_ppr = num_ppr.unsqueeze(-1)

        return num_ppr

    def get_count(
        self,
        node_mask: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """# of nodes for each sample in batch.

        They node have already filtered by PPR beforehand

        Args:
            node_mask (Tensor): IDs of nodes
            batch (Tensor): The batch vector.
                Denotes which node pairs to predict.
        """
        weight = torch.ones(node_mask.size(1), device=node_mask.device)
        num_nodes = scatter(weight, node_mask[0].long(), dim=0,
                            dim_size=batch.size(1), reduce="sum")
        num_nodes = num_nodes.unsqueeze(-1)

        return num_nodes

    def get_non_1hop_ppr(self, batch: Tensor, adj: Tensor,
                         ppr_matrix: Tensor) -> Tensor:
        r"""Get PPR scores for non-1hop nodes.

        Args:
            batch (Tensor): Links in batch
            adj (Tensor): Adjacency matrix
            ppr_matrix (Tensor): Sparse PPR matrix
        """
        # NOTE: Use original adj (one pass in forward() removes links in batch)
        # Done since removing them converts src/tgt nodes to >1-hop nodes.
        # Therefore removing CN and 1-hop will also remove the batch links.

        # During training we add back in the links in the batch
        # (we're removed from adjacency before being passed to model)
        # Done since otherwise they will be mistakenly seen as >1-Hop nodes
        # Instead they're 1-Hop, and get ignored accordingly
        # Ignored during eval since we know the links aren't in the adj
        adj2 = adj
        if self.training:
            n = adj.size(0)
            batch_flip = torch.cat(
                (batch, torch.flip(batch, (0, )).to(batch.device)), dim=-1)
            batch_ones = torch.ones_like(batch_flip[0], device=batch.device)
            adj_edges = torch.sparse_coo_tensor(batch_flip, batch_ones, [n, n],
                                                device=batch.device)
            adj_edges = adj_edges
            adj2 = (adj + adj_edges).coalesce().bool().int()

        src_adj = torch.index_select(adj2, 0, batch[0])
        tgt_adj = torch.index_select(adj2, 0, batch[1])

        src_ppr = torch.index_select(ppr_matrix, 0, batch[0])
        tgt_ppr = torch.index_select(ppr_matrix, 0, batch[1])

        # Remove CN scores
        src_ppr = src_ppr - src_ppr * (src_adj * tgt_adj)
        tgt_ppr = tgt_ppr - tgt_ppr * (src_adj * tgt_adj)
        # Also need to remove CN entries in Adj
        # Otherwise they leak into next computation
        src_adj = src_adj - src_adj * (src_adj * tgt_adj)
        tgt_adj = tgt_adj - tgt_adj * (src_adj * tgt_adj)

        # Remove 1-Hop scores
        src_ppr = src_ppr - src_ppr * (src_adj + tgt_adj)
        tgt_ppr = tgt_ppr - tgt_ppr * (src_adj + tgt_adj)

        # Make sure we include both when we convert to dense so indices align
        # Do so by adding 1 to each based on the other
        src_ppr_add = src_ppr + torch.sign(tgt_ppr)
        tgt_ppr_add = tgt_ppr + torch.sign(src_ppr)

        src_ix = src_ppr_add.coalesce().indices()
        src_vals = src_ppr_add.coalesce().values()
        tgt_vals = tgt_ppr_add.coalesce().values()

        # Now we can remove value which is just 1
        # Technically creates -1 scores for ppr scores that were 0
        # Doesn't matter as they'll be filtered out by condition later
        src_vals = src_vals - 1
        tgt_vals = tgt_vals - 1

        ppr_condition = (src_vals >= self.thresh_non1hop) & (
            tgt_vals >= self.thresh_non1hop)
        src_ix, src_vals, tgt_vals = src_ix[:, ppr_condition], src_vals[
            ppr_condition], tgt_vals[ppr_condition]

        return src_ix, src_vals, tgt_vals

    def calc_sparse_ppr(self, edge_index: Tensor, num_nodes: int,
                        alpha: float = 0.15, eps: float = 5e-5) -> Tensor:
        r"""Calculate the PPR of the graph in sparse format.

        Args:
            edge_index: The edge indices
            num_nodes: Number of nodes
            alpha (float, optional): The alpha value of the PageRank algorithm.
                (default: :obj:`0.15`)
            eps (float, optional): Threshold for stopping the PPR calculation
                (default: :obj:`5e-5`)
        """
        ei, ei_w = get_ppr(edge_index.cpu(), alpha=alpha, eps=eps,
                           num_nodes=num_nodes)
        ppr_matrix = torch.sparse_coo_tensor(ei, ei_w, [num_nodes, num_nodes])

        return ppr_matrix


class LPAttLayer(MessagePassing):
    r"""Attention Layer for pairwise interaction module.

    Args:
        in_channels (int): Size of input dimension
        out_channels (int): Size of output dimension
        node_dim (int): Dimension of nodes being aggregated
        num_heads (int): Number of heads to use in MHA
        dropout (float): Dropout on attention values
        concat (bool, optional): Whether to concat attention
            heads. Otherwise sum (default: :obj:`True`)
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        node_dim: int,
        num_heads: int,
        dropout: float,
        concat: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, flow="target_to_source", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = 0.2  # LeakyRelu

        out_dim = 2
        if node_dim is None:
            node_dim = in_channels * out_dim
        else:
            node_dim = node_dim * out_dim

        self.lin_l = Linear(in_channels, self.heads * out_channels,
                            weight_initializer='glorot')
        self.lin_r = Linear(node_dim, self.heads * out_channels,
                            weight_initializer='glorot')

        att_out = out_channels
        self.att = Parameter(Tensor(1, self.heads, att_out))

        if concat:
            self.bias = Parameter(Tensor(self.heads * out_channels))
        else:
            self.bias = Parameter(Tensor(out_channels))

        self._alpha = None

        self.dropout = dropout
        self.post_att_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.post_att_norm.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        edge_index: Tensor,
        edge_feats: Tensor,
        node_feats: Tensor,
        ppr_rpes: Tensor,
    ) -> Tensor:
        """Runs the forward pass of the module.

        Args:
            edge_index (Tensor): The edge indices.
            edge_feats (Tensor): Concatenated representations
                of src and target nodes for each link
            node_feats (Tensor): Representations for individual
                nodes
            ppr_rpes (Tensor): Relative PEs for each node
        """
        out = self.propagate(edge_index, x=(edge_feats, node_feats),
                             ppr_rpes=ppr_rpes, size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        out = self.post_att_norm(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, ppr_rpes: Tensor,
                index: Tensor, ptr: Tensor, size_i: Optional[int]) -> Tensor:
        H, C = self.heads, self.out_channels

        x_j = torch.cat((x_j, ppr_rpes), dim=-1)
        x_j = self.lin_r(x_j).view(-1, H, C)

        # e=(a, b) attending to v
        e1, e2 = x_i.chunk(2, dim=-1)
        e1 = self.lin_l(e1).view(-1, H, C)
        e2 = self.lin_l(e2).view(-1, H, C)
        x = x_j * (e1 + e2)

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        return x_j * alpha.unsqueeze(-1)


class MLP(nn.Module):
    r"""L Layer MLP."""
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int,
                 num_layers: int = 2, drop: int = 0, norm: str = "layer"):
        super().__init__()
        self.dropout = drop

        if norm == "batch":
            self.norm = nn.BatchNorm1d(hid_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(hid_channels)
        else:
            self.norm = None

        self.linears = torch.nn.ModuleList()

        if num_layers == 1:
            self.linears.append(nn.Linear(in_channels, out_channels))
        else:
            self.linears.append(nn.Linear(in_channels, hid_channels))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hid_channels, hid_channels))
            self.linears.append(nn.Linear(hid_channels, out_channels))

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for lin in self.linears[:-1]:
            x = lin(x)
            x = self.norm(x) if self.norm is not None else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linears[-1](x)

        return x.squeeze(-1)
