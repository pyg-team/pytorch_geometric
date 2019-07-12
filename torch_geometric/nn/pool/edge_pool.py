from torch import nn
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
import numpy as np
from torch_sparse import coalesce
from collections import namedtuple


def _compute_new_edges(old_to_new_node_idx, edge_index, n_nodes):
    """
    Computes the new edge index given a mapping from old to new node idx.

    Args:
        old_to_new_node_idx (int tensor): The mapping from old nodes to new
            ones. Each entry is an old node, and the value is the index of
            the new node it is mapped to.
        edge_index (int tensor): The old edge index.
        n_nodes (int): The total number of nodes in the graph (this might
            not be computable from edge_index if there are nodes that are
            isolated).

    Returns:
        new_edge_index (int tensor): The new edge_index. These contain
        self-loops iff multiple old nodes have been mapped to the same
        new node and have an edge between them. Depending on your
        usecase, either strip self-loops or add the missing ones.
    """
    new_edge_index = torch.stack([
        old_to_new_node_idx[edge_index[0]],
        old_to_new_node_idx[edge_index[1]]
    ])
    new_edge_index, _ = coalesce(new_edge_index, None, n_nodes, n_nodes)
    return new_edge_index


class EdgePooling(nn.Module):
    """
    The EdgePool operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_
    papers.

    In short, a score is computed for each edge. Edges are contracted
    iteratively according to that score unless one of their nodes has
    already been part of a contracted edge.

    To duplicate the configuration from "Towards Graph Pooling by Edge
    Contraction", use either `compute_edge_score_tanh` or
    `compute_edge_score_softmax`, and set `add_to_edge_score` to 0.

    To duplicate the configuration from "Edge Contraction Pooling for
    Graph Neural Networks", set `dropout_prob` to `0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is softmax over all incoming edges for each node.
            Functions that can be used take `raw_edge_score`
            (float tensor of length nodes) and `edge_index`, and produce
            a new tensor of the same size as `raw_edge_score`. Included
            functions are `EdgePooling.compute_edge_score_softmax`,
            `EdgePooling.compute_edge_score_tanh`, and
            `EdgePooling.compute_edge_score_sigmoid`.
        dropout_prob (float in [0, 1), optional): The probability with
            which to drop edge scores during training. 0, by default.
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. By default, it is 0.5.
    """

    unpool_description = namedtuple("UnpoolDescription",
                                    ["edge_index",
                                     "old_to_new_node_idx",
                                     "batch",
                                     "new_nodes_edge_scores"
                                     ])

    def __init__(self,
                 in_channels,
                 edge_score_method=None,
                 dropout_prob=0,
                 add_to_edge_score=0.5
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.build_network(in_channels)
        assert 0 <= dropout_prob < 1
        self.dropout_prob = dropout_prob
        self.add_to_edge_score = add_to_edge_score
        self.compute_edge_score = edge_score_method or \
            self.compute_edge_score_softmax

    def build_network(self, in_channels):
        self.net = nn.Linear(2 * in_channels, 1)

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index):
        """
        Computes the edge scores from raw edge scores using softmax.

        Specifically, each edge's score is normalized such that all
        edge scores leading into one node sum up to 1.

        Args:
            raw_edge_score (float tensor, (n_edges)): The raw edge
                scores.
            edge_index (int tensor, (2, n_edges)): The edges.

        Returns:
            edge_score (float tensor, (n_edges)): The normalized
                edge scores
        """
        max_per_node = scatter_max(
            raw_edge_score,
            edge_index[1],
            fill_value=-np.inf
        )[0]
        edge_logits = raw_edge_score - max_per_node[edge_index[1]]
        edge_exps = torch.exp(edge_logits)
        edge_exps_by_node = scatter_add(edge_exps, edge_index[1])
        edge_score = edge_exps / edge_exps_by_node[edge_index[1]]
        return edge_score

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index):
        """
        Computes the edge scores from raw edge scores using
        element-wise tanh.

        Args:
            raw_edge_score (float tensor, (n_edges)): The raw edge
                scores.
            edge_index (int tensor, (2, n_edges)): The edges.

        Returns:
            edge_score (float tensor, (n_edges)): The normalized
                edge scores
        """

        edge_score = torch.tanh(raw_edge_score)
        return edge_score

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index):
        """
        Computes the edge scores from raw edge scores using
        element-wise sigmoid.

        Args:
            raw_edge_score (float tensor, (n_edges)): The raw edge
                scores.
            edge_index (int tensor, (2, n_edges)): The edges.

        Returns:
            edge_score (float tensor, (n_edges)): The normalized
                edge scores
        """
        edge_score = torch.sigmoid(raw_edge_score)
        return edge_score

    def _compute_raw_edge_score(self, x, edge_index):
        """
        Computes the raw edge scores from node features and edge
        indices.

        For this, the input into the net is the concatenation of
        the source and target node features, possibly including
        dropout during training.

        Args:
            x (float tensor, (n_nodes, in_channels)): The node
                features.
            edge_index (int tensor, (2, n_edges)): The edges.

        Returns:
            edge_raw (float tensor, (n_edges)): The raw scores
                for each edge
        """
        edge_node_features = torch.cat(
            [x[edge_index[0]], x[edge_index[1]]],
            dim=1
        )
        edge_raw = self.net(edge_node_features)[:, 0]
        if self.dropout_prob and self.training:
            edge_raw = F.dropout(edge_raw, p=self.dropout_prob)
        return edge_raw

    def forward(self, x, edge_index, batch, return_unpool_info=False):
        """
        The forward computation for EdgePooling.

        This, in order, computes the raw edge score, normalizes it,
        and merges the edges.

        Args:
            x (float tensor, (n_nodes, in_channels)): The node
                features.
            edge_index (int tensor, (2, n_edges)): The edges.
            batch (int tensor, (n_nodes)): The batch each nodes
                is part of.
            return_unpool_info (bool, optional): If given, this
                will return unpool information that can be used
                to unpool this layer at a later time.

        Returns:
            x (float tensor, (n_nodes, in_channels)): The node
                features.
            edge_index (int tensor, (2, n_edges)): The edges.
            batch (int tensor, (n_nodes)): The batch each nodes
                is part of.
            unpool_info (unpool_description): Information that is
                consumed by `EdgePooling.unpool` for unpooling.

        Notes:
            It is important that you continue your following
            computations using the new edge_index and batch, since
            these have nothing to do with the old ones. If you want
            to unpool, set `return_unpool_info` and see
            `EdgePooling.unpool`.
        """
        raw_edge_score = self._compute_raw_edge_score(x, edge_index)
        edge_score = self.compute_edge_score(raw_edge_score, edge_index)
        edge_score = edge_score + self.add_to_edge_score
        x, edge_index, batch, unpool_info = self._merge_edges(
            x, edge_index, batch, edge_score
        )
        result = [x, edge_index, batch]
        if return_unpool_info:
            result.append(unpool_info)
        return result

    def _merge_edges(self, x, edge_index, batch, edge_score):
        """
        This takes a set of edges and their scores and merges them pairwisely.

        Args:
            x (float tensor, (n_nodes, in_channels)): The node
                features.
            edge_index (int tensor, (2, n_edges)): The edges.
            batch (int tensor, (n_nodes)): The batch each nodes
                is part of.
            edge_score (float tensor, (n_edges)): The normalized
                edge scores
        Returns:
            x (float tensor, (new_nodes, in_channels)): The node
                features.
            edge_index (int tensor, (2, new_edges)): The edges.
            batch (int tensor, (new_nodes)): The batch each nodes
                is part of.
            unpool_info (unpool_description): Information that is
                consumed by `EdgePooling.unpool` for unpooling.
        """
        nodes_remaining = set(range(len(x)))

        # This records, for each node, which merged node id it will
        # be a part of.
        old_to_new_node_idx = np.zeros(shape=(x.shape[0]),
                                       dtype=np.int64)
        node_counter = 0
        edges_chosen = []
        _, edge_ranking = torch.sort(edge_score, descending=True)
        np_edge_index = edge_index.cpu().numpy()

        # iterate through edges, choosing each if it is not incident
        # to another, already chosen, edge.
        for edge_idx in edge_ranking.cpu().numpy():
            from_node = np_edge_index[0, edge_idx]
            if from_node not in nodes_remaining:
                continue
            to_node = np_edge_index[1, edge_idx]
            if to_node not in nodes_remaining:
                continue
            edges_chosen.append(edge_idx)
            old_to_new_node_idx[from_node] = node_counter
            nodes_remaining.remove(from_node)
            if from_node != to_node:
                old_to_new_node_idx[to_node] = node_counter
                nodes_remaining.remove(to_node)
            node_counter += 1

        # the remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            old_to_new_node_idx[node_idx] = node_counter
            node_counter += 1
        old_to_new_node_idx = edge_index.new(old_to_new_node_idx)

        # and we compute the new features as an addition of the
        # old ones, times the edge score that led to its choice.
        new_x = scatter_add(
            src=x,
            index=old_to_new_node_idx,
            dim=0
        )

        if nodes_remaining:
            remaining_node_scores = torch.ones(
                (new_x.shape[0] - len(edges_chosen),),
                device=x.device
            )
            new_nodes_edge_scores = torch.cat(
                [edge_score[edges_chosen], remaining_node_scores]
            )
        else:
            new_nodes_edge_scores = edge_score[edges_chosen]

        new_x = new_x * new_nodes_edge_scores[:, None]

        new_edge_index = _compute_new_edges(
            old_to_new_node_idx,
            edge_index,
            n_nodes=new_x.shape[0]
        )

        # scatter_max is an arbitrary choice here; the used operation
        # is scatter_any since all of the batch numbers pointing to the
        # same new index will be identical.
        new_batch, _ = scatter_max(src=batch, index=old_to_new_node_idx)

        unpool_info = self.unpool_description(
            edge_index=edge_index,
            old_to_new_node_idx=old_to_new_node_idx,
            batch=batch,
            new_nodes_edge_scores=new_nodes_edge_scores
        )
        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(self, x, unpool_info):
        """
        Unpools a previous EdgePooling step.

        For unpooling, `x` should be the same shape (and so must be
        `edge_index` and `batch`) as those produced by this layer's
        `forward` function. Then, it will produce an unpooled `x`
        plus the `edge_index` and `batch`.

        Args:
            x (float tensor, (n_nodes, in_channels)): The current
                node features.
            unpool_info (unpool_description): Information that has
                been produced by `EdgePooling.forward`.


        Returns:
            x (float tensor, (new_n_nodes, in_channels)): The new
                node features.
            edge_index (int tensor, (2, n_edges)): The new edges.
            batch (int tensor, (n_nodes)): The batch each new node
                is part of.
        """
        x = x / unpool_info.new_nodes_edge_scores[:, None]
        new_x = x[unpool_info.old_to_new_node_idx]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{class_name}({edge_score_func}, {in_channels}, ' \
               'dropout={dropout_prob}, ' \
               'add_to_edge_score={add_to_edge_score})'.format(
                class_name=self.__class__.__name__,
                edge_score_func=self.compute_edge_score.__name__,
                in_channels=self.in_channels,
                dropout_prob=self.dropout_prob,
                add_to_edge_score=self.add_to_edge_score
                )
