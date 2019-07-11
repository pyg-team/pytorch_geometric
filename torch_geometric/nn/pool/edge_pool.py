from torch import nn
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_mean
import numpy as np
from torch_sparse import coalesce

DEBUG = False


def _compute_new_edges(old_to_new_node_idx, edge_index, n_nodes):
    new_edge_index = torch.stack([
        old_to_new_node_idx[edge_index[0]],
        old_to_new_node_idx[edge_index[1]]
    ])
    new_edge_index, _ = coalesce(new_edge_index, None, n_nodes, n_nodes)
    return new_edge_index


class EdgePoolLayer(nn.Module):

    EDGE_SCORE_SOFTMAX = "softmax"
    EDGE_SCORE_TANH = "tanh"
    EDGE_SCORE_SIGMOID = "sigmoid"

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index):
        max_per_node = scatter_max(raw_edge_score, edge_index[1], fill_value=-np.inf)[0]
        edge_logits = raw_edge_score - max_per_node[edge_index[1]]
        edge_exps = torch.exp(edge_logits)
        edge_exps_by_node = scatter_add(edge_exps, edge_index[1])
        edge_score = edge_exps / edge_exps_by_node[edge_index[1]]
        return edge_score

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index):
        edge_score = torch.tanh(raw_edge_score)
        return edge_score

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index):
        edge_score = torch.sigmoid(raw_edge_score)
        return edge_score

    def __init__(self,
                 in_channels,
                 edge_score_method=compute_edge_score_softmax,
                 dropout_prob=False,
                 add_to_edge_score=0.5
                 ):
        super().__init__()
        self.build_network(in_channels)
        self.dropout_prob = dropout_prob
        self.add_to_edge_score = add_to_edge_score
        self.compute_edge_score = edge_score_method

    def build_network(self, in_channels):
        self.fc1 = nn.Linear(2 * in_channels, 1)

    def _compute_raw_edge_score(self, x, edge_index):
        edge_node_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_raw = self.fc1(edge_node_features)[:, 0]
        if self.dropout_prob and self.training:
            edge_raw = F.dropout(edge_raw, p=self.dropout_prob)
        return edge_raw

    def forward(self, x, edge_index, batch, return_unpool_info=False):
        raw_edge_score = self._compute_raw_edge_score(x, edge_index)
        edge_score = self.edge_score_method(raw_edge_score, edge_index)
        edge_score = edge_score + self.add_to_edge_score
        x, edge_index, batch, unpool_info, edges_chosen = self._merge_edges(
            x, edge_index, batch, edge_score
        )
        result = [x, edge_index, batch]
        if return_unpool_info:
            result.append(unpool_info)
        return result

    def _merge_edges(self, x, edge_index, batch, edge_score):
        nodes_remaining = set(range(len(x)))

        old_to_new_node_idx = np.zeros(shape=(x.shape[0]),
                                       dtype=np.int64)
        node_counter = 0
        edges_chosen = []
        _, edge_ranking = torch.sort(edge_score, descending=True)
        np_edge_index = edge_index.cpu().numpy()

        for edge_idx in edge_ranking.cpu().numpy():
            from_node = np_edge_index[0, edge_idx]
            if from_node not in nodes_remaining:
                continue
            to_node = np_edge_index[1, edge_idx]
            if to_node not in nodes_remaining:
                # we cannot merge this, because one of the nodes has already been merged
                continue
            edges_chosen.append(edge_idx)
            old_to_new_node_idx[from_node] = node_counter
            nodes_remaining.remove(from_node)
            if from_node != to_node:
                old_to_new_node_idx[to_node] = node_counter
                nodes_remaining.remove(to_node)
            node_counter += 1

        for node_idx in nodes_remaining:
            old_to_new_node_idx[node_idx] = node_counter
            node_counter += 1
        old_to_new_node_idx = edge_index.new(old_to_new_node_idx)

        new_x = self.scatter_add(
            src=x,
            index=old_to_new_node_idx,
            dim=0
        )

        if nodes_remaining:
            remaining_node_scores = torch.ones((new_x.shape[0] - len(edges_chosen),), device = x.device)
            new_nodes_edge_scores = torch.cat(
                [edge_score[edges_chosen], remaining_node_scores]
            )
        else:
            new_nodes_edge_scores = edge_score[edges_chosen]

        if nodes_remaining and self.add_self_loops:
            raise NotImplementedError("With self-loops, none should be remaining")

        new_x = new_x * new_nodes_edge_scores[:, None]

        new_edge_index = _compute_new_edges(old_to_new_node_idx, edge_index, n_nodes=new_x.shape[0])
        new_batch, _ = scatter_max(src=batch, index=old_to_new_node_idx)

        unpool_info = [edge_index, old_to_new_node_idx, batch, new_nodes_edge_scores]
        return new_x, new_edge_index, new_batch, unpool_info, edges_chosen

    def unpool(self, x, unpool_info):
        old_edge_index, old_to_new_node_idx, batch, new_nodes_edge_scores = unpool_info
        x = x / new_nodes_edge_scores[:, None]
        new_x = x[old_to_new_node_idx]
        return new_x, old_edge_index, batch
