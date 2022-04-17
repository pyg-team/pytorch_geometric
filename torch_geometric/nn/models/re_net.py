import math

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, Parameter
from torch_scatter import scatter_mean


class RENet(torch.nn.Module):
    r"""The Recurrent Event Network model from the `"Recurrent Event Network
    for Reasoning over Temporal Knowledge Graphs"
    <https://arxiv.org/abs/1904.05530>`_ paper

    .. math::
        f_{\mathbf{\Theta}}(\mathbf{e}_s, \mathbf{e}_r,
        \mathbf{h}^{(t-1)}(s, r))

    based on a RNN encoder

    .. math::
        \mathbf{h}^{(t)}(s, r) = \textrm{RNN}(\mathbf{e}_s, \mathbf{e}_r,
        g(\mathcal{O}^{(t)}_r(s)), \mathbf{h}^{(t-1)}(s, r))

    where :math:`\mathbf{e}_s` and :math:`\mathbf{e}_r` denote entity and
    relation embeddings, and :math:`\mathcal{O}^{(t)}_r(s)` represents the set
    of objects interacted with subject :math:`s` under relation :math:`r` at
    timestamp :math:`t`.
    This model implements :math:`g` as the **Mean Aggregator** and
    :math:`f_{\mathbf{\Theta}}` as a linear projection.

    Args:
        num_nodes (int): The number of nodes in the knowledge graph.
        num_rels (int): The number of relations in the knowledge graph.
        hidden_channels (int): Hidden size of node and relation embeddings.
        seq_len (int): The sequence length of past events.
        num_layers (int, optional): The number of recurrent layers.
            (default: :obj:`1`)
        dropout (float): If non-zero, introduces a dropout layer before the
            final prediction. (default: :obj:`0.`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """
    def __init__(self, num_nodes, num_rels, hidden_channels, seq_len,
                 num_layers=1, dropout=0., bias=True):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = dropout

        self.ent = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.rel = Parameter(torch.Tensor(num_rels, hidden_channels))

        self.sub_gru = GRU(3 * hidden_channels, hidden_channels, num_layers,
                           batch_first=True, bias=bias)
        self.obj_gru = GRU(3 * hidden_channels, hidden_channels, num_layers,
                           batch_first=True, bias=bias)

        self.sub_lin = Linear(3 * hidden_channels, num_nodes, bias=bias)
        self.obj_lin = Linear(3 * hidden_channels, num_nodes, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ent, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.rel, gain=math.sqrt(2.0))

        self.sub_gru.reset_parameters()
        self.obj_gru.reset_parameters()
        self.sub_lin.reset_parameters()
        self.obj_lin.reset_parameters()

    @staticmethod
    def pre_transform(seq_len):
        r"""Precomputes history objects

        .. math::
            \{ \mathcal{O}^{(t-k-1)}_r(s), \ldots, \mathcal{O}^{(t-1)}_r(s) \}

        of a :class:`torch_geometric.datasets.icews.EventDataset` with
        :math:`k` denoting the sequence length :obj:`seq_len`.
        """
        class PreTransform(object):
            def __init__(self, seq_len):
                self.seq_len = seq_len
                self.inc = 5000
                self.t_last = 0
                self.sub_hist = self.increase_hist_node_size([])
                self.obj_hist = self.increase_hist_node_size([])

            def increase_hist_node_size(self, hist):
                hist_inc = torch.zeros((self.inc, self.seq_len + 1, 0))
                return hist + hist_inc.tolist()

            def get_history(self, hist, node, rel):
                hists, ts = [], []
                for s in range(seq_len):
                    h = hist[node][s]
                    hists += h
                    ts.append(torch.full((len(h), ), s, dtype=torch.long))
                node, r = torch.tensor(hists, dtype=torch.long).view(
                    -1, 2).t().contiguous()
                node = node[r == rel]
                t = torch.cat(ts, dim=0)[r == rel]
                return node, t

            def step(self, hist):
                for i in range(len(hist)):
                    hist[i] = hist[i][1:]
                    hist[i].append([])
                return hist

            def __call__(self, data):
                sub, rel, obj, t = data.sub, data.rel, data.obj, data.t

                if max(sub, obj) + 1 > len(self.sub_hist):  # pragma: no cover
                    self.sub_hist = self.increase_hist_node_size(self.sub_hist)
                    self.obj_hist = self.increase_hist_node_size(self.obj_hist)

                # Delete last timestamp in history.
                if t > self.t_last:
                    self.sub_hist = self.step(self.sub_hist)
                    self.obj_hist = self.step(self.obj_hist)
                    self.t_last = t

                # Save history in data object.
                data.h_sub, data.h_sub_t = self.get_history(
                    self.sub_hist, sub, rel)
                data.h_obj, data.h_obj_t = self.get_history(
                    self.obj_hist, obj, rel)

                # Add new event to history.
                self.sub_hist[sub][-1].append([obj, rel])
                self.obj_hist[obj][-1].append([sub, rel])

                return data

            def __repr__(self) -> str:  # pragma: no cover
                return f'{self.__class__.__name__}(seq_len={self.seq_len})'

        return PreTransform(seq_len)

    def forward(self, data):
        """Given a :obj:`data` batch, computes the forward pass.

        Args:
            data (torch_geometric.data.Data): The input data, holding subject
                :obj:`sub`, relation :obj:`rel` and object :obj:`obj`
                information with shape :obj:`[batch_size]`.
                In addition, :obj:`data` needs to hold history information for
                subjects, given by a vector of node indices :obj:`h_sub` and
                their relative timestamps :obj:`h_sub_t` and batch assignments
                :obj:`h_sub_batch`.
                The same information must be given for objects (:obj:`h_obj`,
                :obj:`h_obj_t`, :obj:`h_obj_batch`).
        """

        assert 'h_sub_batch' in data and 'h_obj_batch' in data
        batch_size, seq_len = data.sub.size(0), self.seq_len

        h_sub_t = data.h_sub_t + data.h_sub_batch * seq_len
        h_obj_t = data.h_obj_t + data.h_obj_batch * seq_len

        h_sub = scatter_mean(self.ent[data.h_sub], h_sub_t, dim=0,
                             dim_size=batch_size * seq_len).view(
                                 batch_size, seq_len, -1)
        h_obj = scatter_mean(self.ent[data.h_obj], h_obj_t, dim=0,
                             dim_size=batch_size * seq_len).view(
                                 batch_size, seq_len, -1)

        sub = self.ent[data.sub].unsqueeze(1).repeat(1, seq_len, 1)
        rel = self.rel[data.rel].unsqueeze(1).repeat(1, seq_len, 1)
        obj = self.ent[data.obj].unsqueeze(1).repeat(1, seq_len, 1)

        _, h_sub = self.sub_gru(torch.cat([sub, h_sub, rel], dim=-1))
        _, h_obj = self.obj_gru(torch.cat([obj, h_obj, rel], dim=-1))
        h_sub, h_obj = h_sub.squeeze(0), h_obj.squeeze(0)

        h_sub = torch.cat([self.ent[data.sub], h_sub, self.rel[data.rel]],
                          dim=-1)
        h_obj = torch.cat([self.ent[data.obj], h_obj, self.rel[data.rel]],
                          dim=-1)

        h_sub = F.dropout(h_sub, p=self.dropout, training=self.training)
        h_obj = F.dropout(h_obj, p=self.dropout, training=self.training)

        log_prob_obj = F.log_softmax(self.sub_lin(h_sub), dim=1)
        log_prob_sub = F.log_softmax(self.obj_lin(h_obj), dim=1)

        return log_prob_obj, log_prob_sub

    def test(self, logits, y):
        """Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)
        and Hits at 1/3/10."""

        _, perm = logits.sort(dim=1, descending=True)
        mask = (y.view(-1, 1) == perm)

        nnz = mask.nonzero(as_tuple=False)
        mrr = (1 / (nnz[:, -1] + 1).to(torch.float)).mean().item()
        hits1 = mask[:, :1].sum().item() / y.size(0)
        hits3 = mask[:, :3].sum().item() / y.size(0)
        hits10 = mask[:, :10].sum().item() / y.size(0)

        return torch.tensor([mrr, hits1, hits3, hits10])
