import copy
from typing import Callable

import torch
from torch.nn import Sequential, Linear, ReLU, GRUCell
from torch_scatter import scatter_max

from torch_geometric.nn.inits import reset, zeros


class TGN(torch.nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int, time_dim: int,
                 message_module: Callable, aggregator_module: Callable,
                 embedding_module: Callable):
        super(TGN, self).__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.message_module_src = message_module
        self.message_module_dst = copy.deepcopy(message_module)
        self.aggregator_module = aggregator_module
        self.embedding_module = embedding_module

        self.time_enc = Linear(1, time_dim)
        self.gru = GRUCell(self.message_module_dst.out_channels, memory_dim)

        embedding_dim = self.embedding_module.out_channels
        self.link_pred = Sequential(
            Linear(2 * embedding_dim, embedding_dim),
            ReLU(),
            Linear(embedding_dim, 1),
        )

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.empty(num_nodes))
        self.previous_events = None

        self.reset_parameters()

    def reset_parameters(self):
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        reset(self.link_pred)
        self.reset_memory_()

    def forward(self, src, pos_dst, neg_dst, t, raw_msg):
        if self.previous_events is not None:
            self.update_memory_(*self.previous_events)

        emb_src = self.embedding_module(self.memory[src], t)
        emb_pos_dst = self.embedding_module(self.memory[pos_dst], t)
        emb_neg_dst = self.embedding_module(self.memory[neg_dst], t)

        self.previous_events = (src, pos_dst, t, raw_msg)

        pos_out = self.link_pred(torch.cat([emb_src, emb_pos_dst], dim=-1))
        neg_out = self.link_pred(torch.cat([emb_src, emb_neg_dst], dim=-1))

        return pos_out, neg_out

    def reset_memory_(self):
        zeros(self.memory)
        zeros(self.last_update)

    def update_memory_(self, src, dst, t, raw_msg):
        # Compute message.
        t_rel_src = (t - self.last_update[src]).view(-1, 1).to(raw_msg.dtype)
        t_src_enc = self.time_enc(t_rel_src).cos()

        t_rel_dst = (t - self.last_update[dst]).view(-1, 1).to(raw_msg.dtype)
        t_dst_enc = self.time_enc(t_rel_dst).cos()

        msg_src = self.message_module_src(self.memory[src], self.memory[dst],
                                          raw_msg, t_src_enc)
        msg_dst = self.message_module_dst(self.memory[dst], self.memory[src],
                                          raw_msg, t_dst_enc)

        # Aggregate messages.
        idx, msg_aggr = self.aggregator_module(
            torch.cat([msg_src, msg_dst], dim=0),
            torch.cat([src, dst], dim=0),
            torch.cat([t, t], dim=0),
        )

        self.last_update[src] = t
        self.last_update[dst] = t

        # Update memory.
        self.memory[idx] = self.gru(msg_aggr, self.memory[idx])

    def detach_memory_(self):
        self.memory.detach_()


class IdentityMessage(torch.nn.Module):
    def __init__(self, memory_dim, raw_msg_dim, time_dim):
        super(IdentityMessage, self).__init__()
        self.out_channels = 2 * memory_dim + raw_msg_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg, index, t):
        _, argmax = scatter_max(t, index, dim=0)
        index = index.unique()
        return index, msg[argmax[index]]


class IdentityEmbedding(torch.nn.Module):
    def __init__(self, memory_dim):
        super(IdentityEmbedding, self).__init__()
        self.in_channels = memory_dim
        self.out_channels = memory_dim

    def forward(self, memory, t):
        return memory
