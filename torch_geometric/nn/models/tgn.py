import copy
from typing import Callable

import torch
from torch.nn import Linear, GRUCell
from torch_scatter import scatter_max, scatter_mean

from torch_geometric.nn.inits import zeros


class TGN(torch.nn.Module):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable):
        super(TGN, self).__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module

        self.time_enc = Linear(1, time_dim)
        self.gru = GRUCell(message_module.out_channels, memory_dim)

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer('last_update', last_update)
        self.register_buffer('assoc', torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        i = self.memory.new_empty((0, ), dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim))
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def forward(self, n_id):
        self.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages from src->dst.
        data = [self.msg_s_store[i] for i in n_id.tolist()]
        src_s, dst_s, t_s, raw_msg_s = list(zip(*data))
        src_s = torch.cat(src_s, dim=0)
        dst_s = torch.cat(dst_s, dim=0)
        t_s = torch.cat(t_s, dim=0)
        raw_msg_s = torch.cat(raw_msg_s, dim=0)
        t_rel_s = t_s - self.last_update[src_s]
        t_enc_s = self.time_enc(t_rel_s.to(raw_msg_s.dtype).view(-1, 1)).cos()

        msg_s = self.msg_s_module(self.memory[src_s], self.memory[dst_s],
                                  raw_msg_s, t_enc_s)

        # Compute messages from dst->src.
        data = [self.msg_d_store[i] for i in n_id.tolist()]
        dst_d, src_d, t_d, raw_msg_d = list(zip(*data))
        dst_d = torch.cat(dst_d, dim=0)
        src_d = torch.cat(src_d, dim=0)
        t_d = torch.cat(t_d, dim=0)
        raw_msg_d = torch.cat(raw_msg_d, dim=0)
        t_rel_d = t_d - self.last_update[dst_d]
        t_enc_d = self.time_enc(t_rel_d.to(raw_msg_d.dtype).view(-1, 1)).cos()

        msg_d = self.msg_d_module(self.memory[dst_d], self.memory[src_d],
                                  raw_msg_d, t_enc_d)

        # Aggregate messages.
        idx = torch.cat([src_s, dst_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self.assoc[idx], t, dim_size=n_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        last_update = self.last_update.scatter(0, idx, t)[n_id]

        return memory, last_update

    def update_state(self, src, dst, t, raw_msg):
        # Update memory.
        n_id = torch.cat([src, dst]).unique()
        memory, last_update = self(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

        # Update message stores.
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            self.msg_s_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

        n_id, perm = dst.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            self.msg_d_store[i] = (dst[idx], src[idx], t[idx], raw_msg[idx])

    def detach_memory(self):
        self.memory.detach_()


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        super(IdentityMessage, self).__init__()
        self.out_channels = 2 * memory_dim + raw_msg_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        return scatter_mean(msg, index, dim=0, dim_size=dim_size)
