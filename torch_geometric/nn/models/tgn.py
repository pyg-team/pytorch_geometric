import copy
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, GRUCell
from torch_scatter import scatter, scatter_max

from torch_geometric.nn.inits import zeros


class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable):
        super(TGNMemory, self).__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
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
        self.reset_message_store()

    def reset_message_store(self):
        i = self.memory.new_empty((0, ), dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim))
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            memory, last_update = self.get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src, dst, t, raw_msg):
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self.update_memory(n_id)
            self.update_message_store(src, dst, t, raw_msg, self.msg_s_store)
            self.update_message_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self.update_message_store(src, dst, t, raw_msg, self.msg_s_store)
            self.update_message_store(dst, src, t, raw_msg, self.msg_d_store)
            self.update_memory(n_id)

    def update_memory(self, n_id):
        """Updates the memory of the model for nodes in :obj:`n_id` using the
        messages stored in the message stores for these nodes."""
        memory, last_update = self.get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def get_updated_memory(self, n_id):
        """Returns the updated memory for nodes in :obj:`n_id` using the
        messages stored in the message stores for these nodes."""
        self.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self.compute_messages(
            n_id, self.msg_s_store, self.msg_s_module)

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self.compute_messages(
            n_id, self.msg_d_store, self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self.assoc[idx], t, dim_size=n_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        last_update = self.last_update.scatter(0, idx, t)[n_id]

        return memory, last_update

    def update_message_store(self, src, dst, t, raw_msg, msg_store):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def compute_messages(self, n_id, msg_store, msg_module):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        if self.training and not mode:  # Eval: Flush message store to memory.
            self.update_memory(
                torch.arange(self.num_nodes, device=self.memory.device))
            self.reset_message_store()
        super(TGNMemory, self).train(mode)

    def detach(self):
        self.memory.detach_()


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super(IdentityMessage, self).__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='mean')


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()


class LastNeighborLoader(object):
    def __init__(self, num_nodes: int, size: int, msg_dim: int, device=None):
        self.size = size

        self.src = torch.empty((num_nodes, size), dtype=torch.long,
                               device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long,
                                device=device)
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def __call__(self, n_id):
        src = self.src[n_id]
        dst = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        mask = e_id >= 0
        src, dst, e_id = src[mask], dst[mask], e_id[mask]

        n_id = torch.cat([n_id, src]).unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        src, dst = self.assoc[src], self.assoc[dst]

        return n_id, torch.stack([src, dst]), e_id

    def insert(self, src, dst):
        src, dst = torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0) // 2,
                            device=src.device).repeat(2)
        self.cur_e_id += src.numel() // 2

        dst, perm = dst.sort()
        src, e_id = src[perm], e_id[perm]

        n_id = dst.unique()
        self.assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(dst.size(0), device=dst.device) % self.size
        dense_id += self.assoc[dst].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size, ), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_src = e_id.new_empty(n_id.numel() * self.size)
        dense_src[dense_id] = src
        dense_src = dense_src.view(-1, self.size)

        # Collect new and old connections...
        e_id = torch.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        src = torch.cat([self.src[n_id, :self.size], dense_src], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.src[n_id] = torch.gather(src, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)
