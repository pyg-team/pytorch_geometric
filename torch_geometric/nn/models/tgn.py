import copy
from typing import Callable, List

import torch
from torch.nn import Sequential, Linear, ReLU, GRUCell
from torch_scatter import scatter_max, scatter_mean
from torch_sparse import SparseTensor

from torch_geometric.nn import MessagePassing
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
        self.register_buffer('last_update',
                             torch.empty(num_nodes, dtype=torch.long))

        self.previous_batch = None
        self.current_event_id = 0

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.message_module_src, 'reset_parameters'):
            self.message_module_src.reset_parameters()
        if hasattr(self.message_module_dst, 'reset_parameters'):
            self.message_module_dst.reset_parameters()
        if hasattr(self.aggregator_module, 'reset_parameters'):
            self.aggregator_module.reset_parameters()
        if hasattr(self.embedding_module, 'reset_parameters'):
            self.embedding_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        reset(self.link_pred)
        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        self.previous_batch = None
        self.current_event_id = 0

    def forward(self, src, pos_dst, neg_dst, t, raw_msg):
        # Update memory to incorporate the previous batch.
        if self.previous_batch is not None:
            self.update_memory(*self.previous_batch)

        # Embed current events.
        emb_src = self.embedding_module(self.memory, src, t,
                                        self.current_event_id)
        emb_pos_dst = self.embedding_module(self.memory, pos_dst, t,
                                            self.current_event_id)
        emb_neg_dst = self.embedding_module(self.memory, neg_dst, t,
                                            self.current_event_id)

        # Update state.
        self.update_state(src, pos_dst, t, raw_msg)

        # Perform final link prediction.
        pos_out = self.link_pred(torch.cat([emb_src, emb_pos_dst], dim=-1))
        neg_out = self.link_pred(torch.cat([emb_src, emb_neg_dst], dim=-1))

        return pos_out, neg_out

    def update_memory(self, src, dst, t, raw_msg):
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

    def update_state(self, src, dst, t, raw_msg):
        self.previous_batch = (src, dst, t, raw_msg)
        self.current_event_id += src.size(0)

    def detach_memory(self):
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


class MeanAggregator(torch.nn.Module):
    def forward(self, msg, index, t):
        out = scatter_mean(msg, index, dim=0)
        index = index.unique()
        return index, out[index]


class IdentityEmbedding(torch.nn.Module):
    def __init__(self, memory_dim):
        super(IdentityEmbedding, self).__init__()
        self.in_channels = memory_dim
        self.out_channels = memory_dim

    def forward(self, memory, index, t, current_event_id):
        return memory[index]


class NeighborSampler(object):
    def __init__(self, src, dst, t, raw_msg, sizes: List[int]):
        row = torch.cat([src, dst], dim=0).cpu()
        col = torch.cat([dst, src], dim=0).cpu()
        e_id = torch.arange(src.size(0)).repeat(2)

        self.adj = SparseTensor(row=row, col=col, value=e_id)
        self.t = t
        self.raw_msg = raw_msg
        self.sizes = sizes

    def sample(self, batch, idx_cutoff):
        _, _, e_id = self.adj.coo()
        adj = self.adj.masked_select_nnz(e_id < idx_cutoff, layout='coo')

        adjs = []
        n_id = batch.cpu()
        batch_size = batch.numel()
        for size in self.sizes:
            if adj.nnz() > 0:
                adj_sub, n_id = adj.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_sub.coo()
                size = adj.sparse_sizes()
                edge_index = torch.stack([row, col], dim=0)
                # t = self.t[e_id]
                raw_msg = self.raw_msg[e_id]
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                # t = torch.empty((0, ), dtype=torch.long)
                raw_msg = torch.empty((0, 172))
                size = [batch_size, batch_size]
            adjs.append((edge_index, raw_msg, size))

        return batch_size, n_id, adjs[::-1]


class TemporalGraphMeanConv(MessagePassing):
    def __init__(self, in_channels, raw_msg_dim, time_dim, out_channels):
        super(TemporalGraphMeanConv, self).__init__(aggr='mean')

        self.time_enc = Linear(1, time_dim)
        self.lin = Linear(in_channels + raw_msg_dim, out_channels)

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(out_channels, out_channels)

    def reset_parameters(self):
        self.time_enc.reset_parameters()
        self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        # print(x[0].shape, x[1].shape)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # print('out', out.shape)
        return self.lin_l(x[1]) + self.lin_r(out.relu())

    def message(self, x_j, edge_attr):
        # t_j = self.time_enc(t_i - t_j).cos()
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))


class TemporalGraphMeanGNN(torch.nn.Module):
    def __init__(self, memory_dim, raw_msg_dim, time_dim, out_channels,
                 sampler):
        super(TemporalGraphMeanGNN, self).__init__()

        self.out_channels = out_channels
        self.sampler = sampler

        self.convs = torch.nn.ModuleList()
        for i in range(len(sampler.sizes)):
            in_channels = memory_dim if i == 0 else out_channels
            conv = TemporalGraphMeanConv(in_channels, raw_msg_dim, time_dim,
                                         out_channels)
            self.convs.append(conv)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, memory, index, t, current_event_id):
        batch_size, n_id, adjs = self.sampler.sample(index, current_event_id)

        x = memory[n_id]
        for conv, (edge_index, raw_msg, size) in zip(self.convs, adjs):
            edge_index = edge_index.to(memory.device)
            t = t.to(memory.device)
            raw_msg = raw_msg.to(memory.device)
            x = conv((x, x[:size[1]]), edge_index, raw_msg)
            x = x.relu()

        return x
