import copy
from typing import Callable, List

import torch
from torch.nn import Sequential, Linear, ReLU, GRUCell
from torch_scatter import scatter_max, scatter_mean
from torch_sparse import SparseTensor
from torch_geometric.data import TemporalData

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset, zeros


class TGN(torch.nn.Module):
    def __init__(self, data: TemporalData, memory_dim: int, time_dim: int,
                 message_module: Callable, aggregator_module: Callable,
                 embedding_module: Callable):
        super(TGN, self).__init__()

        self.data = data
        num_nodes = data.num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.message_s_module = message_module
        self.message_d_module = copy.deepcopy(message_module)
        self.aggregator_module = aggregator_module
        self.embedding_module = embedding_module

        self.time_enc = Linear(1, time_dim)
        self.gru = GRUCell(message_module.out_channels, memory_dim)

        embedding_dim = self.embedding_module.out_channels
        self.link_pred = Sequential(
            Linear(2 * embedding_dim, embedding_dim),
            ReLU(),
            Linear(embedding_dim, 1),
        )

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(num_nodes, dtype=torch.long)
        self.register_buffer('last_update', last_update)

        self.current_event_id = 0
        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.message_s_module, 'reset_parameters'):
            self.message_s_module.reset_parameters()
        if hasattr(self.message_d_module, 'reset_parameters'):
            self.message_d_module.reset_parameters()
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
        self.current_event_id = 0
        empty = torch.tensor([], dtype=torch.long, device=self.data.src.device)
        num_nodes = self.data.num_nodes
        self.msg_s_store = {i: empty for i in range(num_nodes)}
        self.msg_d_store = {i: empty for i in range(num_nodes)}

    def get_updated_memory(self, n_id):
        nodes = n_id.tolist()

        # Get previous raw messages (src->dst) involving all nodes in `n_id`.
        idx = torch.cat([self.msg_s_store[i] for i in nodes], dim=0)
        src1 = self.data.src[idx].to(n_id.device)
        dst1 = self.data.dst[idx].to(n_id.device)
        t1 = self.data.t[idx].to(n_id.device)
        msg = self.data.x[idx].to(n_id.device)
        t_enc = self.time_enc(
            (self.last_update[idx] - t1).to(msg.dtype).view(-1, 1))

        # Compute `msg_s` via message function.
        msg_s = self.message_s_module(
            self.memory[src1],
            self.memory[dst1],
            msg,
            t_enc,
        )

        # Get previous raw messages (dst->src) involving all nodes in `n_id`.
        idx = torch.cat([self.msg_d_store[i] for i in nodes], dim=0)
        dst2 = self.data.dst[idx].to(n_id.device)
        src2 = self.data.src[idx].to(n_id.device)
        t2 = self.data.t[idx].to(n_id.device)
        msg = self.data.x[idx].to(n_id.device)
        t_enc = self.time_enc(
            (self.last_update[idx] - t2).to(msg.dtype).view(-1, 1))

        # Compute `msg_d` via message function.
        msg_d = self.message_d_module(
            self.memory[dst2],
            self.memory[src2],
            msg,
            t_enc,
        )

        # Aggregate messages.
        idx = torch.cat([src1, dst2], dim=0)
        idx, perm = idx.unique(return_inverse=True)

        # Aggregate messages.
        msg_aggr = self.aggregator_module(torch.cat([msg_s, msg_d], dim=0),
                                          perm, torch.cat([t1, t2], dim=0))

        # Update "local" memory (we do not push to the real memory yet).
        memory = self.memory.clone()
        memory[idx] = self.gru(msg_aggr, memory[idx])
        memory = memory[n_id]

        return memory

    def forward(self, src, pos_dst, neg_dst, t, raw_msg):
        batch = torch.cat([src, pos_dst, neg_dst], dim=0).unique()
        print(batch.shape)

        memory = self.get_updated_memory(batch)  # Get local copy of memory.
        # print(memory.shape)

        # raise NotImplementedError

        # Embed current events.
        # emb_src = self.embedding_module(self.memory, src, t,
        #                                 self.current_event_id)
        # emb_pos_dst = self.embedding_module(self.memory, pos_dst, t,
        #                                     self.current_event_id)
        # emb_neg_dst = self.embedding_module(self.memory, neg_dst, t,
        #                                     self.current_event_id)

        # Update state and memory.
        self.update_state(src, pos_dst)

        # # Perform final link prediction.
        # pos_out = self.link_pred(torch.cat([emb_src, emb_pos_dst], dim=-1))
        # neg_out = self.link_pred(torch.cat([emb_src, emb_neg_dst], dim=-1))

        # return pos_out, neg_out

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

    def update_state(self, src, dst):
        event_id = torch.arange(src.size(0)) + self.current_event_id
        self.current_event_id += src.size(0)

        src, perm = src.sort()
        mask = src[1:] != src[:-1]
        nnz = mask.nonzero(as_tuple=False).add_(1).view(-1).tolist()
        nnz = [0] + nnz + [src.size(0)]
        nnz = torch.tensor(nnz)
        split = nnz[1:] - nnz[:-1]
        splits = split.tolist()

        for i, idx in zip(src.unique_consecutive().tolist(),
                          event_id[perm].split(splits)):
            self.msg_s_store[i] = idx

        dst, perm = dst.sort()
        mask = dst[1:] != dst[:-1]
        nnz = mask.nonzero(as_tuple=False).add_(1).view(-1).tolist()
        nnz = [0] + nnz + [dst.size(0)]
        nnz = torch.tensor(nnz)
        split = nnz[1:] - nnz[:-1]
        splits = split.tolist()

        for i, idx in zip(dst.unique_consecutive().tolist(),
                          event_id[perm].split(splits)):
            self.msg_d_store[i] = idx

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
        return msg[argmax]


class MeanAggregator(torch.nn.Module):
    def forward(self, msg, index, t):
        return scatter_mean(msg, index, dim=0)


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
