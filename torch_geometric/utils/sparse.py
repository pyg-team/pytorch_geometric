import torch
from torch_sparse import coalesce

from .num_nodes import maybe_num_nodes


def dense_to_sparse(tensor):
    index = tensor.nonzero().t().contiguous()
    value = tensor[index[0], index[1]]
    index, value = coalesce(index, value, tensor.size(0), tensor.size(1))
    return index, value


def sparse_to_dense(edge_index, edge_attr, num_nodes=None):
    N = maybe_num_nodes(edge_index, num_nodes)

    adj = torch.sparse_coo_tensor(edge_index, edge_attr, torch.Size([N, N]))
    return adj.to_dense()
