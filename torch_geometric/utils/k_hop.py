from torch_sparse import SparseTensor


def k_hop(node_idx, num_hops, edge_index, edge_attr=None, relabel_nodes=False,
          num_nodes=None):

    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr)
