import torch

def maybe_num_nodes(index, num_nodes=None):
    return len(torch.unique(torch.cat((index[0,:], index[1,:])))) if num_nodes is None else num_nodes
