import torch
from torch import LongTensor as Long
from torch_geometric.utils import one_hot, coalesce
from torch_geometric.data import Data

elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def parse_sdf(src):
    counts_line = src[0].split()
    num_atoms, num_bonds = int(counts_line[0]), int(counts_line[1])

    atom_block = [x for x in src[1:num_atoms + 1]]
    atom_block = [x.split()[:4] for x in atom_block]
    pos = torch.Tensor([[float(y) for y in x[:3]] for x in atom_block])
    x = one_hot(Long([elems[x[3]] for x in atom_block]), len(elems))

    bond_block = [x for x in src[1 + num_atoms:1 + num_atoms + num_bonds]]
    bond_block = Long([[int(y) for y in x.split()[:3]] for x in bond_block])
    edge_index = bond_block[:, :2] - 1
    edge_index, perm = coalesce(edge_index.t())
    edge_index = edge_index.t().contiguous()
    edge_attr = bond_block[:, 2][perm] - 1

    return Data(x, edge_index, edge_attr, None, pos)
