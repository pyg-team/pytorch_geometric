from torch_geometric.read import split2d
from torch import LongTensor as Long
from torch_geometric.utils import one_hot, coalesce
from torch_geometric.data import Data

elements = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def parse_sdf(src):
    counts_line = src[0].split()
    num_atoms, num_bonds = int(counts_line[0]), int(counts_line[1])

    atom_block = src[1:num_atoms + 1]
    pos = split2d(atom_block, end=3)
    x = split2d(atom_block, lambda x: elements[x], start=3, end=4, out=Long())
    x = one_hot(x, len(elements))

    bond_block = src[1 + num_atoms:1 + num_atoms + num_bonds]
    edge_index = split2d(bond_block, end=2, out=Long()) - 1
    edge_index, perm = coalesce(edge_index.t())
    edge_index = edge_index.t().contiguous()
    edge_attr = split2d(bond_block, start=2, end=3, out=Long())[perm] - 1

    return Data(x, edge_index, edge_attr, None, pos)
