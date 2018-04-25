from torch_geometric.read import parse_txt
from torch import LongTensor as Long
from torch_geometric.utils import one_hot, coalesce
from torch_geometric.data import Data

elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def parse_sdf(src):
    counts_line = src[0].split()
    num_atoms, num_bonds = int(counts_line[0]), int(counts_line[1])

    atom_block = src[1:num_atoms + 1]
    pos = parse_txt(atom_block, end=3)
    x = parse_txt(atom_block, lambda x: elems[x], start=3, end=4, out=Long())
    x = one_hot(x, len(elems))

    bond_block = src[1 + num_atoms:1 + num_atoms + num_bonds]
    edge_index = parse_txt(bond_block, end=2, out=Long()) - 1
    edge_index, perm = coalesce(edge_index.t())
    edge_attr = parse_txt(bond_block, start=2, end=3, out=Long())[perm] - 1

    return Data(x, edge_index, edge_attr, None, pos)
