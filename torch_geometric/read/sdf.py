import torch
from torch_geometric.read import parse_txt_array
from torch_geometric.utils import one_hot, coalesce
from torch_geometric.data import Data

elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def parse_sdf(src):
    counts_line = src[0].split()
    num_atoms, num_bonds = int(counts_line[0]), int(counts_line[1])

    atom_block = src[1:num_atoms + 1]
    pos = parse_txt_array(atom_block, end=3)
    x = parse_txt_array(
        atom_block, lambda x: elems[x], start=3, end=4, dtype=torch.long)
    x = one_hot(x, len(elems))

    bond_block = src[1 + num_atoms:1 + num_atoms + num_bonds]
    edge_index = parse_txt_array(bond_block, end=2, dtype=torch.long) - 1
    edge_attr = parse_txt_array(bond_block, start=2, end=3, dtype=torch.long)
    edge_index, edge_attr = coalesce(edge_index.t(), edge_attr - 1)

    return Data(x=x, edge=edge_index, edge_attr=edge_attr, pos=pos)
