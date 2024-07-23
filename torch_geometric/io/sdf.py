import torch

from torch_geometric.data import Data
from torch_geometric.io import parse_txt_array
from torch_geometric.utils import coalesce, one_hot

elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def parse_sdf(src: str) -> Data:
    lines = src.split('\n')[3:]
    num_atoms, num_bonds = (int(item) for item in lines[0].split()[:2])

    atom_block = lines[1:num_atoms + 1]
    pos = parse_txt_array(atom_block, end=3)
    x = torch.tensor([elems[item.split()[3]] for item in atom_block])
    x = one_hot(x, num_classes=len(elems))

    bond_block = lines[1 + num_atoms:1 + num_atoms + num_bonds]
    row, col = parse_txt_array(bond_block, end=2, dtype=torch.long).t() - 1
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = parse_txt_array(bond_block, start=2, end=3) - 1
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_atoms)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def read_sdf(path: str) -> Data:
    with open(path) as f:
        return parse_sdf(f.read())
