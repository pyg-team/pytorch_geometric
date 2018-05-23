import torch
from torch_geometric.read import parse_txt_array
from torch_geometric.data import Data


def parse_off(src):
    # Some files may contain a bug and do not have a carriage return after OFF.
    if src[0] == 'OFF':
        src = src[1:]
    else:
        src[0] = src[0][3:]

    num_nodes, num_faces = [int(item) for item in src[0].split()[:2]]

    pos = parse_txt_array(src[1:1 + num_nodes])

    face = src[1 + num_nodes:1 + num_nodes + num_faces]
    face = parse_txt_array(face, start=1, dtype=torch.long).t().contiguous()

    data = Data(pos=pos)
    data.face = face

    return data


def read_off(path):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_off(src)
