import re

import torch
from torch._tensor_str import PRINT_OPTS, _tensor_str
from torch_geometric.io import parse_txt_array
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
    face = face_to_tri(face)

    data = Data(pos=pos)
    data.face = face

    return data


def face_to_tri(face):
    face = [[int(x) for x in line.strip().split()] for line in face]

    triangle = torch.tensor([line[1:] for line in face if line[0] == 3])
    triangle = triangle.to(torch.int64)

    rect = torch.tensor([line[1:] for line in face if line[0] == 4])
    rect = rect.to(torch.int64)

    if rect.numel() > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]
        return torch.cat([triangle, first, second], dim=0).t().contiguous()
    else:
        return triangle.t().contiguous()


def read_off(path):
    r"""Reads an OFF (Object File Format) file, returning both the position of
    nodes and their connectivity in a :class:`torch_geometric.data.Data`
    object.

    Args:
        path (str): The path to the file.
    """
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_off(src)


def write_off(data, path):
    r"""Writes a :class:`torch_geometric.data.Data` object to an OFF (Object
    File Format) file.

    Args:
        data (:class:`torch_geometric.data.Data`): The data object.
        path (str): The path to the file.
    """
    num_nodes, num_faces = data.pos.size(0), data.face.size(1)

    pos = data.pos.to(torch.float)
    face = data.face.t()
    num_vertices = torch.full((num_faces, 1), face.size(1), dtype=torch.long)
    face = torch.cat([num_vertices, face], dim=-1)

    threshold = PRINT_OPTS.threshold
    torch.set_printoptions(threshold=float('inf'))

    pos_repr = re.sub(',', '', _tensor_str(pos, indent=0))
    pos_repr = '\n'.join([x[2:-1] for x in pos_repr.split('\n')])[:-1]

    face_repr = re.sub(',', '', _tensor_str(face, indent=0))
    face_repr = '\n'.join([x[2:-1] for x in face_repr.split('\n')])[:-1]

    with open(path, 'w') as f:
        f.write(f'OFF\n{num_nodes} {num_faces} 0\n')
        f.write(pos_repr)
        f.write('\n')
        f.write(face_repr)
        f.write('\n')
    torch.set_printoptions(threshold=threshold)
