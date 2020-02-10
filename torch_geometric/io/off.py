import re

import pandas
import torch
from torch._tensor_str import PRINT_OPTS
from torch_geometric.data import Data

REGEX = r'[\[\]\(\),]|tensor|(\s)\1{2,}'


def face_to_tri(face):
    if face.size(0) == 4:
        first, second = face[:3, :], face[1:, :]
        return torch.cat([first, second], dim=1)
    else:
        return face


def read_off(path):
    r"""Reads an OFF (Object File Format) file, returning both the position of
    nodes and their connectivity in a :class:`torch_geometric.data.Data`
    object.

    Args:
        path (str): The path to the file.
    """
    skiprows = 1
    with open(path, 'r') as f:
        content = f.readline()[:-1]
        # Some files contain a bug and do not have a carriage return after OFF.
        if content == 'OFF':
            skiprows += 1
            content = f.readline()[:-1]
        else:
            content = content[3:-1]

    num_nodes, num_faces = [int(item) for item in content.split()[:2]]

    pos = pandas.read_csv(path, sep=' ', header=None, skiprows=skiprows,
                          nrows=num_nodes)
    pos = torch.from_numpy(pos.to_numpy()).to(torch.float)

    face = pandas.read_csv(path, sep=' ', header=None,
                           skiprows=skiprows + num_nodes, nrows=num_faces)
    face = torch.from_numpy(face.to_numpy())[:, 1:].to(torch.long).t()
    face = face_to_tri(face)

    return Data(pos=pos, face=face)


def write_off(data, path):
    r"""Writes a :class:`torch_geometric.data.Data` object to an OFF (Object
    File Format) file.

    Args:
        data (:class:`torch_geometric.data.Data`): The data object.
        path (str): The path to the file.
    """
    num_nodes, num_faces = data.pos.size(0), data.face.size(1)

    face = data.face.t()
    num_vertices = torch.full((num_faces, 1), face.size(1), dtype=torch.long)
    face = torch.cat([num_vertices, face], dim=-1)

    threshold = PRINT_OPTS.threshold
    torch.set_printoptions(threshold=float('inf'))
    with open(path, 'w') as f:
        f.write('OFF\n{} {} 0\n'.format(num_nodes, num_faces))
        f.write(re.sub(REGEX, r'', data.pos.__repr__()))
        f.write('\n')
        f.write(re.sub(REGEX, r'', face.__repr__()))
        f.write('\n')
    torch.set_printoptions(threshold=threshold)
