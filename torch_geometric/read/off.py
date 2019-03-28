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
    face = face_to_tri(face)

    data = Data(pos=pos)
    data.face = face

    return data


def face_to_tri(face):
    face = [[int(x) for x in line.strip().split(' ')] for line in face]

    triangle = torch.tensor([line[1:] for line in face if line[0] == 3])
    triangle = triangle.to(torch.int64)

    rect = torch.tensor([line[1:] for line in face if line[0] == 4])
    rect = rect.to(torch.int64)

    if rect.numel() > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]
    else:
        first, second = rect, rect

    return torch.cat([triangle, first, second], dim=0).t().contiguous()


def read_off(path):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_off(src)
