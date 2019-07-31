import os
import sys
import torch
from torch_geometric.data import Data

def yield_file(in_file):
    import torch
    f = open(in_file)
    buf = f.read()
    f.close()
    vertices = []
    faces = []
    for b in buf.split('\n'):
        if b.startswith('v '):
            x = b.split(" ")
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangles = b.split(' ')[1:]
            #We apply -1 because .obj is base 1 but the Data class expects base 0 indices
            yield ['f', [int(t.split("/")[0])-1 for t in triangles]]
        elif b.startswith('g '):
            yield ['g', b.split(" ")[-1]]
        else:
            yield ['', ""]


def read_obj(in_file):
    """Mesh wrapper to use for loading *.obj meshes"""
    vertices = []
    faces = []

    # if mesh.index[x]: f.index[mesh.index[x]]
    # else: mesh.append(x); f[x] = mesh.index[x]
    for k,v in yield_file(in_file):
        if k == 'v':
            vertices.append(torch.tensor(v))
        elif k == 'f':
            faces.append(torch.tensor(v, dtype=torch.int64))

    if not len(faces) or not len(vertices):
        return None

    pos = torch.stack(vertices, dim=0)
    face = torch.stack(faces, dim=-1)
    data = Data(pos=pos)
    data.face = face
    return data

