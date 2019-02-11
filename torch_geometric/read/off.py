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
    ### triangles to triangles list, as before
    tri = [[float(x) for x in line.split(None)[1:]] for line in face if line[0] == '3']

    ### squares to triangle list
    first = [1,2,3]
    second = [1,3,4]
    rects = [
           [float(x) for x in line.split(None)]
        for line in face if line[0] == '4']
    rects = [
        [
        [a[i] for i in first],
        [a[i] for i in second]
        ]
        for a in rects]
    rects = [item for sublist in rects for item in sublist]


    ### put all lists together, convert and return
    tri.extend(rects)
    tensor = torch.tensor(tri, dtype=torch.long) 
    
    return tensor.t().contiguous()


def read_off(path):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_off(src)
