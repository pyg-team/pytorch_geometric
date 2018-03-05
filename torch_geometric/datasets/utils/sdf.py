import torch

from ..dataset import Data


def read_sdf(content):
    lines = content.split('\n')
    target = torch.FloatTensor([float(x) for x in lines[0].split()[5:]])
    target = target.view(1, -1)
    N, E = [int(x) for x in lines[3].split()[:2]]

    atoms = lines[4:4 + N]
    position = torch.Tensor([[float(y) for y in x.split()[:3]] for x in atoms])
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    types = torch.LongTensor([[types[x.split()[3]]] for x in atoms])
    value = torch.FloatTensor(N, 1).fill_(1)
    input = torch.FloatTensor(N, 5).fill_(0).scatter_(1, types, value)

    bonds = [[int(y) for y in x.split()[:3]] for x in lines[4 + N:4 + N + E]]
    bonds = torch.LongTensor(bonds)
    index = bonds[:, :2] - 1
    index_reverse = torch.cat([index[:, 1:], index[:, :1]], dim=1)
    index = torch.cat([index, index_reverse]).t()
    weight = bonds[:, 2].repeat(2).float()

    return Data(input, position, index, weight, target)
