import torch


def read_txt(path, out=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
        src = [[float(x) for x in line.split(',')] for line in src]

    src = torch.Tensor(src).squeeze()
    return src if out is None else out.resize_(src.size()).copy_(src)
