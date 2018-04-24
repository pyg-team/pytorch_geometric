import torch


def read_txt(path, out=None):
    with open(path, 'r') as f:
        content = f.read().split('\n')[:-1]
        content = [[float(x) for x in line.split(',')] for line in content]

    src = torch.Tensor(content).squeeze()
    return src if out is None else out.resize_(src.size()).copy_(src)
