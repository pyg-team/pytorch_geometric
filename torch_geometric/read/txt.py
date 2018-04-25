import torch


def split2d(src, func=float, sep=None, start=0, end=None, out=None):
    src = [[func(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.Tensor(src).squeeze()
    return src if out is None else out.resize_(src.size()).copy_(src)


def read_txt(path, func=float, sep=None, start=0, end=None, out=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return split2d(src, func, sep, start, end, out=out)
