import math

def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def glorot(tensor):
    stdv = math.sqrt(6.0/(tensor.shape[0]+tensor.shape[1]))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.zero_()