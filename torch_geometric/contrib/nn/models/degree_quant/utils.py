import torch
import torch_scatter
import time
import numpy as np
from torch.nn import Sequential
from torch_geometric.utils import degree
from torch import nn
# Finds the quantization minimum and maximum range along with the zero point and the scaling factor



# Builds a tensor using the bernoulli mask generated using the number of elements in the tensor.


def scatter_(name, src, index, dim=0, dim_size=None):
    assert name in ["add", "mean", "min", "max"]

    op = getattr(torch_scatter, "scatter_{}".format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out
    if name == "max":
        out[out < -10000] = 0
    elif name == "min":
        out[out > 10000] = 0
    return out


    

class ResettableSequential(Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()

def evaluate_prob_mask(data):
    return torch.bernoulli(data.prob_mask).to(torch.bool)



class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def evaluate_prob_mask(data):
    return torch.bernoulli(data.prob_mask).to(torch.bool)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Add for the testing datset
def inference_timing(model,loader, batch_size):



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batches = len(loader)
    model.eval()
    batch_time=[]
    print(f"Testing inference on {device}")
    torch.cuda.synchronize(device)
    
    for data in loader:

      start = time.time()
      out = model(data.to(device))
      
      torch.cuda.synchronize(device)
      
      end = time.time()
      batch_time.append((end-start) )

    batch_time = np.array(batch_time)/batch_size
    mean = np.mean(batch_time)
    std = np.std(batch_time)
    print(f'Inference Speed of each batch is {mean} ± {std} seconds for batch size {batch_size}')


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
