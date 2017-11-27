import torch
from torch.autograd import Function

from .voxel_cluster_gpu import voxel_cluster_gpu
from .scatter_max_gpu import scatter_max_gpu
from .scatter_arg_max_gpu import scatter_arg_max_gpu


class MaxPoolVoxel(Function):
    def __init__(self, adj, position, cluster_size, K):
        super(MaxPoolVoxel, self).__init__()
        self.adj = adj
        self.position = position
        self.cluster_size = cluster_size
        self.K = K

    def forward(self, input):
        cluster = voxel_cluster_gpu(self.position, self.cluster_size, self.K)
        max = scatter_max_gpu(input, cluster.int(), self.K)
        argmax = scatter_arg_max_gpu(input, cluster, max)
        self.n, self.argmax = input.size(0), argmax

        return max

    def backward(self, grad_output):
        k, m = grad_output.size()
        grad_input = grad_output.new(self.n * grad_output.size(1)).fill_(0)

        index = self.argmax * m
        index += torch.arange(0, m).type_as(index)

        grad_input[index.view(-1)] = grad_output.view(-1)

        return grad_input.view(-1, m)
