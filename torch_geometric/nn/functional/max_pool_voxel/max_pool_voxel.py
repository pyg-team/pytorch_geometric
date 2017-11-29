import torch
from torch.autograd import Function

from .voxel_cluster_gpu import voxel_cluster_gpu
from .scatter_max_gpu import scatter_max_gpu
from .scatter_arg_max_gpu import scatter_arg_max_gpu

from ....sparse import SparseTensor


class MaxPoolVoxel(Function):
    def __init__(self, adj, position, cluster_size, K):
        super(MaxPoolVoxel, self).__init__()
        self.adj = adj
        self.position = position
        self.cluster_size = cluster_size
        self.n, self.dim = position.size()
        self.K = K

    def forward(self, input):
        cluster = voxel_cluster_gpu(self.position, self.cluster_size, self.K)
        max = scatter_max_gpu(input, cluster.int(), self.K)
        argmax = scatter_arg_max_gpu(input, cluster, max)
        self.argmax = argmax

        node_count = self.position.new(self.K).fill_(0)
        node_count.scatter_add_(0, cluster, self.position.new(self.n).fill_(1))

        row, col = self.adj._indices()
        row, col = cluster[row], cluster[col]
        weight = self.adj._values()
        mask = row != col
        row, col, weight = row[mask], col[mask], weight[mask]
        index = torch.stack([row, col], dim=0)
        size = torch.Size([self.K, self.K])
        adj = SparseTensor(index, weight, size)

        node_count = self.position.new(self.K).fill_(0)
        node_count.scatter_add_(0, cluster, self.position.new(self.n).fill_(1))

        position = self.position.new(self.K * self.dim).fill_(0)
        cluster = cluster.view(-1, 1).repeat(1, self.dim) * 2
        cluster += torch.arange(0, self.dim, out=torch.cuda.LongTensor())
        position.scatter_add_(0, cluster.view(-1), self.position.view(-1))
        position = position.view(-1, self.dim)
        position /= node_count.view(-1, 1)

        return max, adj, position

    def backward(self, grad_output, adj=None, position=None):
        k, m = grad_output.size()
        grad_input = grad_output.new(self.n * grad_output.size(1)).fill_(0)

        index = self.argmax * m
        index += torch.arange(0, m).type_as(index)

        grad_input[index.view(-1)] = grad_output.view(-1)

        return grad_input.view(-1, m)
