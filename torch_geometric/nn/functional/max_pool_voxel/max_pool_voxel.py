from torch.autograd import Function

from .voxel_cluster_gpu import voxel_cluster_gpu
from .scatter_max_gpu import scatter_max_gpu
from .scatter_arg_max_gpu import scatter_arg_max_gpu


class MaxPoolVoxel(Function):
    def __init__(self, cluster_size, K):
        super(MaxPoolVoxel, self).__init__()
        self.cluster_size = cluster_size
        self.K = K

    def forward(self, input, adj, position):
        cluster = voxel_cluster_gpu(position, self.cluster_size, self.K)
        max = scatter_max_gpu(input, cluster.int(), self.K)
        argmax = scatter_arg_max_gpu(input, cluster, max)
        self.save_for_backward(cluster, argmax)

        return input

    def backward(self, grad_output):
        cluster, argmax = self.saved_tensors
        n, m = cluster.size(0), grad_output.size(1)

        grad_input = grad_output.new(n, m).fill_(0)
        grad_input[argmax] = grad_output

        return grad_input
