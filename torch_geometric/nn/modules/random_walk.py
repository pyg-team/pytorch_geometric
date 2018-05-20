import torch
from torch.nn import Module, Parameter
from torch_geometric.nn.functional.random_walk import random_walk

from .utils.inits import uniform


class RandomWalk(Module):
    """Random Walk Operator from the `"Adaptive Diffusions for Scalable
    Learning over Graphs" <https://arxiv.org/abs/1804.02081>`_ paper.

    Args:
        num_classes (int): Number of output classes.
        num_steps (int): Number of steps the random walker takes.
    """

    def __init__(self, num_classes, num_steps):
        super(RandomWalk, self).__init__()

        self.num_classes = num_classes
        self.num_steps = num_steps
        self.weight = Parameter(torch.Tensor(num_classes, num_steps))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_steps
        uniform(size, self.weight)

    def forward(self, edge_index, edge_attr, target):
        return random_walk(edge_index, edge_attr, target, self.weight)
