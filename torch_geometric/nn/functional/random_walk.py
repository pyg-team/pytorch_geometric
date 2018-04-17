import torch
from torch_geometric.utils import new, matmul, one_hot


def random_walk(edge_index, edge_attr, target, weight):
    num_classes, num_steps = weight.size()

    p0 = one_hot(target, num_classes, out=new(edge_attr))  # Initialize p0.
    p0 /= p0.t().sum(dim=1, keepdim=True).t()  # Rescale p0.

    # Perform random walks.
    probs = [p0]
    for k in range(num_steps):
        probs.append(matmul(edge_index, edge_attr, probs[-1]))
    probs = probs[1:]
    probs = torch.stack(probs, dim=-1)  # num_nodes x num_classes x num_steps

    # Weight probabilities for each class and step separately.
    probs = weight.unsqueeze(0) * probs

    # Sum up probabilities of each step.
    probs = probs.sum(dim=-1)

    return probs
