import torch
from torch_geometric.utils import matmul


def random_walk(edge_index, edge_attr, one_hot, weight):
    num_classes, num_steps = weight.size()

    # Rescale to initial probabilities.
    # initial_prob = one_hot / one_hot.t().sum(dim=1, keepdim=True).t()
    initial_prob = one_hot

    # Perform random walks.
    probs = [initial_prob]
    for k in range(num_steps):
        probs.append(matmul(edge_index, edge_attr, probs[-1]))
    probs = probs[1:]
    probs = torch.stack(probs, dim=-1)  # num_nodes x num_classes x num_steps

    # Weight probabilities for each class and step separately.
    probs = weight.unsqueeze(0) * probs

    # Sum up probabilities of each step.
    probs = probs.sum(dim=-1)

    return probs
