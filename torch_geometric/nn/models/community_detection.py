from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_add
from tqdm import tqdm

from torch_geometric.utils import to_undirected


class BigClam(torch.nn.Module):
    r"""Parallelized version of the original BigClam community detection
    algorithm as described in the paper:
    `BigClam: A Scalable and Parallelized
    Community Detection Algorithm <https://arxiv.org/abs/1303.4248>`_

    This version of the BigClam algorithm updates
    all of the nodes' assignments in parallel, which loses the
    original algorithm's optimization space convexity guarantees,
    but allows training orders of magnitude faster on large graphs.

    Args:
        edge_index (torch.Tensor): The undirected edge indices (2, E).
        num_communities (int): Number of communities (default: 8).
        device (torch.device, optional): Computation device (CPU/GPU).
    """
    def __init__(
        self,
        edge_index: Tensor,
        num_communities: int = 8,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.edge_index = to_undirected(edge_index).to(self.device)
        self.num_nodes = int(self.edge_index.max().item()) + 1

        if self.num_nodes == 0:
            raise ValueError("edge_index must contain at least one node.")
        self.num_communities = num_communities

        self.assignments = torch.nn.Parameter(
            torch.zeros(
                (self.num_nodes, self.num_communities),
                device=self.device,
                requires_grad=False,
            ),
            requires_grad=False,
        )
        self.sum_f = torch.empty((1, self.num_communities), device=self.device,
                                 requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters to initial state.
        """
        # Locally minimal neighborhoods seeding heuristic

        edge_index = self.edge_index.cpu().numpy()
        num_nodes = self.num_nodes
        adj = [set() for _ in range(num_nodes)]
        for u, v in zip(edge_index[0], edge_index[1]):
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[u]) for u in range(num_nodes)]
        vol_total = sum(deg)

        cond = [0.0] * num_nodes
        for u in range(num_nodes):

            S = adj[u].union({u})
            vol_S = sum(deg[v] for v in S)

            cut = 0
            for v in S:
                cut += sum(1 for w in adj[v] if w not in S)
            denom = min(vol_S, vol_total - vol_S) or 1
            cond[u] = cut / denom
        seeds = []
        for u in range(num_nodes):
            if all(cond[u] <= cond[v] for v in adj[u]):
                seeds.append(u)

        for i, seed in enumerate(seeds):
            if i >= self.num_communities:
                break
            S = adj[seed].union({seed})
            for v in S:
                self.assignments[v, i] = 1.0

        self.sum_f = self.assignments.sum(dim=0, keepdim=True)

    def forward(self, node_idx: Optional[Tensor] = None,
                normalize: bool = False) -> Tensor:
        """Compute soft community assignments for nodes.

        Args:
            node_idx (torch.Tensor, optional): Indices of nodes to return
            assignments for (default: None).
            normalize (bool): If True, apply softmax normalization
            (default: False).

        Returns:
            torch.Tensor: Community assignments for nodes,
            shape (N, num_communities).
        """
        F = torch.nn.functional.relu(self.assignments)
        if normalize:
            F = torch.nn.functional.softmax(F, dim=1)
        return F if node_idx is None else F[node_idx]

    def fit(
        self,
        iterations: int = 100,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> None:
        r"""Fit the BigClam model using gradient descent.

        Args:
            iterations (int): Number of iterations for training (default: 100).
            lr (float): Learning rate for gradient descent (default: 0.05).
            verbose (bool): If True, show progress bar (default: False).
        """
        self.train()
        row, col = self.edge_index
        for _ in (tqdm(range(iterations), desc="BigClam Training")
                  if verbose else range(iterations)):
            F = torch.nn.functional.relu(self.assignments)

            dot = (F[row] * F[col]).sum(dim=1)
            raw = torch.clamp(dot, min=-10.0, max=10.0)
            exp_neg = torch.exp(-raw)

            denom = (1.0 - exp_neg).clamp(min=1e-6)
            scores = (exp_neg / denom).unsqueeze(1)

            neb_grad = scatter_add(scores * F[col], row, dim=0,
                                   dim_size=self.num_nodes)
            neighbor_sum = scatter_add(F[col], row, dim=0,
                                       dim_size=self.num_nodes)

            without_grad = self.sum_f - F - neighbor_sum
            grad = neb_grad - without_grad
            grad /= grad.norm(p=2) + 1e-8

            new_F = torch.clamp(F + lr * grad, min=1e-8, max=10.0)

            delta = new_F - F
            with torch.no_grad():
                self.assignments.copy_(new_F)
                self.sum_f += delta.sum(dim=0, keepdim=True)
        self.eval()
