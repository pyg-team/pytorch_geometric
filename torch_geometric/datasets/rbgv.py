from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from torch_geometric.data import InMemoryDataset
from torch_geometric.explain import Explanation
from torch_geometric.utils import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    to_undirected,
)

# Color codes used throughout the dataset.
RED, BLUE, GREEN, VIOLET = 0, 1, 2, 3


class RBGVDataset(InMemoryDataset):
    r"""The synthetic RBGV (Red, Blue, Green, Violet) graph classification
    dataset for evaluating explainability algorithms, as described in the
    `"GNN Explanations that do not Explain and How to find Them"
    <https://arxiv.org/abs/2601.20815>`_ paper.

    Each graph contains a main subgraph of red and blue nodes whose label is
    *fully* determined by their ratio (class :obj:`1` iff blue nodes strictly
    outnumber red nodes), plus exactly two confounder nodes (one green and one
    violet) that carry no causal information.

    On top of the original construction, this implementation exposes a
    **structural leakage** mechanism: the confounders can be wired into the
    main subgraph through a configurable target pool (:obj:`leakage_target`)
    and connection strategy (:obj:`leakage_strategy`). Leakage injects a
    spurious structural correlation an explainer may latch onto, which makes
    the dataset well suited to stress-test GNN explainers.

    Every graph is returned as an
    :class:`~torch_geometric.explain.Explanation` object whose ground-truth
    :obj:`node_mask` and :obj:`edge_mask` mark the causally relevant
    substructure (the red/blue subgraph), so explainer fidelity can be
    measured directly.

    .. code-block:: python

        from torch_geometric.datasets import RBGVDataset

        dataset = RBGVDataset(
            num_graphs=1000,
            topology='barabasi_albert',
            leakage_target='blue',
            leakage_strategy='normal',
        )

    Args:
        num_graphs (int, optional): The number of graphs to generate.
            (default: :obj:`1000`)
        min_nodes (int, optional): Lower bound (inclusive) for the number of
            red/blue nodes in the main subgraph. (default: :obj:`5`)
        max_nodes (int, optional): Upper bound (inclusive) for the number of
            red/blue nodes in the main subgraph. (default: :obj:`15`)
        topology (str, optional): Topology of the main subgraph, either
            :obj:`"erdos_renyi"` or :obj:`"barabasi_albert"`.
            (default: :obj:`"erdos_renyi"`)
        edge_prob (float, optional): Edge probability for the Erdos-Renyi
            topology. Also reused by the :obj:`"normal"` leakage strategy under
            the Erdos-Renyi topology. (default: :obj:`0.3`)
        num_edges_per_node (int, optional): Number of edges attached per node
            for the Barabasi-Albert topology. Also reused by the
            :obj:`"normal"` leakage strategy under that same topology.
            (default: :obj:`2`)
        leakage_target (str, optional): Pool of main-subgraph nodes the
            confounders may connect to, one of :obj:`"none"` (isolated
            confounders, original paper setup), :obj:`"red"`, :obj:`"blue"` or
            :obj:`"both"`. (default: :obj:`"none"`)
        leakage_strategy (str, optional): How confounders connect to the target
            pool, either :obj:`"all"` (dense, connect to every target node) or
            :obj:`"normal"` (stochastic, mirroring the main-subgraph topology).
            (default: :obj:`"all"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 1000
          - 7 to 17
          - varies
          - 4
          - 2
    """
    def __init__(
        self,
        num_graphs: int = 1000,
        min_nodes: int = 5,
        max_nodes: int = 15,
        topology: str = 'erdos_renyi',
        edge_prob: float = 0.3,
        num_edges_per_node: int = 2,
        leakage_target: str = 'none',
        leakage_strategy: str = 'all',
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=None, transform=transform)

        if min_nodes < 1 or max_nodes < min_nodes:
            raise ValueError(f"Invalid node range "
                             f"[{min_nodes}, {max_nodes}]")
        if topology not in ('erdos_renyi', 'barabasi_albert'):
            raise ValueError(f"Unknown topology '{topology}'")
        if leakage_target not in ('none', 'red', 'blue', 'both'):
            raise ValueError(f"Unknown leakage target '{leakage_target}'")
        if leakage_strategy not in ('all', 'normal'):
            raise ValueError(f"Unknown leakage strategy '{leakage_strategy}'")

        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.topology = topology
        self.edge_prob = edge_prob
        self.num_edges_per_node = num_edges_per_node
        self.leakage_target = leakage_target
        self.leakage_strategy = leakage_strategy

        data_list: List[Explanation] = [
            self.get_graph() for _ in range(num_graphs)
        ]
        self.data, self.slices = self.collate(data_list)

    def get_graph(self) -> Explanation:
        r"""Samples and returns a single RBGV graph as an
        :class:`~torch_geometric.explain.Explanation` object.
        """
        # --- 1. Main subgraph: red/blue nodes and the causal label ----------
        num_main = int(torch.randint(self.min_nodes, self.max_nodes + 1,
                                     (1, )))

        # Random 50/50 color assignment; 0 == red, 1 == blue.
        main_colors = torch.randint(0, 2, (num_main, ))
        num_blue = int((main_colors == BLUE).sum())
        num_red = num_main - num_blue

        # Ground-truth label: blue strictly outnumbers red.
        y = torch.tensor([int(num_blue > num_red)], dtype=torch.long)

        # Main-subgraph topology (both helpers return undirected edges).
        if self.topology == 'erdos_renyi':
            main_edge_index = erdos_renyi_graph(num_main, self.edge_prob,
                                                directed=False)
        else:  # 'barabasi_albert'
            main_edge_index = barabasi_albert_graph(num_main,
                                                    self.num_edges_per_node)

        # --- 2. Confounders: one green, one violet node ---------------------
        green_idx, violet_idx = num_main, num_main + 1
        num_total = num_main + 2

        # Full color vector and its 4-class one-hot node feature matrix.
        colors = torch.cat([main_colors, torch.tensor([GREEN, VIOLET])])
        x = F.one_hot(colors, num_classes=4).to(torch.float)

        # --- 3. Structural leakage ------------------------------------------
        # Resolve the pool of main-subgraph nodes confounders may attach to.
        if self.leakage_target == 'none':
            target_pool = main_colors.new_empty((0, ))
        elif self.leakage_target == 'red':
            target_pool = (main_colors == RED).nonzero(as_tuple=False).view(-1)
        elif self.leakage_target == 'blue':
            target_pool = (main_colors == BLUE).nonzero(
                as_tuple=False).view(-1)
        else:  # 'both'
            target_pool = torch.arange(num_main)

        confounders = torch.tensor([green_idx, violet_idx])
        num_pool = int(target_pool.numel())

        if num_pool == 0:
            # No reachable targets -> isolated confounders.
            leak_edge_index = main_edge_index.new_empty((2, 0))
        elif self.leakage_strategy == 'all':
            # Dense leakage: cartesian product confounders x target_pool.
            src = confounders.repeat_interleave(num_pool)
            dst = target_pool.repeat(confounders.numel())
            leak_edge_index = torch.stack([src, dst], dim=0)
        else:  # 'normal': stochastic leakage mirroring the main topology.
            if self.topology == 'erdos_renyi':
                # Bernoulli(edge_prob) draw per (confounder, target) pair.
                mask = torch.rand(confounders.numel(), num_pool)
                mask = mask < self.edge_prob
                conf_idx, pool_idx = mask.nonzero(as_tuple=True)
                src = confounders[conf_idx]
                dst = target_pool[pool_idx]
            else:  # 'barabasi_albert': pick num_edges_per_node random targets.
                k = min(self.num_edges_per_node, num_pool)
                # argsort of uniform noise == sampling without replacement.
                chosen = torch.rand(confounders.numel(), num_pool)
                chosen = chosen.argsort(dim=1)[:, :k]  # [num_confounders, k]
                src = confounders.view(-1, 1).expand(-1, k).reshape(-1)
                dst = target_pool[chosen].reshape(-1)
            leak_edge_index = torch.stack([src, dst], dim=0)

        # --- 4. Assemble the final undirected graph -------------------------
        # `to_undirected` symmetrizes and coalesces (sorts + dedupes) edges.
        edge_index = torch.cat([main_edge_index, leak_edge_index], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=num_total)

        # --- 5. Ground-truth explanation masks ------------------------------
        # Causally relevant nodes are exactly the red/blue ones (color < 2).
        node_mask = (colors < GREEN).to(torch.float).view(-1, 1)

        # An edge is internal to the main subgraph iff neither endpoint is a
        # confounder; every leakage edge touches a confounder (id >= num_main).
        # Deriving the mask from endpoint ids stays correct after coalescing.
        edge_mask = (edge_index < num_main).all(dim=0).to(torch.float)

        return Explanation(
            x=x,
            edge_index=edge_index,
            y=y,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'topology={self.topology}, '
                f'leakage_target={self.leakage_target}, '
                f'leakage_strategy={self.leakage_strategy})')
