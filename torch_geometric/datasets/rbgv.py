from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    to_undirected,
)

# Color codes used throughout the dataset.
RED, BLUE, GREEN, VIOLET = 0, 1, 2, 3


def _as_spurious_dict(
    value: Union[str, Dict[str, str]],
    valid: Tuple[str, ...],
    name: str,
) -> Dict[str, str]:
    r"""Normalizes a per-spurious-node option to a :obj:`{'green', 'violet'}`
    dictionary. A single string is broadcast to both spurious nodes, while a
    dictionary is validated to hold exactly those two keys.
    """
    if isinstance(value, str):
        resolved = {'green': value, 'violet': value}
    elif isinstance(value, dict):
        if set(value.keys()) != {'green', 'violet'}:
            raise ValueError(
                f"'{name}' dictionary must contain exactly the keys 'green' "
                f"and 'violet' (got {sorted(value.keys())})")
        resolved = {'green': value['green'], 'violet': value['violet']}
    else:
        raise TypeError(f"'{name}' must be a string or a dictionary "
                        f"(got '{type(value).__name__}')")

    for color, option in resolved.items():
        if option not in valid:
            raise ValueError(
                f"Unknown {name} '{option}' for the {color} spurious node")
    return resolved


class RBGVDataset(InMemoryDataset):
    r"""The synthetic RBGV (Red, Blue, Green, Violet) graph classification
    dataset for evaluating explainability algorithms, as described in the
    `"GNN Explanations that do not Explain and How to find Them"
    <https://arxiv.org/abs/2601.20815>`_ paper.

    Each graph contains a main subgraph of red and blue nodes whose label is
    *fully* determined by their ratio (class :obj:`1` iff blue nodes strictly
    outnumber red nodes), plus exactly two spurious nodes (one green and one
    violet) that carry no causal information about the label.

    Despite the simplicity of the task, the paper above has been shown 
    that some explainability algorithm may provide *only* the green 
    and the violet node as explanations, despite being uncorrelated
    with the task.

    On top of the original construction, this implementation exposes a
    **spurious connection** mechanism: the spurious nodes can be wired into the
    main subgraph through a configurable target pool (:obj:`spurious_target`)
    and connection strategy (:obj:`spurious_strategy`). Both options accept
    either a single string, applied symmetrically to both spurious nodes, or a
    :obj:`{'green', 'violet'}` dictionary to control each spurious node
    independently. These connections inject a spurious correlation between the
    spurious nodes and the label, letting the dataset test whether a GNN relies
    on the causal red/blue ratio or latches onto the spurious structure.

    Each graph is returned as a :class:`~torch_geometric.data.Data` object.

    .. code-block:: python

        from torch_geometric.datasets import RBGVDataset

        # Original dataset: both spurious nodes densely connect to every red node.
        dataset = RBGVDataset(
            num_graphs=5000,
            topology=barabasi_albert
        )

        # Symmetric: both spurious nodes densely connect to every red node.
        dataset = RBGVDataset(
            num_graphs=5000,
            spurious_target='red',
            spurious_strategy='all',
        )

        # Asymmetric: green densely connects to red nodes, while violet
        # stochastically connects to blue nodes.
        dataset = RBGVDataset(
            num_graphs=5000,
            topology='barabasi_albert',
            spurious_target={'green': 'red', 'violet': 'blue'},
            spurious_strategy={'green': 'all', 'violet': 'normal'},
        )

    Args:
        num_graphs (int, optional): The number of graphs to generate.
            (default: :obj:`5000`)
        min_nodes (int, optional): Lower bound (inclusive) for the number of
            red/blue nodes in the main subgraph. (default: :obj:`1`)
        max_nodes (int, optional): Upper bound (inclusive) for the number of
            red/blue nodes in the main subgraph. (default: :obj:`100`)
        topology (str, optional): Topology of the main subgraph, either
            :obj:`"erdos_renyi"` or :obj:`"barabasi_albert"`.
            (default: :obj:`"erdos_renyi"`)
        edge_prob (float, optional): Edge probability for the Erdos-Renyi
            topology. Also reused by the :obj:`"normal"` spurious strategy
            under that same topology. (default: :obj:`0.3`)
        num_edges_per_node (int, optional): Number of edges attached per node
            for the Barabasi-Albert topology. Also reused by the
            :obj:`"normal"` spurious strategy under that same topology.
            (default: :obj:`2`)
        spurious_target (str or dict, optional): Pool of main-subgraph nodes
            the spurious nodes may connect to, one of :obj:`"none"` (isolated
            spurious nodes, original paper setup), :obj:`"red"`, :obj:`"blue"`
            or :obj:`"both"`. Pass a :obj:`{'green', 'violet'}` dictionary to
            set it per spurious node. (default: :obj:`"none"`)
        spurious_strategy (str or dict, optional): How the spurious nodes
            connect to their target pool, either :obj:`"all"` (dense, connect
            to every target node) or :obj:`"normal"` (stochastic, mirroring the
            main-subgraph topology). Pass a :obj:`{'green', 'violet'}`
            dictionary to set it per spurious node. (default: :obj:`"all"`)
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
        num_graphs: int = 5000,
        min_nodes: int = 1,
        max_nodes: int = 100,
        topology: str = 'erdos_renyi',
        edge_prob: float = 0.3,
        num_edges_per_node: int = 2,
        spurious_target: Union[str, Dict[str, str]] = 'none',
        spurious_strategy: Union[str, Dict[str, str]] = 'all',
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=None, transform=transform)

        if min_nodes < 1 or max_nodes < min_nodes:
            raise ValueError(f"Invalid node range "
                             f"[{min_nodes}, {max_nodes}]")
        if topology not in ('erdos_renyi', 'barabasi_albert'):
            raise ValueError(f"Unknown topology '{topology}'")

        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.topology = topology
        self.edge_prob = edge_prob
        self.num_edges_per_node = num_edges_per_node
        self.spurious_target = spurious_target
        self.spurious_strategy = spurious_strategy

        # Normalize the per-spurious-node options to `{'green', 'violet'}`
        # dicts:
        self._target = _as_spurious_dict(spurious_target,
                                         ('none', 'red', 'blue', 'both'),
                                         'spurious_target')
        self._strategy = _as_spurious_dict(spurious_strategy,
                                           ('all', 'normal'),
                                           'spurious_strategy')

        data_list: List[Data] = [self.get_graph() for _ in range(num_graphs)]
        self.data, self.slices = self.collate(data_list)

    def _target_pool(self, target: str, main_colors: torch.Tensor,
                     num_main: int) -> torch.Tensor:
        r"""Returns the indices of the main-subgraph nodes a spurious node with
        the given :obj:`target` may connect to.
        """
        if target == 'none':
            return main_colors.new_empty((0, ))
        if target == 'red':
            return (main_colors == RED).nonzero(as_tuple=False).view(-1)
        if target == 'blue':
            return (main_colors == BLUE).nonzero(as_tuple=False).view(-1)
        return torch.arange(num_main)  # 'both'

    def _sample_targets(self, pool: torch.Tensor,
                        strategy: str) -> torch.Tensor:
        r"""Selects the subset of :obj:`pool` a spurious node connects to,
        according to the given :obj:`strategy` and the main-subgraph topology.
        """
        num_pool = int(pool.numel())
        if strategy == 'all':
            # Dense: connect to every node in the pool.
            return pool
        elif self.topology == 'erdos_renyi':
            # Bernoulli(edge_prob) draw per target node.
            mask = torch.rand(num_pool) < self.edge_prob
            return pool[mask]
        # 'barabasi_albert': pick num_edges_per_node random targets without
        # replacement (argsort of uniform noise == random permutation).
        k = min(self.num_edges_per_node, num_pool)
        idx = torch.rand(num_pool).argsort()[:k]
        return pool[idx]

    def get_graph(self) -> Data:
        r"""Samples and returns a single RBGV graph as a
        :class:`~torch_geometric.data.Data` object.
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

        # --- 2. Spurious nodes: one green, one violet -----------------------
        green_idx, violet_idx = num_main, num_main + 1
        num_total = num_main + 2

        # Full color vector and its 4-class one-hot node feature matrix.
        colors = torch.cat([main_colors, torch.tensor([GREEN, VIOLET])])
        x = F.one_hot(colors, num_classes=4).to(torch.float)

        # --- 3. Spurious connections ----------------------------------------
        # Each spurious node is wired independently, with its own target pool
        # and connection strategy.
        src_parts, dst_parts = [], []
        for node_idx, color in ((green_idx, 'green'), (violet_idx, 'violet')):
            pool = self._target_pool(self._target[color], main_colors,
                                     num_main)
            if pool.numel() == 0:
                continue
            dst = self._sample_targets(pool, self._strategy[color])
            src_parts.append(torch.full_like(dst, node_idx))
            dst_parts.append(dst)

        if src_parts:
            spurious_edge_index = torch.stack(
                [torch.cat(src_parts),
                 torch.cat(dst_parts)], dim=0)
        else:
            spurious_edge_index = main_edge_index.new_empty((2, 0))

        # --- 4. Assemble the final undirected graph -------------------------
        # `to_undirected` symmetrizes and coalesces (sorts + dedupes) edges.
        edge_index = torch.cat([main_edge_index, spurious_edge_index], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=num_total)

        return Data(x=x, edge_index=edge_index, y=y)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'topology={self.topology}, '
                f'spurious_target={self.spurious_target}, '
                f'spurious_strategy={self.spurious_strategy})')
