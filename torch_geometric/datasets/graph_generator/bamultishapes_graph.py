import torch

from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator

from networkx import to_numpy_array
from networkx.generators import random_graphs

house = Data(
    num_nodes=5,
    edge_index=torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1],
    ]),
)
grid = Data(
    num_nodes=9,
    edge_index=torch.tensor([
        [
            0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7, 1, 3, 2, 4, 5, 6, 4, 7, 5, 8,
            7, 8
        ],
        [
            1, 3, 2, 4, 5, 6, 4, 7, 5, 8, 7, 8, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5,
            6, 7
        ],
    ]),
)
wheel = Data(
    num_nodes=6,
    edge_index=torch.tensor([
        [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 1, 4, 5, 5, 2, 3, 5, 4, 5, 5],
        [1, 4, 5, 5, 2, 3, 5, 4, 5, 5, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
    ]),
)


class BAMultiShapesGraph(GraphGenerator):
    r"""Generates random graphs as described inthe `"Global Explainability of
    GNNs via Logic Combination of Learned Concepts"
    <https://arxiv.org/abs/2210.07147>`__ paper.
    Given three atomic motifs, namely House (H), Wheel (W), and Grid (G),
    :class:`~torch_geometric.datasets.graph_generator.BAMultiShapesGraph`
    generates 1000 new graphs where each graph is obtained by attaching to a
    random Barabasi-Albert (BA) the motifs as follows:

        class 0: :math:`\emptyset` :math:`\lor` H :math:`\lor` W :math:`\lor`
        G :math:`\lor` :math:`\{H, W, G\}`
        class 1: H :math:`\land` W :math:`\lor` H :math:`\land` G :math:`\lor`
        W :math:`\land` G


    Args:
        target_class (int): The class for which generating a graph.
            Can be either :math:`0` or :math:`1`.
        num_nodes (int): The number of nodes in each graph.
            (default: :obj:`40`)
        num_connecting_edges (int): The number of random connections between
            the random Barabasi-Albert base and the motifs. (default: :obj:`1`)
    """
    def __init__(self, target_class, num_nodes=40, num_connecting_edges=1):
        if target_class < 0 or target_class > 2:
            raise ValueError(
                f"The value for target_class must lie in 0<=target_class<=2"
                f" got {target_class}")

        super().__init__()

        self.target_class = target_class
        self.num_nodes = num_nodes
        self.num_connecting_edges = num_connecting_edges

    def _barabasi_albert_graph(self, **kwargs):
        ba = random_graphs.barabasi_albert_graph(**kwargs)
        row, col = to_numpy_array(ba).nonzero()
        row = torch.from_numpy(row)
        col = torch.from_numpy(col)

        return Data(
            edge_index=torch.vstack([row, col]),
            y=self.target_class,
            x=torch.full((row.unique().shape[0], 10), 0.1),
            edge_mask=torch.zeros(row.shape[0]),
            node_mask=torch.zeros(row.unique().shape[0]),
        )

    def _merge(self, base, motifs):
        assert isinstance(motifs, list)

        edge_indices = [base.edge_index]
        num_nodes = base.num_nodes
        node_masks = [base.node_mask]
        edge_masks = [base.edge_mask]

        for motif in motifs:
            # Add motif to the graph
            edge_indices.append(motif.edge_index + num_nodes)
            node_masks.append(torch.ones(motif.num_nodes))
            edge_masks.append(torch.ones(motif.num_edges))

            for _ in range(self.num_connecting_edges):
                # Add random connection to the graph.
                i = int(torch.randint(0, base.num_nodes, (1, )))
                j = int(torch.randint(0, motif.num_nodes, (1, ))) + num_nodes
                edge_indices.append(torch.tensor([[i, j], [j, i]]))
                edge_masks.append(torch.zeros(2))

            num_nodes += motif.num_nodes

        return Data(
            edge_index=torch.cat(edge_indices, dim=1),
            y=base.y,
            x=torch.full((num_nodes, 10), 0.1),
            edge_mask=torch.cat(edge_masks, dim=0),
            node_mask=torch.cat(node_masks, dim=0),
        )

    def __call__(self, seed=None) -> Data:
        r"""
        Args:
            seed (int): Random seed to select which motif to add to the
                generated graph. If :obj:`None` the seed will be randomly
                extracted. (default: :obj:`None`)
        """
        # extract number and return selected motif as list
        seed = int(torch.randint(0, 10 if self.target_class == 0 else 3,
                                 (1, ))) if seed is None else seed
        if self.target_class == 0:
            if seed > 3:  # No motifs
                ret = self._barabasi_albert_graph(n=self.num_nodes, m=1)
            elif seed == 0:  # W
                ba = self._barabasi_albert_graph(
                    n=self.num_nodes - wheel.num_nodes, m=1)
                ret = self._merge(ba, [wheel])
            elif seed == 1:  # H
                ba = self._barabasi_albert_graph(
                    n=self.num_nodes - house.num_nodes, m=1)
                ret = self._merge(ba, [house])
            elif seed == 2:  # G
                ba = self._barabasi_albert_graph(
                    n=self.num_nodes - grid.num_nodes, m=1)
                ret = self._merge(ba, [grid])
            elif seed == 3:  # All
                ba = self._barabasi_albert_graph(
                    n=self.num_nodes - wheel.num_nodes - house.num_nodes -
                    grid.num_nodes, m=1)
                ret = self._merge(ba, [house, grid, wheel])
        else:
            if seed == 0:  # W + G
                ba = self._barabasi_albert_graph(
                    n=self.num_nodes - wheel.num_nodes - grid.num_nodes, m=1)
                ret = self._merge(ba, [wheel, grid])
            elif seed == 1:  # W + H
                ba = self._barabasi_albert_graph(
                    n=self.num_nodes - house.num_nodes - wheel.num_nodes, m=1)
                ret = self._merge(ba, [wheel, house])
            elif seed == 2:  # G + H
                ba = self._barabasi_albert_graph(
                    n=self.num_nodes - grid.num_nodes - house.num_nodes, m=1)
                ret = self._merge(ba, [grid, house])
        return ret

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(target_class={self.target_class}, '
                f'num_nodes={self.num_nodes}, '
                f'num_connecting_edges={self.num_connecting_edges}), ')
