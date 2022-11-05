"""Class representing explanations."""
from typing import Optional, Tuple

from torch import Tensor

from torch_geometric.data import Data


class Explanation(Data):
    r"""Graph level explanation for a homogenous graph.

    The explanation object is a :obj:`torch_geometric.data.Data` object and
    can hold node-attribution, edge-attribution, feature-attribution. It can
    also hold the original graph if needed.

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:[num_nodes,
            num_node_features]. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        node_mask (Tensor, optional): dim (n_nodes,). (default: :obj:`None`)
        edge_mask (Tensor, optional): dim (n_edges,). (default: :obj:`None`)
        node_features_mask (Tensor, optional):dim (n_nodes, n_node_features).
            (default: :obj:`None`)
        edge_features_mask (Tensor, optional): dim (n_edges, n_edge_features).
            (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: Optional[Tensor] = None,
                 edge_index: Optional[Tensor] = None,
                 edge_attr: Optional[Tensor] = None,
                 y: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                 node_mask: Optional[Tensor] = None,
                 edge_mask: Optional[Tensor] = None,
                 node_features_mask: Optional[Tensor] = None,
                 edge_features_mask: Optional[Tensor] = None, **kwargs):

        self.node_mask: Optional[Tensor]
        self.edge_mask: Optional[Tensor]
        self.node_features_mask: Optional[Tensor]
        self.edge_features_mask: Optional[Tensor]

        super().__init__(x, edge_index, edge_attr, y, pos, node_mask=node_mask,
                         edge_mask=edge_mask,
                         node_features_mask=node_features_mask,
                         edge_features_mask=edge_features_mask, **kwargs)
        # how to update number of nodes of super() based only on masks ?
        # Could be useful if x is None
        self.basic_masks = [
            "node_mask", "edge_mask", "node_features_mask",
            "edge_features_mask"
        ]

    @property
    def available_explanations(self) -> Tuple[str, ...]:
        """Returns the available explanation masks."""
        return tuple(key for key in self.keys
                     if key.endswith('_mask') and self[key] is not None)

    def visualise_graph(self, threshold_node: Optional[float] = None,
                        threhsold_edge: Optional[float] = None, **kwargs):
        r"""Visualizes the node and edge attributions of the graph.

        Args:
            threshold_node (float, optional): Sets a threshold for
                important nodes.  If set to :obj:`None`, will visualize all
                nodes with transparancy indicating the importance of nodes.
                (default: :obj:`None`)
            threhsold_edge (float, optional): Same as threhsold nodes
                but for edges. (default: :obj:`None`)

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        raise NotImplementedError()

    def visualize_subgraph(self, node_idx: Optional[int], edge_index: Tensor,
                           edge_mask: Tensor, y: Optional[Tensor] = None,
                           threshold: Optional[int] = None,
                           edge_y: Optional[Tensor] = None,
                           node_alpha: Optional[Tensor] = None, seed: int = 10,
                           **kwargs):
        r"""Visualizes the subgraph given an edge mask :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`None` to explain a graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node.
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        raise NotImplementedError()
