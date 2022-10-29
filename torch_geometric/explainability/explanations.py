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

    def get_available_explanations(self) -> Tuple[str, ...]:
        """Returns the available explanations."""
        return tuple(key for key in self.keys
                     if key.endswith('_mask') and self[key] is not None)

    def threshold(self, threshold: float) -> None:
        r"""Thresholds the attributions of the graph.

        Modify the explanation in place by setting to zero the attributions
        below the threshold.

        Args:
            threshold (float): Sets a threshold for important nodes and edges.
        """
        if self.node_mask is not None:
            self.node_mask = (self.node_mask > threshold).float()
        if self.edge_mask is not None:
            self.edge_mask = (self.edge_mask > threshold).float()
        if self.node_features_mask is not None:
            self.node_features_mask = (self.node_features_mask >
                                       threshold).float()
        if self.edge_features_mask is not None:
            self.edge_features_mask = (self.edge_features_mask >
                                       threshold).float()

    def threhsold_topk(self, threshold: int) -> None:
        r"""Thresholds the attributions of the graph.

        Modify the explanation in place keeping only the top most important
        attributions for each mask independently.

        Args:
            threshold (int): number of top most important attributions to keep.
        """
        raise NotImplementedError()

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
