"""Class representing explanations."""
from typing import Optional, Tuple, Union

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

    @staticmethod
    def from_masks(edge_index: Tensor, masks: Union[Tensor, Tuple[Tensor,
                                                                  ...]],
                   mask_keys: Union[str, Tuple[str, ...]],
                   auto_correct_names: bool = False,
                   **kwargs) -> 'Explanation':
        """Create an explanation from a mask or a tuple of masks.

        Args:
            edge_index (Tensor): The edge index of the graph.
            masks (Union[torch.Tensor, Tuple[torch.Tensor]]): the mask or a
                tuple of masks.
            mask_keys (Union[str, Tuple[str]]): the key or a tuple of keys for
                the masks. The names should end with "_mask".
            auto_correct_names (bool, optional): If set to :obj:`True`, the
                mask keys will be corrected if they do not end with "_mask".
                (default: :obj:`False`)
            **kwargs (optional): Additional attributes.

        Raises:
            ValueError: if mask_keys and masks do not have the same length.

        :rtype: :class:`torch_geometric.explain.Explanation`
        """
        # sanitize input
        if isinstance(masks, Tensor):
            masks = (masks, )
        if isinstance(mask_keys, str):
            mask_keys = (mask_keys, )
        if len(masks) != len(mask_keys):
            raise ValueError(
                "The number of masks and of mask keys should be equal.")
        if auto_correct_names:
            mask_keys = tuple(key if key.endswith("_mask") else key + "_mask"
                              for key in mask_keys)
        # validate mask_keys
        for key in mask_keys:
            if not key.endswith("_mask"):
                raise ValueError(
                    f"The mask keys should end with '_mask', got {key}.")
        masks_dict = dict(zip(mask_keys, masks))
        masks_dict.update(kwargs)
        return Explanation(edge_index=edge_index, **masks_dict)

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
