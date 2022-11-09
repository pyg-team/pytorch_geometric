from typing import List, Optional

from torch import Tensor

from torch_geometric.data import Data


class Explanation(Data):
    r"""Graph level explanation for a homogenous graph.

    The explanation object is a :obj:`~torch_geometric.data.Data` object and
    can hold node-attribution, edge-attribution, feature-attribution. It can
    also hold the original graph if needed.

    Args:
        node_mask (Tensor, optional): dim (n_nodes,). (default: :obj:`None`)
        edge_mask (Tensor, optional): dim (n_edges,). (default: :obj:`None`)
        node_feat_mask (Tensor, optional):dim (n_nodes, n_node_features).
            (default: :obj:`None`)
        edge_feat_mask (Tensor, optional): dim (n_edges, n_edge_features).
            (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(
        self,
        node_mask: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
        node_feat_mask: Optional[Tensor] = None,
        edge_feat_mask: Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__(
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_feat_mask=node_feat_mask,
            edge_feat_mask=edge_feat_mask,
            **kwargs,
        )

    @property
    def available_explanations(self) -> List[str]:
        """Returns the available explanation masks."""
        return [
            key for key in self.keys
            if key.endswith('_mask') and self[key] is not None
        ]
