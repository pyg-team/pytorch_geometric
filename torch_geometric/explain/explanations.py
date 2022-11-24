import copy
from typing import List, Optional

from torch import Tensor

from torch_geometric.data import Data


class Explanation(Data):
    r"""Holds all the obtained explanations of a homogenous graph.

    The explanation object is a :obj:`~torch_geometric.data.Data` object and
    can hold node-attribution, edge-attribution, feature-attribution. It can
    also hold the original graph if needed.

    Args:
        node_mask (Tensor, optional): Node-level mask with shape
            :obj:`[num_nodes]`. (default: :obj:`None`)
        edge_mask (Tensor, optional): Edge-level mask with shape
            :obj:`[num_edges]`. (default: :obj:`None`)
        node_feat_mask (Tensor, optional): Node-level feature mask with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_feat_mask (Tensor, optional): Edge-level feature mask with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
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

    def get_explanation_subgraph(self) -> Data:
        r"""Returns the explanation graph :obj:`~torch_geometric.data.Data`
        for Explanations node, edge, and feature masks.
        """
        x_ = copy.copy(self.x)
        edge_attr_ = copy.copy(self.edge_attr)

        # update node and edge attributes
        if self.node_feat_mask is not None:
            x_ = self.x * self.node_feat_mask
        if self.edge_feat_mask is not None:
            edge_attr_ = self.edge_attr * self.edge_feat_mask

        # filter nodes and edges
        edge_index_ = self.edge_index[
            self.edge_mask] if self.edge_mask is not None else copy.copy(
                self.edge_mask)
        node_subset = self.x[
            self.node_mask] if self.node_mask is not None else copy.copy(
                self.x)

        explanation_graph = Data(x=x_, edge_index=edge_index_,
                                 edge_attr=edge_attr_)

        return explanation_graph.subgraph(node_subset)

    def get_complement_subgraph(self) -> Data:
        pass  # TODO (blaz)

    @property
    def available_explanations(self) -> List[str]:
        """Returns the available explanation masks."""
        return [
            key for key in self.keys
            if key.endswith('_mask') and self[key] is not None
        ]
