import copy
from typing import List, Optional

from torch import Tensor

from torch_geometric.data.data import Data, warn_or_raise


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

    @property
    def available_explanations(self) -> List[str]:
        """Returns the available explanation masks."""
        return [
            key for key in self.keys
            if key.endswith('_mask') and self[key] is not None
        ]

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the explanation"""
        status = super().validate()

        if 'node_mask' in self and self.num_nodes != self.node_mask.size(0):
            status = False
            warn_or_raise(
                f"Expected a 'node_mask' with {self.num_nodes} nodes "
                f"(got {self.node_mask.size(0)} nodes)", raise_on_error)

        if 'edge_mask' in self and self.num_edges != self.edge_mask.size(0):
            status = False
            warn_or_raise(
                f"Expected an 'edge_mask' with {self.num_edges} edges "
                f"(got {self.edge_mask.size(0)} edges)", raise_on_error)

        if 'node_feat_mask' in self:
            if 'x' in self and self.x.size() != self.node_feat_mask.size():
                status = False
                warn_or_raise(
                    f"Expected a 'node_feat_mask' of shape "
                    f"{list(self.x.size())} (got shape "
                    f"{list(self.node_feat_mask.size())})", raise_on_error)
            elif self.num_nodes != self.node_feat_mask.size(0):
                status = False
                warn_or_raise(
                    f"Expected a 'node_feat_mask' with {self.num_nodes} nodes "
                    f"(got {self.node_feat_mask.size(0)} nodes)",
                    raise_on_error)

        if 'edge_feat_mask' in self:
            if ('edge_attr' in self
                    and self.edge_attr.size() != self.edge_feat_mask.size()):
                status = False
                warn_or_raise(
                    f"Expected an 'edge_feat_mask' of shape "
                    f"{list(self.edge_attr.size())} (got shape "
                    f"{list(self.edge_feat_mask.size())})", raise_on_error)
            elif self.num_edges != self.edge_feat_mask.size(0):
                status = False
                warn_or_raise(
                    f"Expected an 'edge_feat_mask' with {self.num_edges} "
                    f"edges (got {self.edge_feat_mask.size(0)} edges)",
                    raise_on_error)

        return status

    def get_explanation_subgraph(self) -> 'Explanation':
        r"""Returns the induced subgraph, in which all nodes and edges with
        zero attribution are masked out."""
        return self._apply_masks(
            node_mask=self.node_mask > 0 if 'node_mask' in self else None,
            edge_mask=self.edge_mask > 0 if 'edge_mask' in self else None,
        )

    def get_complement_subgraph(self) -> 'Explanation':
        r"""Returns the induced subgraph, in which all nodes and edges with any
        attribution are masked out."""
        return self._apply_masks(
            node_mask=self.node_mask == 0 if 'node_mask' in self else None,
            edge_mask=self.edge_mask == 0 if 'edge_mask' in self else None,
        )

    def _apply_masks(
        self,
        node_mask: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
    ) -> 'Explanation':
        out = copy.copy(self)

        if edge_mask is not None:
            for key, value in self.items():
                if key == 'edge_index':
                    out.edge_index = value[:, edge_mask]
                elif self.is_edge_attr(key):
                    out[key] = value[edge_mask]

        if node_mask is not None:
            out = out.subgraph(node_mask)

        return out
