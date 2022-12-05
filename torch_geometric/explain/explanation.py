import copy
from typing import Dict, List, Optional

from torch import Tensor

from torch_geometric.data.data import Data, warn_or_raise
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.typing import EdgeType, NodeType


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


class HeteroExplanation(HeteroData):
    r"""Holds all the obtained explanations of a heterogeneous graph.

    The explanation object is a :obj:`~torch_geometric.hetero_data.HeteroData`
    object and can hold node-attribution, edge-attribution,
    feature-attribution. It can also hold the original graph if needed.

    Args:
        node_mask (Dict[NodeType, Tensor], optional): Dictionary of Node-level
            mask with  key `NodeType` and value `Tensor` of
            shape :obj:`[num_nodes]`. (default: :obj:`None`)
        edge_mask (Dict[EdgeType, Tensor], optional): Dictionary of Edge-level
            mask with  key `EdgeType` and value `Tensor` of
            shape :obj:`[num_edges]`. (default: :obj:`None`)
        node_feat_mask (Dict[NodeType, Tensor], optional): Dictionary of
            Node-level feature mask with  key `NodeType` and value `Tensor` of
            shape :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_feat_mask (Dict[EdgeType, Tensor], optional): Dictionary of
            Edge-level feature mask with  key `EdgeType` and value `Tensor` of
            shape :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(
        self,
        node_mask: Optional[Dict[NodeType, Tensor]] = None,
        edge_mask: Optional[Dict[EdgeType, Tensor]] = None,
        node_feat_mask: Optional[Dict[NodeType, Tensor]] = None,
        edge_feat_mask: Optional[Dict[EdgeType, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # HeteroData doesn't set all kwargs as attributes, so we need to do it
        if node_mask is not None:
            self.node_mask = node_mask
        if edge_mask is not None:
            self.edge_mask = edge_mask
        if node_feat_mask is not None:
            self.node_feat_mask = node_feat_mask
        if edge_feat_mask is not None:
            self.edge_feat_mask = edge_feat_mask

    @property
    def available_explanations(self) -> List[str]:
        """Returns the available explanation masks."""
        return [
            key for key in self.keys if isinstance(key, str)
            and key.endswith('_mask') and self[key] is not None
        ]

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the explanation"""
        status = super().validate()

        x_dict = None
        edge_index_dict = None
        edge_attr_dict = None
        for key, value in self.node_items():
            if key == "x_dict":
                x_dict = value
            elif key == "edge_index_dict":
                edge_index_dict = value
            elif key == "edge_attr_dict":
                edge_attr_dict = value

        if 'node_mask' in self:
            for key in x_dict.keys():
                if key not in self.node_mask:
                    status = False
                    warn_or_raise(
                        f"Expected a 'node_mask' for node type {key}, but got "
                        f"none", raise_on_error)
                elif x_dict[key].size(0) != self.node_mask[key].size(0):
                    status = False
                    warn_or_raise(
                        f"Expected a 'node_mask' with {x_dict[key].size(0)} "
                        f"nodes (got {self.node_mask[key].size(0)} nodes)",
                        raise_on_error)

        if 'edge_mask' in self:
            for key in edge_index_dict.keys():
                if key not in self.edge_mask:
                    status = False
                    warn_or_raise(
                        f"Expected an 'edge_mask' for edge type {key}, but "
                        f"got none", raise_on_error)
                elif edge_index_dict[key].size(1) != self.edge_mask[key].size(
                        0):
                    status = False
                    warn_or_raise(
                        f"Expected an 'edge_mask' with "
                        f"{edge_index_dict[key].size(1)} edges (got "
                        f"{self.edge_mask[key].size(0)} edges)",
                        raise_on_error)

        if 'node_feat_mask' in self:
            for key in x_dict.keys():
                if key not in self.node_feat_mask:
                    status = False
                    warn_or_raise(
                        f"Expected a 'node_feat_mask' for node type {key}, "
                        f"but got none", raise_on_error)
                elif x_dict[key].size() != self.node_feat_mask[key].size():
                    status = False
                    warn_or_raise(
                        f"Expected a 'node_feat_mask' of shape "
                        f"{list(x_dict[key].size())} (got shape "
                        f"{list(self.node_feat_mask[key].size())})",
                        raise_on_error)

        if 'edge_feat_mask' in self:
            for key in edge_attr_dict.keys():
                if key not in self.edge_feat_mask:
                    status = False
                    warn_or_raise(
                        f"Expected an 'edge_feat_mask' for edge type {key}, "
                        f"but got none", raise_on_error)
                elif edge_attr_dict[key].size(
                ) != self.edge_feat_mask[key].size():
                    status = False
                    warn_or_raise(
                        f"Expected a 'edge_feat_mask' of shape "
                        f"{list(edge_attr_dict[key].size())} (got shape "
                        f"{list(self.edge_feat_mask[key].size())})",
                        raise_on_error)

        return status

    def get_explanation_subgraph(self) -> 'HeteroExplanation':
        r"""Returns the induced subgraph, in which all nodes and edges with
        zero attribution are masked out."""
        return self._apply_masks(
            node_mask={
                key: value > 0
                for key, value in self.node_mask.items()
            } if 'node_mask' in self else None,
            edge_mask={
                key: value > 0
                for key, value in self.edge_mask.items()
            } if 'edge_mask' in self else None,
        )

    def get_complement_subgraph(self) -> 'HeteroExplanation':
        r"""Returns the induced subgraph, in which all nodes and edges with any
        attribution are masked out."""
        return self._apply_masks(
            node_mask={
                key: value == 0
                for key, value in self.node_mask.items()
            } if 'node_mask' in self else None,
            edge_mask={
                key: value == 0
                for key, value in self.edge_mask.items()
            } if 'edge_mask' in self else None,
        )

    def _apply_masks(
        self,
        node_mask: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
    ) -> 'HeteroExplanation':
        out = copy.copy(self)

        if edge_mask is not None:
            for edge_type in self.edge_types:
                if edge_type not in edge_mask:
                    continue
                value = edge_mask[edge_type]
                out[edge_type].edge_index = out[edge_type].edge_index[:, value]
                if out[edge_type].edge_attr is not None:
                    out[edge_type].edge_attr = out[edge_type].edge_attr[value]

        if node_mask is not None:
            out = out.subgraph(node_mask)

        return out
