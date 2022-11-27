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

    def get_explanation_subgraph(self) -> Data:
        r"""
        Return the explanation graph :obj:`~torch_geometric.data.Data` for
        this :obj:`~toch_geometric.explain.Explanation`.
        """
        mask_dict = {
            'node_mask': not self.node_mask,
            'edge_mask': not self.edge_mask,
            'node_feat_mask': not self.node_feat_mask,
            'edge_feat_mask': not self.edge_feat_mask
        }

        return self._mask_data(mask_dict)

    def get_complement_subgraph(self) -> Data:
        r"""
        Return the complement of the explanation graph
        :obj:`~torch_geometric.data.Data` for this
        :obj:`~toch_geometric.explain.Explanation`.
        """
        mask_dict = {
            'node_mask': self.node_mask,
            'edge_mask': self.edge_mask,
            'node_feat_mask': self.node_feat_mask,
            'edge_feat_mask': self.edge_feat_mask
        }

        return self._mask_data(mask_dict)

    def _mask_data(self, mask_dict) -> Data:
        r"""Returns the a masked graph :obj:`~torch_geometric.data.Data`
        for given `node_mask`, `edge_mask`, `node_feat_mask`, and
        `node_feat_mask` in `mask_dict`.
        """
        x_ = copy.copy(self.x)
        edge_index_ = copy.copy(self.edge_index)
        edge_attr_ = copy.copy(self.edge_attr)

        # update node and edge attributes
        if 'node_feat_mask' in mask_dict:
            x_ *= mask_dict['node_feat_mask']
        if 'edge_feat_mask' in mask_dict:
            edge_attr_ *= mask_dict['edge_feat_mask']
        if 'node_mask' in mask_dict:
            node_subset = self.x[mask_dict['node_mask']]
        if 'edge_mask' in mask_dict:
            edge_index_ = edge_index_[mask_dict['edge_mask']]

        masked_data = Data(x=x_, edge_index=edge_index_, edge_attr=edge_attr_)

        return masked_data.subgraph(node_subset)

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validate the correctness of the explanation"""
        status = super().validate()

        if self.node_mask is not None:
            if self.num_nodes != self.node_mask.shape[0]:
                status = False
                warn_or_raise(f"'num_nodes' ({self.num_nodes}) does not match "
                              f"'node_mask' shape ({self.node_mask.shape}")
        if self.edge_mask is not None:
            if self.num_edges != self.edge_mask.shape[0]:
                status = False
                warn_or_raise(f"'num_edges' ({self.num_edges}) does not match "
                              f"'edge_mask' shape ({self.edge_mask.shape}")
        if self.node_feat_mask is not None:
            if self.x.shape != self.node_feat_mask.shape:
                status = False
                warn_or_raise(
                    f"'node_feat_mask' shape ({self.node_feat_mask.shape}) "
                    f"and the shape of 'x' ({self.x.shape}) do not match")
        if self.edge_feat_mask is not None:
            if self.edge_attr is None:
                status = False
                warn_or_raise(
                    "'edge_attr' is None while `edge_feat_mask` is defined")
            elif self.edge_attr.shape != self.edge_feat_mask:
                status = False
                warn_or_raise(
                    f"'edge_feat_mask' shape ({self.edge_feat_mask.shape}) "
                    f"and the shape of 'edge_attr' ({self.edge_attr.shape}) "
                    "do not match.")

        return status

    @property
    def available_explanations(self) -> List[str]:
        """Returns the available explanation masks."""
        return [
            key for key in self.keys
            if key.endswith('_mask') and self[key] is not None
        ]
