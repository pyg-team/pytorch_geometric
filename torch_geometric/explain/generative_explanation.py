import copy
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data.data import Data, warn_or_raise
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.explain.config import ThresholdConfig, ThresholdType
from torch_geometric.visualization import visualize_graph


class GenerativeExplanation(Data):
    r"""Holds all the obtained explanations of a homogeneous graph.

    The explanation object is a :obj:`~torch_geometric.data.Data` object and
    can hold node attributions and edge attributions.
    It can also hold the original graph if needed.

    Args:
        node_mask (Tensor, optional): Node-level mask with shape
            :obj:`[num_nodes, 1]`, :obj:`[1, num_features]` or
            :obj:`[num_nodes, num_features]`. (default: :obj:`None`)
        edge_mask (Tensor, optional): Edge-level mask with shape
            :obj:`[num_edges]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` object."""
        status = super().validate(raise_on_error)
        status &= self.validate_masks(raise_on_error)
        return status

    def get_explanation_subgraph(self) -> 'Explanation':
        r"""Returns the induced subgraph, in which all nodes and edges with
        zero attribution are masked out.
        """
        node_mask = self.get('node_mask')
        if node_mask is not None:
            node_mask = node_mask.sum(dim=-1) > 0
        edge_mask = self.get('edge_mask')
        if edge_mask is not None:
            edge_mask = edge_mask > 0
        return self._apply_masks(node_mask, edge_mask)

    def get_complement_subgraph(self) -> 'Explanation':
        r"""Returns the induced subgraph, in which all nodes and edges with any
        attribution are masked out.
        """
        node_mask = self.get('node_mask')
        if node_mask is not None:
            node_mask = node_mask.sum(dim=-1) == 0
        edge_mask = self.get('edge_mask')
        if edge_mask is not None:
            edge_mask = edge_mask == 0
        return self._apply_masks(node_mask, edge_mask)

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

    def visualize_feature_importance(
        self,
        path: Optional[str] = None,
        feat_labels: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ):
        r"""Creates a bar plot of the node feature importances by summing up
        the node mask across all nodes.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            feat_labels (List[str], optional): The labels of features.
                (default :obj:`None`)
            top_k (int, optional): Top k features to plot. If :obj:`None`
                plots all features. (default: :obj:`None`)
        """
        node_mask = self.get('node_mask')
        if node_mask is None:
            raise ValueError(f"The attribute 'node_mask' is not available "
                             f"in '{self.__class__.__name__}' "
                             f"(got {self.available_explanations})")
        if node_mask.dim() != 2 or node_mask.size(1) <= 1:
            raise ValueError(f"Cannot compute feature importance for "
                             f"object-level 'node_mask' "
                             f"(got shape {node_mask.size()})")

        if feat_labels is None:
            feat_labels = range(node_mask.size(1))

        score = node_mask.sum(dim=0)

        return _visualize_score(score, feat_labels, path, top_k)

    def visualize_graph(self, path: Optional[str] = None,
                        backend: Optional[str] = None):
        r"""Visualizes the explanation graph with edge opacity corresponding to
        edge importance.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            backend (str, optional): The graph drawing backend to use for
                visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
                If set to :obj:`None`, will use the most appropriate
                visualization backend based on available system packages.
                (default: :obj:`None`)
        """
        edge_mask = self.get('edge_mask')
        if edge_mask is None:
            raise ValueError(f"The attribute 'edge_mask' is not available "
                             f"in '{self.__class__.__name__}' "
                             f"(got {self.available_explanations})")
        visualize_graph(self.edge_index, edge_mask, path, backend)

