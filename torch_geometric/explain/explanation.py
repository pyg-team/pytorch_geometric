import copy
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data.data import Data, warn_or_raise
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.explain.config import ThresholdConfig, ThresholdType
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.visualization import visualize_graph


class ExplanationMixin:
    @property
    def available_explanations(self) -> List[str]:
        """Returns the available explanation masks."""
        return [key for key in self.keys if key.endswith('_mask')]

    def validate_masks(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` masks."""
        status = True

        for store in self.node_stores:
            mask = store.get('node_mask')
            if mask is not None and store.num_nodes != mask.size(0):
                status = False
                warn_or_raise(
                    f"Expected a 'node_mask' with {store.num_nodes} nodes "
                    f"(got {mask.size(0)} nodes)", raise_on_error)

            mask = store.get('node_feat_mask')
            if (mask is not None and 'x' in store
                    and store.x.size() != mask.size()):
                status = False
                warn_or_raise(
                    f"Expected a 'node_feat_mask' of shape "
                    f"{list(store.x.size())} (got shape {list(mask.size())})",
                    raise_on_error)
            elif mask is not None and store.num_nodes != mask.size(0):
                status = False
                warn_or_raise(
                    f"Expected a 'node_feat_mask' with {store.num_nodes} "
                    f"nodes (got {mask.size(0)} nodes)", raise_on_error)

        for store in self.edge_stores:
            mask = store.get('edge_mask')
            if mask is not None and store.num_edges != mask.size(0):
                status = False
                warn_or_raise(
                    f"Expected an 'edge_mask' with {store.num_edges} edges "
                    f"(got {mask.size(0)} edges)", raise_on_error)

            mask = store.get('edge_feat_mask')
            if (mask is not None and 'edge_attr' in store
                    and store.edge_attr.size() != mask.size()):
                status = False
                warn_or_raise(
                    f"Expected an 'edge_feat_mask' of shape "
                    f"{list(store.edge_attr.size())} (got shape "
                    f"{list(mask.size())})", raise_on_error)
            elif mask is not None and store.num_edges != mask.size(0):
                status = False
                warn_or_raise(
                    f"Expected an 'edge_feat_mask' with {store.num_edges} "
                    f"edges (got {mask.size(0)} edges)", raise_on_error)

        return status

    def _threshold_mask(
        self,
        mask: Optional[Tensor],
        threshold_config: ThresholdConfig,
    ) -> Optional[Tensor]:

        if mask is None:
            return None

        if threshold_config.type == ThresholdType.hard:
            return (mask > threshold_config.value).float()

        if threshold_config.type in [
                ThresholdType.topk,
                ThresholdType.topk_hard,
        ]:
            if threshold_config.value >= mask.numel():
                if threshold_config.type == ThresholdType.topk:
                    return mask
                else:
                    return torch.ones_like(mask)

            value, index = torch.topk(
                mask.flatten(),
                k=threshold_config.value,
            )

            out = torch.zeros_like(mask.flatten())
            if threshold_config.type == ThresholdType.topk:
                out[index] = value
            else:
                out[index] = 1.0
            return out.view(mask.size())

        assert False

    def threshold(
        self,
        *args,
        **kwargs,
    ) -> Union['Explanation', 'HeteroExplanation']:
        """Thresholds the explanation masks according to the thresholding
        method.

        Args:
            threshold_config (ThresholdConfig): The threshold configuration.
        """
        threshold_config = ThresholdConfig.cast(*args, **kwargs)

        if threshold_config is None:
            return self

        # Avoid modification of the original explanation:
        out = copy.copy(self)

        for store in out.node_stores:
            for key in ['node_mask', 'node_feat_mask']:
                store[key] = self._threshold_mask(store.get(key),
                                                  threshold_config)

        for store in out.edge_stores:
            for key in ['edge_mask', 'edge_feat_mask']:
                store[key] = self._threshold_mask(store.get(key),
                                                  threshold_config)

        return out


class Explanation(Data, ExplanationMixin):
    r"""Holds all the obtained explanations of a homogenous graph.

    The explanation object is a :obj:`~torch_geometric.data.Data` object and
    can hold node-attributions, edge-attributions and feature-attributions.
    It can also hold the original graph if needed.

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
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` object."""
        status = super().validate(raise_on_error)
        status &= self.validate_masks(raise_on_error)
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

    def visualize_feature_importance(
        self,
        path: Optional[str] = None,
        feat_labels: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ):
        r"""Creates a bar plot of the node features importance by summing up
        :attr:`self.node_feat_mask` across all nodes.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            feat_labels (List[str], optional): Optional labels for features.
                (default :obj:`None`)
            top_k (int, optional): Top k features to plot. If :obj:`None`
                plots all features. (default: :obj:`None`)
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'node_feat_mask' not in self.available_explanations:
            raise ValueError(f"The attribute 'node_feat_mask' is not "
                             f"available in '{self.__class__.__name__}' "
                             f"(got {self.available_explanations})")

        feat_importance = self.node_feat_mask.sum(dim=0).cpu().numpy()

        if feat_labels is None:
            feat_labels = range(feat_importance.shape[0])

        if len(feat_labels) != feat_importance.shape[0]:
            raise ValueError(f"The '{self.__class__.__name__}' object holds "
                             f"{feat_importance.numel()} features, but "
                             f"only {len(feat_labels)} were passed")

        df = pd.DataFrame({'feat_importance': feat_importance},
                          index=feat_labels)
        df = df.sort_values("feat_importance", ascending=False)
        df = df.round(decimals=3)

        if top_k is not None:
            df = df.head(top_k)
            title = f"Feature importance for top {len(df)} features"
        else:
            title = f"Feature importance for {len(df)} features"

        ax = df.plot(
            kind='barh',
            figsize=(10, 7),
            title=title,
            xlabel='Feature label',
            xlim=[0, float(feat_importance.max()) + 0.3],
            legend=False,
        )
        plt.gca().invert_yaxis()
        ax.bar_label(container=ax.containers[0], label_type='edge')

        if path is not None:
            plt.savefig(path)
        else:
            plt.show()

        plt.close()

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
        visualize_graph(self.edge_index, self.edge_mask, path, backend)


class HeteroExplanation(HeteroData, ExplanationMixin):
    r"""Holds all the obtained explanations of a heterogeneous graph.

    The explanation object is a :obj:`~torch_geometric.hetero_data.HeteroData`
    and can hold node-attributions, edge-attributions and feature-attributions.
    It can also hold the original graph if needed.
    """
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` object."""
        status = super().validate(raise_on_error)
        status &= self.validate_masks(raise_on_error)
        return status

    def get_explanation_subgraph(self) -> 'HeteroExplanation':
        r"""Returns the induced subgraph, in which all nodes and edges with
        zero attribution are masked out."""
        return self._apply_masks(
            node_mask_dict={
                key: value > 0
                for key, value in self.node_mask_dict.items()
            },
            edge_mask_dict={
                key: value > 0
                for key, value in self.edge_mask_dict.items()
            },
        )

    def get_complement_subgraph(self) -> 'HeteroExplanation':
        r"""Returns the induced subgraph, in which all nodes and edges with any
        attribution are masked out."""
        return self._apply_masks(
            node_mask_dict={
                key: value == 0
                for key, value in self.node_mask_dict.items()
            },
            edge_mask_dict={
                key: value == 0
                for key, value in self.edge_mask_dict.items()
            },
        )

    def _apply_masks(
        self,
        node_mask_dict: Dict[NodeType, Tensor],
        edge_mask_dict: Dict[EdgeType, Tensor],
    ) -> 'HeteroExplanation':
        out = copy.deepcopy(self)

        for edge_type, edge_mask in edge_mask_dict.items():
            for key, value in self[edge_type].items():
                if key == 'edge_index':
                    out[edge_type].edge_index = value[:, edge_mask]
                elif self[edge_type].is_edge_attr(key):
                    out[edge_type][key] = value[edge_mask]

        return out.subgraph(node_mask_dict)
