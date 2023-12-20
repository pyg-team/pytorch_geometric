import copy
from abc import ABC, abstractmethod
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
        return [key for key in self.keys() if key.endswith('_mask')]

    def validate_masks(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` masks."""
        status = True

        for store in self.node_stores:
            if 'node_mask' not in store:
                continue

            if store.node_mask.dim() != 2:
                status = False
                warn_or_raise(
                    f"Expected a 'node_mask' with two dimensions (got "
                    f"{store.node_mask.dim()} dimensions)", raise_on_error)

            if store.node_mask.size(0) not in {1, store.num_nodes}:
                status = False
                warn_or_raise(
                    f"Expected a 'node_mask' with {store.num_nodes} nodes "
                    f"(got {store.node_mask.size(0)} nodes)", raise_on_error)

            if 'x' in store:
                num_features = store.x.size(-1)
            else:
                num_features = store.node_mask.size(-1)

            if store.node_mask.size(1) not in {1, num_features}:
                status = False
                warn_or_raise(
                    f"Expected a 'node_mask' with {num_features} features ("
                    f"got {store.node_mask.size(1)} features)", raise_on_error)

        for store in self.edge_stores:
            if 'edge_mask' not in store:
                continue

            if store.edge_mask.dim() != 1:
                status = False
                warn_or_raise(
                    f"Expected an 'edge_mask' with one dimension (got "
                    f"{store.edge_mask.dim()} dimensions)", raise_on_error)

            if store.edge_mask.size(0) != store.num_edges:
                status = False
                warn_or_raise(
                    f"Expected an 'edge_mask' with {store.num_edges} edges "
                    f"(got {store.edge_mask.size(0)} edges)", raise_on_error)

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
            *args: Arguments passed to :class:`ThresholdConfig`.
            **kwargs: Keyword arguments passed to :class:`ThresholdConfig`.
        """
        threshold_config = ThresholdConfig.cast(*args, **kwargs)

        if threshold_config is None:
            return self

        # Avoid modification of the original explanation:
        out = copy.copy(self)

        for store in out.node_stores:
            store.node_mask = self._threshold_mask(store.get('node_mask'),
                                                   threshold_config)

        for store in out.edge_stores:
            store.edge_mask = self._threshold_mask(store.get('edge_mask'),
                                                   threshold_config)

        return out


class Explanation(Data, ExplanationMixin):
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


class HeteroExplanation(HeteroData, ExplanationMixin):
    r"""Holds all the obtained explanations of a heterogeneous graph.

    The explanation object is a :obj:`~torch_geometric.data.HeteroData` object
    and can hold node attributions and edge attributions.
    It can also hold the original graph if needed.
    """
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` object."""
        status = super().validate(raise_on_error)
        status &= self.validate_masks(raise_on_error)
        return status

    def get_explanation_subgraph(self) -> 'HeteroExplanation':
        r"""Returns the induced subgraph, in which all nodes and edges with
        zero attribution are masked out.
        """
        return self._apply_masks(
            node_mask_dict={
                key: mask.sum(dim=-1) > 0
                for key, mask in self.collect('node_mask', True).items()
            },
            edge_mask_dict={
                key: mask > 0
                for key, mask in self.collect('edge_mask', True).items()
            },
        )

    def get_complement_subgraph(self) -> 'HeteroExplanation':
        r"""Returns the induced subgraph, in which all nodes and edges with any
        attribution are masked out.
        """
        return self._apply_masks(
            node_mask_dict={
                key: mask.sum(dim=-1) == 0
                for key, mask in self.collect('node_mask', True).items()
            },
            edge_mask_dict={
                key: mask == 0
                for key, mask in self.collect('edge_mask', True).items()
            },
        )

    def _apply_masks(
        self,
        node_mask_dict: Dict[NodeType, Tensor],
        edge_mask_dict: Dict[EdgeType, Tensor],
    ) -> 'HeteroExplanation':
        out = copy.copy(self)

        for edge_type, edge_mask in edge_mask_dict.items():
            for key, value in self[edge_type].items():
                if key == 'edge_index':
                    out[edge_type].edge_index = value[:, edge_mask]
                elif self[edge_type].is_edge_attr(key):
                    out[edge_type][key] = value[edge_mask]

        return out.subgraph(node_mask_dict)

    def visualize_feature_importance(
        self,
        path: Optional[str] = None,
        feat_labels: Optional[Dict[NodeType, List[str]]] = None,
        top_k: Optional[int] = None,
    ):
        r"""Creates a bar plot of the node feature importances by summing up
        node masks across all nodes for each node type.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            feat_labels (Dict[NodeType, List[str]], optional): The labels of
                features for each node type. (default :obj:`None`)
            top_k (int, optional): Top k features to plot. If :obj:`None`
                plots all features. (default: :obj:`None`)
        """
        node_mask_dict = self.node_mask_dict
        for node_mask in node_mask_dict.values():
            if node_mask.dim() != 2 or node_mask.size(1) <= 1:
                raise ValueError(f"Cannot compute feature importance for "
                                 f"object-level 'node_mask' "
                                 f"(got shape {node_mask_dict.size()})")

        if feat_labels is None:
            feat_labels = {}
            for node_type, node_mask in node_mask_dict.items():
                feat_labels[node_type] = range(node_mask.size(1))

        score = torch.cat(
            [node_mask.sum(dim=0) for node_mask in node_mask_dict.values()],
            dim=0)

        all_feat_labels = []
        for node_type in node_mask_dict.keys():
            all_feat_labels += [
                f'{node_type}#{label}' for label in feat_labels[node_type]
            ]

        return _visualize_score(score, all_feat_labels, path, top_k)


def _visualize_score(
    score: torch.Tensor,
    labels: List[str],
    path: Optional[str] = None,
    top_k: Optional[int] = None,
):
    import matplotlib.pyplot as plt
    import pandas as pd

    if len(labels) != score.numel():
        raise ValueError(f"The number of labels (got {len(labels)}) must "
                         f"match the number of scores (got {score.numel()})")

    score = score.cpu().numpy()

    df = pd.DataFrame({'score': score}, index=labels)
    df = df.sort_values('score', ascending=False)
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
        ylabel='Feature label',
        xlim=[0, float(df['score'].max()) + 0.3],
        legend=False,
    )
    plt.gca().invert_yaxis()
    ax.bar_label(container=ax.containers[0], label_type='edge')

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


class ExplanationSetSampler(ABC):
    r"""Serves as a base class for sampling from an "Explanation Set" of a
    neural network. This set comprises data points that maximize the network's
    activation. It can be extended by various generative models or fixed size
    datasets to perform sampling.
    """
    @abstractmethod
    def sample(self, num_samples: int, **kwargs):
        r"""Abstract method to sample data points from the Explanation Set.

        Args:
            num_samples (int): The number of samples to generate.
            **kwargs: Additional arguments for sampling.
        """
        raise NotImplementedError(
            "The method sample must be implemented in subclasses")


class GenerativeExplanation(Data):
    r"""Holds all the obtained explanations of a homogeneous graph.

    The generative explanation object is a :obj:`~torch_geometric.data.Data`
    object and holds the explanation set.

    Args:
        explanation_set (ExplanationSetSampler, required):
            The explanation set used to explain NN activations, can be a finite
            set, generative model or anything that can sample from the abstract
            explanation set.
        is_finite (bool, required): Indicates whether the explanation set is
        finite. Should be set appropriately in subclasses.
    """
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the `GenerativeExplanation` object."""
        status = super().validate(raise_on_error)
        explanation_set = self.get("explanation_set")
        is_finite = self.get('is_finite')

        if explanation_set is None or is_finite is None:
            if raise_on_error:
                raise ValueError(
                    "Both 'explanation_set' and 'is_finite' must be set.")
            status = False
        return status

    def is_finite(self) -> bool:
        r"""Check if the Explanation Set is finite."""
        is_finite = self.get('is_finite')
        return is_finite

    def get_explanation_set(self, **kwargs):
        r"""Retrieves the Explanation Set. If the set is not finite, expects
        'num_samples' in kwargs to be provided for the sake of sampling a
        finite subset of the explanation set.

        Args:
            **kwargs: Key arguments, expected to contain 'num_samples' for
            infinite sets.

        Raises:
            ValueError: If the Explanation Set is infinite and 'num_samples' is
            not provided.
        """
        explanation_set = self.get("explanation_set")
        if not isinstance(explanation_set, ExplanationSetSampler):
            raise TypeError(
                "'explanation_set' must extend ExplanationSetSampler")

        if not self.is_finite():
            if 'num_samples' not in kwargs:
                raise ValueError(
                    "Expected 'num_samples' argument for an infinite",
                    "Explanation Set."
                )

        return explanation_set.sample(**kwargs)

    def visualize_explanation_graph(self, graph_state,
                                    path: Optional[str] = None,
                                    backend: Optional[str] = None):
        r"""Visualizes the explanation graph with edge weights set to be equal.

        Args:
            graph_state: The state of the graph to be visualized.
            path (Optional[str]): The path to where the plot is saved.
                If set to `None`, will visualize the plot on-the-fly.
            backend (Optional[str]): The graph drawing backend to use for
                visualization (`"graphviz"`, `"networkx"`).
                If set to `None`, will use the most appropriate
                visualization backend based on available system packages.
        """
        edge_index = graph_state.edge_index
        visualize_graph(edge_index, path=path, backend=backend)
