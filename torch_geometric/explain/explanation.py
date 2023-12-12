import copy
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data.data import Data, warn_or_raise
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.explain.config import ThresholdConfig, ThresholdType
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.visualization import visualize_graph

from torch_geometric.data.batch import Batch


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
    
    
class GenerativeExplanation(Data):
    r"""Holds all the obtained explanations of a homogeneous graph.

    The explanation object is a :obj:`~torch_geometric.data.Data` object and
    holds the generative model for a specific predicted class.

    Args:
        generative_model (torch.nn.Module, required):
            The generative model used to generate the explanation graphs. (default: :obj:`None`)
        for_class (int, required):
            Class for which the explanation is computed. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the :class:`Explanation` object."""
        status = super().validate(raise_on_error)
        status &= self.validate_masks(raise_on_error)
        return status
    
    def create_single_batch(dataset):
        data_list = [data for data in dataset]
        batched_data = Batch.from_data_list(data_list)
        return batched_data

    def sample_graphs(self, n=1, class_index=None):
        model = self.get('model')
        generative_models = self.get('generative_models')
        candidate_set = self.get('candidate_set')
        if class_index is None or class_index not in generative_models:
            raise ValueError(f"The argument 'class_index' is expected to be one of the possible targets.")
        
        if model is None:
            raise ValueError(f"The attribute 'model' is not available "
                             f"in '{self.__class__.__name__}' "
                             f"(got {self.available_explanations})")
            
        if generative_models is None:
            raise ValueError(f"The attribute 'generative_models' is not available "
                             f"in '{self.__class__.__name__}' "
                             f"(got {self.available_explanations})") 

        if candidate_set is None:  
            raise ValueError(f"The attribute 'candidate_set' is not available "
                             f"in '{self.__class__.__name__}' "
                             f"(got {self.available_explanations})")
        else:
            if not isinstance(candidate_set, dict):
                raise ValueError(f"The attribute 'candidate_set' must be a dict "
                                 f"in '{self.__class__.__name__}' "
                                 f"(got {type(candidate_set)})")

        # for _ in range(n): TODO: sample n graphs for each class
        # create initial graph state
        keys = list(candidate_set.keys())
        random_node_type = keys[torch.randint(len(keys), (1,)).item()]
        feature = candidate_set[random_node_type].unsqueeze(0)
        edge_index = torch.tensor([], dtype=torch.long).view(2, -1)
        node_type = [random_node_type,]
        initial_graph = Data(x=feature, edge_index=edge_index, node_type=node_type)

        if class_index is not None:
            current_graph_state = initial_graph
            _, _ = generative_models[class_index](current_graph_state, candidate_set)
            graph_state_batch = self.create_single_batch([current_graph_state,])
            gnn_output = model(graph_state_batch)
            probability_of_target_class = gnn_output[0][class_index]
            return current_graph_state, probability_of_target_class
        else:
            sampled_graphs = []
            for class_index, class_specific_generative_model in zip(range(len(generative_models)), generative_models):
                # make copy of initial graph state
                current_graph_state = Data(x=initial_graph.x.clone(), 
                                           edge_index=initial_graph.edge_index.clone(), 
                                           node_type=initial_graph.node_type.copy())
                _, _ = class_specific_generative_model(current_graph_state, candidate_set)
                graph_state_batch = self.create_single_batch([current_graph_state,])
                gnn_output = model(graph_state_batch)
                probability_of_target_class = gnn_output[0][class_index]
                sampled_graphs.append((current_graph_state, probability_of_target_class))
                

                
        return sampled_graphs