from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.explain.explanations import Explanation

from .base import ExplainerAlgorithm

EPS = 1e-15


class GNNExplainer(ExplainerAlgorithm):

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

    def forward(self, model: torch.nn.Module, x: Tensor, edge_index: Tensor,
                explainer_config: ExplainerConfig, model_config: ModelConfig,
                target: Tensor, target_index: Optional[Union[int,
                                                             Tensor]] = None,
                index: Optional[Union[int, Tensor]] = None,
                **kwargs) -> Explanation:

        if isinstance(target_index, Tensor):
            raise ValueError(
                'GNNExplainer only supports single target index for now')

        # can it be an issue for some models ?
        clear_masks(model)

        if model_config.task_level == ModelTaskLevel.node:
            node_mask, edge_mask = self.explain_node(model, x, edge_index,
                                                     explainer_config,
                                                     model_config, target,
                                                     target_index, index,
                                                     **kwargs)
        elif model_config.task_level == ModelTaskLevel.graph:
            node_mask, edge_mask = self.explain_graph(model, x, edge_index,
                                                      explainer_config,
                                                      model_config, target,
                                                      target_index, index,
                                                      **kwargs)
        else:
            raise ValueError(
                'GNNExplainer does not support task level {}'.format(
                    model_config.task_level))

        model = clear_masks(model)

        # build explanation
        return Explanation(x=x, edge_index=edge_index, edge_mask=edge_mask,
                           node_features_mask=node_mask)

    def explain_graph(self, model: torch.nn.Module, x: Tensor,
                      edge_index: Tensor, explainer_config: ExplainerConfig,
                      model_config: ModelConfig, target: Tensor,
                      target_index: Optional[Union[int, Tensor]] = None,
                      index: Optional[Union[int, Tensor]] = None,
                      **kwargs) -> Tuple[Tensor, Tensor]:
        node_mask, edge_mask = self.train_node_edge_mask(
            model, x, edge_index, explainer_config, model_config, target,
            target_index, index, kwargs)

        node_mask = node_mask.squeeze()
        if explainer_config.edge_mask_type == MaskType.object:
            edge_mask = edge_mask.detach().sigmoid()
        else:
            edge_mask = torch.ones(edge_index.size(1))
        return node_mask, edge_mask

    def explain_node(self, model: torch.nn.Module, x: Tensor,
                     edge_index: Tensor, explainer_config: ExplainerConfig,
                     model_config: ModelConfig, target: Tensor,
                     target_index: Optional[Union[int, Tensor]] = None,
                     index: Optional[Union[int, Tensor]] = None,
                     **kwargs) -> Tuple[Tensor, Tensor]:

        # if we are dealing with a node level task, we can restrict the
        # computation to the node of interest and its computation graph
        if model_config.task_level == ModelTaskLevel.node:
            num_nodes = x.size(0)
            num_edges = edge_index.size(1)
            x, edge_index, new_index, subset, hard_edge_mask, kwargs =\
                self.subgraph(model, index, x, edge_index, **kwargs)
            if target_index is not None and model_config.mode ==\
                    ModelMode.classification:
                target = torch.index_select(target, 1, subset)
            else:
                target = target[subset]
        if model_config.task_level in [ModelTaskLevel.node]:
            index = new_index
        else:
            index = None

        node_mask, edge_mask = self.train_node_edge_mask(
            model, x, edge_index, explainer_config, model_config, target,
            target_index, index, kwargs)

        new_mask = x.new_zeros(num_nodes, x.size(-1))
        new_mask[subset] = node_mask
        node_mask = new_mask.squeeze()

        if explainer_config.edge_mask_type == MaskType.object:
            new_edge_mask = edge_mask.new_zeros(num_edges)
            new_edge_mask[hard_edge_mask] = edge_mask.detach().sigmoid()
            edge_mask = new_edge_mask
        else:
            edge_mask = torch.zeros(num_edges)
            edge_mask[hard_edge_mask] = 1

        model = clear_masks(model)

        return node_mask, edge_mask

    def train_node_edge_mask(self, model, x, edge_index, explainer_config,
                             model_config, target, target_index, index,
                             kwargs):
        node_mask, edge_mask = self._initialize_masks(x, edge_index)
        self.to(x.device)

        if explainer_config.edge_mask_type == MaskType.object:
            set_masks(model, edge_mask, edge_index, apply_sigmoid=True)
            parameters = [node_mask, edge_mask]
        else:
            parameters = [node_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for _ in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * node_mask.sigmoid()
            out = model(x=h, edge_index=edge_index, **kwargs)
            loss = self.loss(out, target, node_mask, edge_mask,
                             return_type=model_config.return_type,
                             target_idx=target_index, node_index=index,
                             edge_mask_type=explainer_config.edge_mask_type,
                             model_mode=model_config.mode)
            loss.backward(retain_graph=True)
            optimizer.step()

        node_mask = node_mask.detach().sigmoid()
        return node_mask, edge_mask

    def _initialize_masks(
            self, x: Tensor,
            edge_index: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        node_feat_mask = Parameter(torch.randn(N, F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        edge_mask = Parameter(torch.randn(E) * std)
        return node_feat_mask, edge_mask

    def loss_regression(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        target_idx: Optional[int] = None,
        node_index: Optional[int] = None,
    ):
        if target_idx is not None:
            y_hat = y_hat[..., target_idx].unsqueeze(0)
            y = y[..., target_idx].unsqueeze(0)

        if node_index is not None and node_index >= 0:
            loss = torch.cdist(y_hat[node_index], y[node_index])
        else:
            loss = torch.cdist(y_hat, y)

        return loss

    def loss_classification(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        return_type: ModelReturnType,
        target_idx: Optional[int] = None,
        node_index: Optional[int] = None,
    ):
        if target_idx is not None:
            y_hat = y_hat[target_idx]
            y = y[target_idx]

        if return_type == ModelReturnType.probs:
            y_hat = y_hat.log()
        if return_type == ModelReturnType.raw:
            y_hat = y_hat.softmax(dim=-1)
        if node_index is not None and node_index >= 0:
            loss = -y_hat[node_index, y[node_index]]
        else:
            loss = -y_hat[0, y[0]]
        return loss

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor,
             node_mask: torch.Tensor, edge_mask: torch.Tensor,
             edge_mask_type: MaskType, return_type: ModelReturnType,
             node_index: Optional[int] = None,
             target_idx: Optional[int] = None,
             model_mode: ModelMode = ModelMode.regression) -> torch.Tensor:

        if model_mode == ModelMode.regression:
            loss = self.loss_regression(y_hat, y, target_idx, node_index)
        else:
            loss = self.loss_classification(y_hat, y, return_type, target_idx,
                                            node_index)

        if edge_mask_type is not None:
            m = edge_mask.sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = node_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def supports(self, explainer_config: ExplainerConfig,
                 model_config: ModelConfig) -> bool:
        if model_config.task_level not in [
                ModelTaskLevel.node, ModelTaskLevel.graph
        ]:
            return False
        if explainer_config.edge_mask_type == MaskType.attributes:
            return False

        if explainer_config.node_mask_type != MaskType.attributes:
            return False

        if model_config.task_level not in [
                ModelTaskLevel.node, ModelTaskLevel.graph
        ]:
            return False

        return True
