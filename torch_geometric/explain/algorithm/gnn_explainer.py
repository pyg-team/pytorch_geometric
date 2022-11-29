import logging
from math import sqrt
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)

from .base import ExplainerAlgorithm


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    The following configurations are currently supported:

    - :class:`torch_geometric.explain.config.ModelConfig`

        - :attr:`task_level`: :obj:`"node"` or :obj:`"graph"`

    - :class:`torch_geometric.explain.config.ExplainerConfig`

        - :attr:`node_mask_type`: :obj:`"object"`, :obj:`"common_attributes"`
          or :obj:`"attributes"`

        - :attr:`edge_mask_type`: :obj:`"object"` or :obj:`None`

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/gnn_explainer.py>`_ and
        `examples/gnn_explainer_ba_shapes.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/gnn_explainer_ba_shapes.py>`_.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.edge_mask = None

    def supports(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> bool:

        task_level = model_config.task_level
        if task_level not in [ModelTaskLevel.node, ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False

        edge_mask_type = explainer_config.edge_mask_type
        if edge_mask_type not in [MaskType.object, None]:
            logging.error(f"Edge mask type '{edge_mask_type.value}' not "
                          f"supported")
            return False

        node_mask_type = explainer_config.node_mask_type
        if node_mask_type not in [
                MaskType.common_attributes, MaskType.object,
                MaskType.attributes
        ]:
            logging.error(f"Node mask type '{node_mask_type.value}' not "
                          f"supported.")
            return False

        return True

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[int] = None,
        **kwargs,
    ) -> Explanation:

        hard_node_mask = hard_edge_mask = None
        if model_config.task_level == ModelTaskLevel.node:
            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            hard_node_mask, hard_edge_mask = self._get_hard_masks(
                model, index, edge_index, num_nodes=x.size(0))

        self._train(model, x, edge_index, explainer_config, model_config,
                    target, index, target_index, **kwargs)

        node_mask = self._post_process_mask(self.node_mask, x.size(0),
                                            hard_node_mask, apply_sigmoid=True)
        edge_mask = self._post_process_mask(self.edge_mask, edge_index.size(1),
                                            hard_edge_mask, apply_sigmoid=True)

        self._clean_model(model)

        # TODO Consider dropping differentiation between `mask` and `feat_mask`
        node_feat_mask = None
        if explainer_config.node_mask_type in {
                MaskType.attributes, MaskType.common_attributes
        }:
            node_feat_mask = node_mask
            node_mask = None

        return Explanation(x=x, edge_index=edge_index, edge_mask=edge_mask,
                           node_mask=node_mask, node_feat_mask=node_feat_mask)

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[int] = None,
        **kwargs,
    ):
        self._initialize_masks(
            x,
            edge_index,
            node_mask_type=explainer_config.node_mask_type,
            edge_mask_type=explainer_config.edge_mask_type,
        )

        parameters = [self.node_mask]  # We always learn a node mask.
        if explainer_config.edge_mask_type == MaskType.object:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()

            h = x * self.node_mask.sigmoid()

            y_hat = model(x=h, edge_index=edge_index, **kwargs)
            y = target

            if target_index is not None:
                y_hat, y = y_hat[target_index], y[target_index]
            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y, explainer_config, model_config)

            loss.backward()
            optimizer.step()

    def _initialize_masks(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_mask_type: MaskType,
        edge_mask_type: MaskType,
    ):
        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            raise NotImplementedError

        if edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        elif edge_mask_type is not None:
            raise NotImplementedError

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.mse_loss(y_hat, y)

    def _loss_classification(
        self,
        y_hat: Tensor,
        y: Tensor,
        return_type: ModelReturnType,
    ) -> Tensor:

        if y.dim() == 0:  # `index` was given as an integer.
            y_hat, y = y_hat.unsqueeze(0), y.unsqueeze(0)

        y_hat = self._to_log_prob(y_hat, return_type)
        return (-y_hat).gather(1, y.view(-1, 1)).mean()

    def _loss(
        self,
        y_hat: Tensor,
        y: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> Tensor:

        if model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        elif model_config.mode == ModelMode.classification:
            loss = self._loss_classification(y_hat, y,
                                             model_config.return_type)
        else:
            raise NotImplementedError

        if explainer_config.edge_mask_type is not None:
            m = self.edge_mask.sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = None
        self.edge_mask = None


class GNNExplainer_:
    r"""Deprecated version for :class:`GNNExplainer`."""

    coeffs = GNNExplainer.coeffs

    conversion_node_mask_type = {
        'feature': 'common_attributes',
        'individual_feature': 'attributes',
        'scalar': 'object',
    }

    conversion_return_type = {
        'log_prob': 'log_probs',
        'prob': 'probs',
        'raw': 'raw',
        "regression": 'raw',
    }

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = 'log_prob',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

        self.model = model
        self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
        self.explainer_config = ExplainerConfig(
            explanation_type='model',
            node_mask_type=self.conversion_node_mask_type[feat_mask_type],
            edge_mask_type=MaskType.object if allow_edge_mask else None,
        )
        self.model_config = ModelConfig(
            mode='regression'
            if return_type == 'regression' else 'classification',
            task_level=ModelTaskLevel.node,
            return_type=self.conversion_return_type[return_type],
        )

    ###########################################################################

    @torch.no_grad()
    def get_initial_prediction(self, model: torch.nn.Module, *args,
                               model_mode: ModelMode, **kwargs) -> Tensor:

        training = model.training
        model.eval()

        out = model(*args, **kwargs)
        if model_mode == ModelMode.classification:
            out = out.argmax(dim=-1)

        model.train(training)

        return out

    def explain_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self.model_config.task_level = ModelTaskLevel.graph

        explanation = self._explainer(
            model=self.model,
            x=x,
            edge_index=edge_index,
            explainer_config=self.explainer_config,
            model_config=self.model_config,
            target=self.get_initial_prediction(
                self.model,
                x,
                edge_index,
                model_mode=self.model_config.mode,
                **kwargs,
            ),
            **kwargs,
        )
        return self._convert_output(explanation, edge_index)

    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self.model_config.task_level = ModelTaskLevel.node
        explanation = self._explainer(
            model=self.model,
            x=x,
            edge_index=edge_index,
            explainer_config=self.explainer_config,
            model_config=self.model_config,
            target=self.get_initial_prediction(
                self.model,
                x,
                edge_index,
                model_mode=self.model_config.mode,
                **kwargs,
            ),
            index=node_idx,
            **kwargs,
        )
        return self._convert_output(explanation, edge_index, index=node_idx,
                                    x=x)

    def _convert_output(self, explanation, edge_index, index=None, x=None):
        if 'node_mask' in explanation.available_explanations:
            node_mask = explanation.node_mask
        else:
            if (self.explainer_config.node_mask_type ==
                    MaskType.common_attributes):
                node_mask = explanation.node_feat_mask[0]
            else:
                node_mask = explanation.node_feat_mask

        edge_mask = None
        if 'edge_mask' in explanation.available_explanations:
            edge_mask = explanation.edge_mask
        else:
            if index is not None:
                _, edge_mask = self._explainer._get_hard_masks(
                    self.model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask.to(x.dtype)
            else:
                edge_mask = torch.ones(edge_index.shape[1],
                                       device=edge_index.device)

        return node_mask, edge_mask
