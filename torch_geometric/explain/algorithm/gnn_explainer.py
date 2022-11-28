import logging
from math import sqrt
from typing import Optional, Tuple, Union

import torch
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

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:

        if isinstance(index, Tensor):
            if index.numel() > 1:
                raise NotImplementedError(
                    f"'{self.__class__.__name}' only supports a single "
                    f"`index` for now")
            index = index.item()

        if isinstance(target_index, Tensor):
            if target_index.numel() > 1:
                raise NotImplementedError(
                    f"'{self.__class__.__name__}' only supports a single "
                    f"`target_index` for now")
            target_index = target_index.item()

        model.eval()
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        data_kwargs = dict(x=x, edge_index=edge_index)

        if model_config.task_level == ModelTaskLevel.node:
            # If we are dealing with a node-level task, we can restrict the
            # computation to the node of interest and its computation graph:
            if index is None:
                raise NotImplementedError("The `index` must currently be "
                                          "provided for a node-level task")

            x, edge_index, index, subset, hard_edge_mask, kwargs = (
                self.subgraph(model, index, x, edge_index, **kwargs))

            if (target_index is not None
                    and model_config.mode == ModelMode.classification):
                target = torch.index_select(target, 1, subset)
            else:
                target = target[subset]

        elif model_config.task_level == ModelTaskLevel.graph:
            subset = None
            hard_edge_mask = None
        else:
            raise NotImplementedError

        self._train_node_edge_mask(model, x, edge_index, explainer_config,
                                   model_config, target, index, target_index,
                                   **kwargs)

        node_mask, node_feat_mask, edge_mask = self._get_masks(
            num_nodes=num_nodes,
            num_edges=num_edges,
            explainer_config=explainer_config,
            task_level=model_config.task_level,
            hard_edge_mask=hard_edge_mask,
            subset=subset,
        )

        self._clean_model(model)

        return Explanation(edge_mask=edge_mask, node_mask=node_mask,
                           node_feat_mask=node_feat_mask, **data_kwargs)

    def _get_masks(
        self,
        num_nodes: int,
        num_edges: int,
        explainer_config: ExplainerConfig,
        task_level: ModelTaskLevel,
        hard_edge_mask: Optional[Tensor],
        subset: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Extracts and reshapes the masks from the model parameters."""
        node_mask_, node_feat_mask_, edge_mask_ = None, None, None

        if self.node_mask is not None:
            node_mask = self.node_mask.detach().sigmoid().squeeze(-1)
            if explainer_config.node_mask_type == MaskType.object:
                if task_level == ModelTaskLevel.node:
                    node_mask_ = node_mask.new_zeros(num_nodes)
                    node_mask_[subset] = node_mask
                elif task_level == ModelTaskLevel.graph:
                    node_mask_ = node_mask
                else:
                    raise NotImplementedError
            elif explainer_config.node_mask_type == MaskType.attributes:
                if task_level == ModelTaskLevel.node:
                    node_feat_mask_ = node_mask.new_zeros(
                        num_nodes, node_mask.size(-1))
                    node_feat_mask_[subset] = node_mask
                elif task_level == ModelTaskLevel.graph:
                    node_feat_mask_ = node_mask
                else:
                    raise NotImplementedError
            elif explainer_config.node_mask_type == MaskType.common_attributes:
                if task_level == ModelTaskLevel.node:
                    node_feat_mask_ = node_mask.new_zeros(
                        num_nodes, node_mask.numel())
                    node_feat_mask_[subset] = node_mask
                elif task_level == ModelTaskLevel.graph:
                    node_feat_mask_ = self._reshape_common_attributes(
                        node_mask, num_nodes)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        if self.edge_mask is not None:
            edge_mask = self.edge_mask.detach().sigmoid()
            if task_level == ModelTaskLevel.node:
                edge_mask_ = edge_mask.new_zeros(num_edges)
                edge_mask_[hard_edge_mask] = edge_mask
            elif task_level == ModelTaskLevel.graph:
                edge_mask_ = edge_mask
            else:
                raise NotImplementedError

        return node_mask_, node_feat_mask_, edge_mask_

    def _train_node_edge_mask(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(
            x,
            edge_index,
            node_mask_type=explainer_config.node_mask_type,
            edge_mask_type=explainer_config.edge_mask_type,
        )

        if explainer_config.edge_mask_type == MaskType.object:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters = [self.node_mask, self.edge_mask]
        else:
            assert explainer_config.edge_mask_type is None
            parameters = [self.node_mask]

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            h = x * self.node_mask.sigmoid()
            out = model(x=h, edge_index=edge_index, **kwargs)
            loss_value = self.loss(
                out,
                target,
                return_type=model_config.return_type,
                target_index=target_index,
                index=index,
                edge_mask_type=explainer_config.edge_mask_type,
                model_mode=model_config.mode,
            )
            loss_value.backward(retain_graph=False)
            optimizer.step()

    def _initialize_masks(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_mask_type: MaskType,
        edge_mask_type: MaskType,
    ):
        (N, F), E = x.size(), edge_index.size(1)
        device = x.device
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

    def _loss_regression(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        index: Optional[int] = None,
        target_index: Optional[int] = None,
    ):
        if target_index is not None:
            y_hat = y_hat[..., target_index].unsqueeze(-1)
            y = y[..., target_index].unsqueeze(-1)

        if index is not None:
            loss = torch.cdist(y_hat[index], y[index])
        else:
            loss = torch.cdist(y_hat, y)

        return loss

    def _loss_classification(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        return_type: ModelReturnType,
        index: Optional[int] = None,
        target_index: Optional[int] = None,
    ):
        if target_index is not None:
            y_hat = y_hat[target_index]
            y = y[target_index]

        y_hat = self._to_log_prob(y_hat, return_type)

        if index is not None:
            loss = -y_hat[index, y[index]]
        else:
            loss = -y_hat[0, y[0]]

        return loss

    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        edge_mask_type: MaskType,
        return_type: ModelReturnType,
        index: Optional[int] = None,
        target_index: Optional[int] = None,
        model_mode: ModelMode = ModelMode.regression,
    ) -> torch.Tensor:

        if model_mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y, index, target_index)
        elif model_mode == ModelMode.classification:
            loss = self._loss_classification(y_hat, y, return_type, index,
                                             target_index)
        else:
            raise NotImplementedError

        if edge_mask_type is not None:
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

    def supports(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> bool:
        if model_config.task_level not in [
                ModelTaskLevel.node, ModelTaskLevel.graph
        ]:
            logging.error("Model task level not supported.")
            return False
        if explainer_config.edge_mask_type not in [MaskType.object, None]:
            logging.error("Edge mask type not supported.")
            return False

        if explainer_config.node_mask_type not in [
                MaskType.common_attributes, MaskType.object,
                MaskType.attributes
        ]:
            logging.error("Node mask type not supported.")
            return False

        return True

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
            target=self._explainer.get_initial_prediction(
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
            target=self._explainer.get_initial_prediction(
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
            if self.explainer_config.node_mask_type ==\
                    MaskType.common_attributes:
                node_mask = explanation.node_feat_mask[0]
            else:
                node_mask = explanation.node_feat_mask

        if 'edge_mask' in explanation.available_explanations:
            edge_mask = explanation.edge_mask
        else:
            if index is not None:
                _, _, _, _, hard_edge_mask, _ = self._explainer.subgraph(
                    self.model,
                    index,
                    x,
                    edge_index,
                )
                edge_mask = torch.zeros(edge_index.shape[1],
                                        device=edge_index.device)
                edge_mask[hard_edge_mask] = 1
            else:
                edge_mask = torch.ones(edge_index.shape[1],
                                       device=edge_index.device)

        return node_mask, edge_mask
