from math import sqrt
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.utils import to_dense_adj


class CFExplainer(ExplainerAlgorithm):
    r"""The CF-Explainer model from the `"CF-GNNExplainer: Counterfactual Explanations for Graph Neural
Networks"
    <https://arxiv.org/abs/2102.03322>`_ paper for generating CF explanations for GNNs:
    the minimal perturbation to the input (graph) data such that the prediction changes."""

    coeffs = {
        'edge_size': 0.1,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, cf_optimizer="SGD",
                 n_momentum=0, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.cf_optimizer = cf_optimizer
        self.n_momentum = n_momentum
        self.coeffs.update(kwargs)
        self.edge_mask = self.hard_edge_mask = None
        self.best_cf_example = None
        self.best_loss = np.inf

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        index: int = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, index=index, **kwargs)

        if np.isinf(self.best_loss):
            raise Exception("No Counterfactual found.")

        edge_mask = self._post_process_mask(
            self.best_cf_example,
            apply_sigmoid=False,
        )
        self._clean_model(model)
        return Explanation(node_mask=None, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def _train(self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, *,
               target: Tensor, index: int = None, **kwargs):

        # Set edge mask to all ones
        self._initialize_edge_mask(x, edge_index)
        parameters = []
        if self.edge_mask is not None:
            # This line sets masks in the Message passing module of PyG
            # as learnable Parameters and enables the call of "explain_message"
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        if self.cf_optimizer == "SGD" and self.n_momentum == 0.0:
            optimizer = torch.optim.SGD(parameters, lr=self.lr)
        elif self.cf_optimizer == "SGD" and self.n_momentum != 0.0:
            optimizer = torch.optim.SGD(parameters, lr=self.lr, nesterov=True,
                                        momentum=self.n_momentum)
        elif self.cf_optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(parameters, lr=self.lr)
        else:
            raise Exception("Optimizer is not currently supported.")

        for i in range(self.epochs):
            optimizer.zero_grad()
            y_hat, y = model(x, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)
            loss.backward()
            optimizer.step()

            # Log best CF
            if (torch.argmax(y_hat) != y).float():
                if self.best_loss > loss:
                    binary_mask = (torch.sigmoid(self.edge_mask) > 0.5).int()
                    self.best_cf_example = torch.clone(1 - binary_mask)
                    self.best_loss = loss.item()

            # Log summary
            binary_mask = (torch.sigmoid(self.edge_mask) > 0.5).int()
            summary = f"Epoch {i} loss: {loss:.2f} | n_masked_edges: {sum(binary_mask == 0)}"
            summary = summary + f"| pred_class: {torch.argmax(y_hat)}"
            summary = summary + f"| max_grad: {self.edge_mask.grad.max():.2f}"
            print(summary)

    def _initialize_edge_mask(self, x: Tensor, edge_index: Tensor):
        edge_mask_type = self.explainer_config.edge_mask_type
        device = x.device
        E = edge_index.size(1)

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            self.edge_mask = Parameter(torch.ones(E, device=device))
        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        indicator = (torch.argmax(y_hat) == y).float()
        dist_loss_l1 = sum(
            abs(self.edge_mask - torch.ones_like(self.edge_mask)))
        loss = -indicator * loss + self.coeffs['edge_size'] * dist_loss_l1
        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.edge_mask = None
