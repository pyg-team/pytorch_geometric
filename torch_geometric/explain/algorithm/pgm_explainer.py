import logging
from math import sqrt
from typing import Optional, Tuple, Union
import numpy as np
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
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import k_hop_subgraph

from .base import ExplainerAlgorithm


class PGMExplainer(ExplainerAlgorithm):
    r"""The PGMExplainer model from the `"PGM-Explainer: Probabilistic Graphical Model
Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/2010.05788>


    Args:

        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    def __init__(
        self,
        num_layers: int = None,
        perturb_feature_list=None,
        perturb_mode="mean",  # mean, zero, max or uniform
        perturb_indicator="diff",  # diff or abs
        snorm_n=None,
        snorm_e=None,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        # edge_mask is only None if edge_mask_type is != MaskType.object


        print("Explainer settings")
        print("Number of layers: ", self.num_layers)
        print("Perturbation mode: ", self.mode)

    def perturb_features_on_node(self, feature_matrix, node_idx, is_random=False, is_pertubation_scaled=False):
        # return a random perturbed feature matrix
        # random = False for nothing, True for random.randint.
        # mode = "random" for random 0-1, "scale" for scaling with original feature

        X_perturb = feature_matrix
        if not is_pertubation_scaled:
            if not is_random == 0:
                perturb_array = X_perturb[node_idx]
            else:
                perturb_array = torch.randint(low=0, high=2, size=X_perturb[node_idx].shape[0])
            X_perturb[node_idx] = perturb_array
        elif is_pertubation_scaled:
            if not is_random == 0:
                perturb_array = X_perturb[node_idx]
            else:
                perturb_array = torch.multiply(X_perturb[node_idx],
                                            torch.rand(size=X_perturb[node_idx].shape[0]))*2
            X_perturb[node_idx] = perturb_array
        return X_perturb


    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        node_index: Optional[int] = None,
        **kwargs,
    ) -> Explanation:

        assert model_config.task_level in [
            ModelTaskLevel.graph, ModelTaskLevel.node
        ]

        model.eval()

        if model_config.task_level == ModelTaskLevel.node:
            pass
            # TODO
            # node_mask, edge_mask = self._explain_node(model, x, edge_index,
            #                                           explainer_config,
            #                                           model_config, target,
            #                                           node_index, target_index,
            #                                           **kwargs)
        elif model_config.task_level == ModelTaskLevel.graph:
            pass
            # TODO
            # node_mask, edge_mask = self._explain_graph(model, x, edge_index,
            #                                            explainer_config,
            #                                            model_config, target,
            #                                            target_index, **kwargs)

        # if explainer_config.node_mask_type == MaskType.object:
        #     node_feat_mask = None
        # else:
        #     node_feat_mask = node_mask
        #     node_mask = None

        self._clean_model(model)

        # build explanation
        return Explanation(x=x, edge_index=edge_index, edge_mask=edge_mask,
                           node_mask=node_mask, node_feat_mask=node_feat_mask)

    def n_hops_adj(self, n_hops, edge_index):
        # edge_index is the sparse representation of the adj matrix
        # Compute the n-hops adjacency matrix
        # adj = to_dense_adj(edge_index) # todo might not need this??

        n_hop_adjacency = power_adj = edge_index
        for i in range(n_hops - 1):
            power_adj = power_adj @ edge_index
            n_hop_adjacency = n_hop_adjacency + power_adj
            n_hop_adjacency = (n_hop_adjacency > 0).int()
        return n_hop_adjacency

    def extract_n_hops_neighbors(self, edge_index, node_idx):
        # Return the n-hops neighbors of a node
        node_adjacency = edge_index[node_idx]
        neighbors = torch.nonzero(node_adjacency)[0]
        node_idx_new = sum(node_adjacency[:node_idx])
        sub_A = self.A[neighbors][:, neighbors]
        sub_X = self.X[neighbors]
        return node_idx_new, sub_A, sub_X, neighbors

    def _explain_graph(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        pass

    def _explain_node(
        self,
        model: torch.nn.Module,
        x: Tensor,
        node_index: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        num_samples: int =100,
        top_node=None,
        p_threshold=0.05,
        pred_threshold=0.1

    ) -> Tuple[Tensor, Optional[Tensor]]:

        neighbors, edge_index, mapping, edge_mask = k_hop_subgraph(node_index, num_hops=5,
                   edge_index=edge_index, relabel_nodes=True)

        neighbors = neighbors.cpu().detach().numpy()

        if node_index not in neighbors:
            neighbors = np.append(neighbors, node_index)

        pred_torch = model(x, edge_index, edge_weight).cpu()
        oftmax_pred = np.asarray([torch.softmax(pred_torch[node_].data) for node_ in range(x.shape[0])])
        # softmax_pred = np.asarray([softmax(np.asarray(pred_torch[node_].data)) for node_ in range(self.X.shape[0])])

        return node_mask, edge_mask

    def _train_node_edge_mask(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index,
                               node_mask_type=explainer_config.node_mask_type,
                               edge_mask_type=explainer_config.edge_mask_type)

        if explainer_config.edge_mask_type == MaskType.object:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters = [self.node_mask, self.edge_mask]
        else:
            parameters = [self.node_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for _ in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_mask.sigmoid()
            out = model(x=h, edge_index=edge_index, **kwargs)
            loss_value = self.loss(
                out, target, return_type=model_config.return_type,
                target_idx=target_index, node_index=index,
                edge_mask_type=explainer_config.edge_mask_type,
                model_mode=model_config.mode)
            loss_value.backward(retain_graph=True)
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
        else:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)

        if edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)

    def _loss_regression(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        target_idx: Optional[int] = None,
        node_index: Optional[int] = None,
    ):
        if target_idx is not None:
            y_hat = y_hat[..., target_idx].unsqueeze(-1)
            y = y[..., target_idx].unsqueeze(-1)

        if node_index is not None and node_index >= 0:
            loss_ = torch.cdist(y_hat[node_index], y[node_index])
        else:
            loss_ = torch.cdist(y_hat, y)

        return loss_

    def _loss_classification(
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

        y_hat = self._to_log_prob(y_hat, return_type)

        if node_index is not None and node_index >= 0:
            loss = -y_hat[node_index, y[node_index]]
        else:
            loss = -y_hat[0, y[0]]
        return loss

    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        edge_mask_type: MaskType,
        return_type: ModelReturnType,
        node_index: Optional[int] = None,
        target_idx: Optional[int] = None,
        model_mode: ModelMode = ModelMode.regression,
    ) -> torch.Tensor:

        if model_mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y, target_idx, node_index)
        else:
            loss = self._loss_classification(y_hat, y, return_type, target_idx,
                                             node_index)

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
        if explainer_config.edge_mask_type == MaskType.attributes:
            logging.error("Edge mask type not supported.")
            return False

        if explainer_config.node_mask_type is None:
            logging.error("Node mask type not supported.")
            return False

        if model_config.task_level not in [
                ModelTaskLevel.node, ModelTaskLevel.graph
        ]:
            logging.error("Model task level not supported.")
            return False

        return True

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = None
        self.edge_mask = None
