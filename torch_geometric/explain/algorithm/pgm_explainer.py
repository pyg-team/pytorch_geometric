import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pgmpy.estimators.CITests import chi_square
from torch import Tensor

from torch_geometric.explain.config import ModelTaskLevel
from torch_geometric.explain.explanation import Explanation
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.subgraph import get_num_hops

from .base import ExplainerAlgorithm


class PGMExplainer(ExplainerAlgorithm):
    r"""The PGMExplainer model from the `"PGM-Explainer:
     Probabilistic Graphical Model Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/2010.05788>


    Args:
        perturb_mode (str): which method to generate the variations in features


    """
    def __init__(
        self,
        perturb_feature_list=None,
        perturb_mode="mean",  # mean, zero, max or uniform
        perturb_indicator="diff",  # diff or abs
        **kwargs,
    ):
        super().__init__()
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator

    def perturb_features_on_node(
            self, feature_matrix: torch.Tensor, node_idx: int,
            perturbation_mode: str = "randint", is_random: bool = False,
            is_perturbation_scaled: bool = False,
            perturb_feature_list: List = None) -> torch.Tensor:
        r"""
        perturb node feature matrix. This calculates how much influence
        neighbouring nodes has on the output
        Args:
            feature_matrix (torch.Tensor) : node feature matrix of
                the input graph of shape [num_nodes, num_features]
            node_idx (int): index of the node we are calculating
                the explanation for
            perturbation_mode (str): how to perturb the features.
                Must be one of  [random, randint, zero, mean, max]
            is_random (bool): whether to perturb the features,
                if set to False return the original feature matrix
            is_perturbation_scaled (bool): whether to scale the perturbed
                matrix with the original feature matrix
            perturb_feature_list (List)
        Returns:
            a randomly perturbed feature matrix
        """

        X_perturb = feature_matrix.detach().clone()
        perturb_array = X_perturb[node_idx].clone()
        epsilon = 0.05 * torch.max(feature_matrix, dim=0).values

        if is_random:
            if not is_perturbation_scaled:
                if perturbation_mode == "randint":
                    perturb_array = torch.randint(
                        high=2, size=X_perturb[node_idx].shape[0])
                # graph explainers
                elif perturbation_mode in ("mean", "zero", "max", "uniform"):
                    if perturbation_mode == "mean":
                        perturb_array[perturb_feature_list] = torch.mean(
                            feature_matrix[:, perturb_feature_list])
                    elif perturbation_mode == "zero":
                        perturb_array[perturb_feature_list] = 0
                    elif perturbation_mode == "max":
                        perturb_array[perturb_feature_list] = torch.max(
                            feature_matrix[:, perturb_feature_list])
                    elif perturbation_mode == "uniform":
                        random_perturbations = torch.rand(
                            perturb_array.shape) * 2 * epsilon - epsilon
                        perturb_array[perturb_feature_list] = perturb_array[
                            perturb_feature_list] + random_perturbations
                        perturb_array.clamp(
                            min=0, max=torch.max(feature_matrix, dim=0))

            elif is_perturbation_scaled:
                perturb_array = torch.multiply(
                    X_perturb[node_idx],
                    torch.rand(size=X_perturb[node_idx].shape[0])) * 2
        X_perturb[node_idx] = perturb_array
        return X_perturb

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        index: Optional[int] = None,  # node index
        **kwargs,
    ) -> Explanation:
        r"""
        generate the explanations
        Args:
            model (torch.nn.Module): model used to generate predictions
            x (torch.Tensor): the node feature matrix tensor
            edge_index:
            target (torch.Tensor): the prediction labels
            target_index:
            index (int): index of the node generating explanations for
            **kwargs:

        Returns:

        """
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

        assert self.model_config.task_level in [
            ModelTaskLevel.graph, ModelTaskLevel.node
        ]

        model.eval()

        edge_weight = kwargs.pop('edge_weight', None)
        num_samples = kwargs.pop('num_samples', 100)
        significance_threshold = kwargs.pop('significance_threshold', 0.05)
        max_subgraph_size = kwargs.get('max_subgraph_size', None)
        pred_threshold = kwargs.pop('pred_threshold', 0.1)
        if self.model_config.task_level == ModelTaskLevel.node:

            neighbors, pgm_stats = self.explain_node(
                model, x, index, edge_index, edge_weight, target[index],
                num_samples=num_samples, max_subgraph_size=max_subgraph_size,
                significance_threshold=significance_threshold,
                pred_threshold=pred_threshold, **kwargs)
            return Explanation(
                x=x,
                edge_index=edge_index,
                node_mask=neighbors,
                pgm_stats=pgm_stats,
            )
        elif self.model_config.task_level == ModelTaskLevel.graph:
            pgm_nodes, p_values, candidate_nodes = self.explain_graph(
                model=model, x=x, index=index, target=target,
                num_samples=num_samples, max_subgraph_size=max_subgraph_size,
                significance_threshold=significance_threshold,
                pred_threshold=pred_threshold, **kwargs)
            return Explanation(node_mask=p_values < significance_threshold)

        # build explanation

    def batch_perturb_features_on_node(
            self,
            num_samples: int,
            index_to_perturb: int,
            X_features: torch.Tensor,
            model: torch.nn.Module,
            percentage: float = 50.,  # % time node gets perturbed
            perturbation_mode: str = "mean",
            **kwargs) -> List:
        r"""
        perturb the node features of a batch of graphs
        for graph classification tasks
        Args:
            num_samples:
            index_to_perturb:
            X_features:
            model:
            percentage:
            perturbation_mode:
            **kwargs:

        Returns:

        """

        pred_torch = model(X_features, **kwargs)
        soft_pred = torch.softmax(pred_torch, dim=1)
        pred_label = torch.argmax(soft_pred, dim=1)
        num_nodes = X_features.shape[0]

        samples = []
        for iteration in range(num_samples):
            X_perturb = X_features.detach().clone()
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(
                            X_perturb, node,
                            perturbation_mode=perturbation_mode,
                            perturb_feature_list=self.perturb_feature_list,
                            is_random=True)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)

            pred_perturb_torch = model(X_perturb, **kwargs)
            soft_pred_perturb = torch.softmax(pred_perturb_torch,
                                              dim=1).squeeze()

            pred_change = torch.max(soft_pred) - soft_pred_perturb[pred_label]

            sample.append(pred_change)
            samples.append(sample)

        samples = torch.tensor(samples)
        if self.perturb_indicator == "abs":
            samples = torch.abs(samples)

        top = int(num_samples / 8)
        top_idx = torch.argsort(samples[:, num_nodes])[-top:]
        for i in range(num_samples):
            if i in top_idx:
                samples[i, num_nodes] = 1
            else:
                samples[i, num_nodes] = 0

        return samples

    def explain_graph(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        target=None,
        num_samples: int = 100,
        max_subgraph_size: int = None,  # num of neightbours to consider
        significance_threshold: float = 0.05,
        perturbation_mode: str = "mean",
        **kwargs,
    ):
        r"""
        generate explanations for graph classification tasks
        Args:
            model: pytorch model
            x (torch.Tensor): node features
            target(torch.Tensor):
            num_samples (int): number of samples to use to
                generate perturbations
            max_subgraph_size (int): max number of neighbours to consider
                for the explanation
            significance_threshold (float): threshold below which to consider
                a node has an effect on the prediction
            perturbation_mode (str): how to perturb the features.
                Must be one of  [random, randint, zero, mean, max]
            **kwargs:

        Returns:

        """
        model.eval()
        num_nodes = x.shape[0]
        if not max_subgraph_size:
            max_subgraph_size = int(num_nodes / 20)

        samples = self.batch_perturb_features_on_node(
            num_samples=int(num_samples / 2),
            index_to_perturb=range(num_nodes), X_features=x, model=model,
            perturbation_mode=perturbation_mode, **kwargs)

        data = pd.DataFrame(np.array(samples.detach().cpu()))
        # est = PC(data)

        p_values = []
        # todo to check --
        # https://github.com/vunhatminh/PGMExplainer/blob/715402aa9a014403815f518c4c7d9258eb49bbe9/PGM_Graph/pgm_explainer_graph.py#L138 # noqa
        # sets target = num_nodes, which doesnt seem correct?
        for node in range(num_nodes):
            chi2, p, _ = chi_square(node, int(target.detach().cpu()), [], data,
                                    boolean=False,
                                    significance_level=significance_threshold)
            p_values.append(p)

        # the original code uses number_candidates_nodes = int(top_nodes * 4)
        # if we consider 'top nodes' to equate to max number of nodes
        # it does not seem correct to set number_candidates to 4 times that...
        candidate_nodes = np.argpartition(
            p_values, max_subgraph_size)[0:max_subgraph_size]

        # Round 2
        samples = self.batch_perturb_features_on_node(
            num_samples=num_samples, index_to_perturb=candidate_nodes,
            X_features=x, model=model, perturbation_mode=perturbation_mode,
            **kwargs)

        # note: the PC estimator is in the original code but
        # as it does nothing it is not included here
        data = pd.DataFrame(np.array(samples.detach().cpu()))

        p_values = []
        dependent_nodes = []

        target = num_nodes
        for node in range(num_nodes):
            chi2, p, _ = chi_square(node, target, [], data, boolean=False,
                                    significance_level=significance_threshold)
            p_values.append(p)
            if p < significance_threshold:
                dependent_nodes.append(node)

        top_p = np.min((max_subgraph_size, num_nodes - 1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)

        return pgm_nodes, torch.tensor(p_values), candidate_nodes

    def explain_node(self, model: torch.nn.Module, x: Tensor, node_index: int,
                     edge_index: Tensor, edge_weight: Tensor, target: Tensor,
                     num_samples: int = 100, max_subgraph_size: int = None,
                     significance_threshold=0.05, pred_threshold=0.1):
        logging.info(f'Explaining node: {node_index}')

        neighbors, edge_index_new, mapping, edge_mask_new = k_hop_subgraph(
            node_idx=node_index,
            num_hops=get_num_hops(model),
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=x.size(0),
        )

        if node_index not in neighbors:
            neighbors = torch.cat([neighbors, node_index], dim=1)

        pred_model = model(x, edge_index, edge_weight)

        softmax_pred = torch.softmax(pred_model, dim=1)

        samples = []
        pred_samples = []

        for iteration in range(num_samples):
            X_perturb = x.detach().clone()

            sample = []
            for node in neighbors:
                seed = np.random.choice([1, 0])
                if seed == 1:
                    perturbation_mode = "random"
                else:
                    perturbation_mode = "zero"
                X_perturb = self.perturb_features_on_node(
                    X_perturb, node, perturbation_mode=perturbation_mode)
                sample.append(seed)

            # prediction after pertubation
            pred_perturb = model(X_perturb, edge_index, edge_weight)
            softmax_pred_perturb = torch.softmax(pred_perturb, dim=1)

            sample_bool = []
            for node in neighbors:
                if (softmax_pred_perturb[node, target] +
                        pred_threshold) < softmax_pred[node, target]:
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            samples.append(sample)
            pred_samples.append(sample_bool)

        samples = np.asarray(samples)
        pred_samples = np.asarray(pred_samples)
        combine_samples = (samples * 10 + pred_samples) + 1

        neighbors = np.array(neighbors.detach().cpu())
        data_pgm = pd.DataFrame(combine_samples)
        data_pgm = data_pgm.rename(columns={
            0: "A",
            1: "B"
        })  # Trick to use chi_square test on first two data columns
        index_original_to_subgraph = dict(
            zip(neighbors, list(data_pgm.columns)))
        index_subgraph_to_original = dict(
            zip(list(data_pgm.columns), neighbors))
        p_values = []

        dependent_neighbors = []
        dependent_neighbors_p_values = []
        for node in neighbors:
            if node == node_index:
                # null hypothesis is pertubing a particular
                # node has no effect on result
                p = 0
            else:
                chi2, p, _ = chi_square(
                    index_original_to_subgraph[node],
                    index_original_to_subgraph[node_index], [], data_pgm,
                    boolean=False, significance_level=significance_threshold)
            p_values.append(p)
            if p < significance_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)
        pgm_stats = dict(zip(neighbors, p_values))

        if max_subgraph_size is None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((max_subgraph_size, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [
                index_subgraph_to_original[node] for node in ind_top_p
            ]
        # todo not sure what this is doing?
        data_pgm = data_pgm.rename(columns={
            "A": 0,
            "B": 1
        }).rename(columns=index_subgraph_to_original)

        return pgm_nodes, pgm_stats

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.node, ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False
        return True
