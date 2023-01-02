import logging
from typing import List, Optional, Tuple

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
        perturb_feature_list (List): indicis of the perturbed features
             for graph classification explanations
        perturb_mode (str): which method to generate the variations in
            features one of ['randint', 'mean', 'zero', 'max', 'uniform']
        perturbations_is_positive_only (bool): whether to apply the
            abs function to restrict perturbed values to be +ve
        is_perturbation_scaled (bool): whether to normalise the range
            of the perturbed features
    """
    def __init__(
        self,
        perturb_feature_list: List = None,
        perturbation_mode: str = "randint",
        perturbations_is_positive_only: bool = False,
        is_perturbation_scaled: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.perturb_feature_list = perturb_feature_list
        self.perturbation_mode = perturbation_mode
        self.perturbations_is_positive_only = perturbations_is_positive_only
        self.is_perturbation_scaled = is_perturbation_scaled

    def perturb_features_on_nodes(
        self,
        feature_matrix: torch.Tensor,
        node_indexes: torch.Tensor,
    ) -> torch.Tensor:
        r"""
            perturb node feature matrix. This is used later to calculate
            how much influence neighbouring nodes has on the output
            Args:
                feature_matrix (torch.Tensor) : node feature matrix of
                    the input graph of shape [num_nodes, num_features]
                node_indexes (torch.Tensor): indexes of the nodes
                    that the perturbed features will be generated for
            Returns:
                a randomly perturbed feature matrix
            """
        X_perturb = feature_matrix.detach().clone()
        perturb_array = X_perturb[node_indexes].detach().clone()
        epsilon = 0.05 * torch.max(feature_matrix, dim=0).values

        if self.perturbation_mode == "randint":
            perturb_array = torch.randint(high=2, size=perturb_array.size(),
                                          device=feature_matrix.device)
        # graph explainers
        elif self.perturbation_mode == "mean":
            perturb_array[:, self.perturb_feature_list] = torch.mean(
                feature_matrix[:, self.perturb_feature_list])
        elif self.perturbation_mode == "zero":
            perturb_array[:, self.perturb_feature_list] = 0
        elif self.perturbation_mode == "max":
            perturb_array[:, self.perturb_feature_list] = torch.max(
                feature_matrix[:, self.perturb_feature_list])
        elif self.perturbation_mode == "uniform":
            random_perturbations = torch.rand(
                perturb_array.shape) * 2 * epsilon - epsilon
            perturb_array[:, self.perturb_feature_list] = perturb_array[
                self.perturb_feature_list] + random_perturbations
            perturb_array.clamp(min=0, max=torch.max(feature_matrix, dim=0))

        if self.is_perturbation_scaled:
            perturb_array = torch.multiply(
                perturb_array, torch.rand(size=perturb_array.size())) * 2

        X_perturb[node_indexes] = perturb_array.type(X_perturb.dtype)
        return X_perturb

    def batch_perturb_features_on_node(
            self,
            model: torch.nn.Module,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            num_samples: int,
            indices_to_perturb: np.array,
            percentage: float = 50.,  # % time node gets perturbed
            **kwargs) -> torch.Tensor:
        r"""
        perturb the node features of a batch of graphs
        for graph classification tasks
        Args:
            model (torch.nn.Module): graph neural net model
            x (torch.Tensor): node feature matrix
            edge_index (torch.Tensor): edge_index of the input graph
            num_samples: num of samples to generate for constructing the pgm
            indices_to_perturb (np.array): the index numbers of
                the nodes to be pertubed
            percentage:

        Returns:
            samples (torch.Tensor): the
        """
        pred_torch = model(x, edge_index, **kwargs)
        soft_pred = torch.softmax(pred_torch, dim=1)
        pred_label = torch.argmax(soft_pred, dim=1)
        num_nodes = x.shape[0]

        samples = []
        for iteration in range(num_samples):
            X_perturb = x.detach().clone()

            seeds = np.random.randint(0, 100, size=len(indices_to_perturb))
            perturbed_node_indexes = indices_to_perturb[(seeds < percentage)]
            X_perturb = self.perturb_features_on_nodes(
                feature_matrix=X_perturb,
                node_indexes=perturbed_node_indexes,
            )
            sample = np.zeros(num_nodes + 1)
            sample[perturbed_node_indexes] = 1

            pred_perturb_torch = model(X_perturb, edge_index, **kwargs)
            soft_pred_perturb = torch.softmax(pred_perturb_torch,
                                              dim=1).squeeze()

            pred_change = torch.max(soft_pred) - soft_pred_perturb[pred_label]

            sample[num_nodes] = pred_change
            samples.append(sample)

        samples = torch.tensor(np.array(samples))
        if self.perturbations_is_positive_only:
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
        edge_index: torch.Tensor,
        target=None,
        num_samples: int = 100,
        max_subgraph_size: int = None,
        significance_threshold: float = 0.05,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        generate explanations for graph classification tasks
        Args:
            model: pytorch model
            x (torch.Tensor): node features
            edge_index (torch.Tensor): edge_index of the input graph
            target(torch.Tensor):
            num_samples (int): number of samples to use to
                generate perturbations
            max_subgraph_size (int): max number of neighbours to consider
                for the explanation
            significance_threshold (float): threshold below which to consider
                a node has an effect on the prediction

        Returns:
            pgm_nodes (List): neighbour nodes that are significant
                 in the selected node's prediction
            pgm_stats (torch.Tensor): : p-values of all the nodes in the graph
                ordered by node index
        """
        num_nodes = x.shape[0]
        if not max_subgraph_size:
            max_subgraph_size = int(num_nodes / 20)

        samples = self.batch_perturb_features_on_node(
            num_samples=int(num_samples / 2),
            indices_to_perturb=np.array(range(num_nodes)), x=x, model=model,
            edge_index=edge_index, **kwargs)

        # note: the PC estimator is in the original code, ie. est= PC(data)
        # but as it does nothing it is not included here
        data = pd.DataFrame(np.array(samples.detach().cpu()))

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
        # it seems more correct to limit number_candidates_nodes to this
        candidate_nodes = np.argpartition(
            p_values, max_subgraph_size)[0:max_subgraph_size]

        # Round 2
        samples = self.batch_perturb_features_on_node(
            num_samples=num_samples, indices_to_perturb=candidate_nodes, x=x,
            edge_index=edge_index, model=model, **kwargs)

        # note: the PC estimator is in the original code, ie. est= PC(data)
        # but as it does nothing it is not included here
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

        node_mask = torch.zeros(x.size(), dtype=torch.int)
        node_mask[pgm_nodes] = 1
        pgm_stats = torch.tensor(p_values)

        return node_mask, pgm_stats

    def explain_node(
        self,
        model: torch.nn.Module,
        x: Tensor,
        node_index: int,
        edge_index: Tensor,
        edge_weight: Tensor,
        target: Tensor,
        num_samples: int = 100,
        max_subgraph_size: int = None,
        significance_threshold=0.05,
        pred_threshold=0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Generate explanations for node classification tasks
        Args:
            model (torch.nn.Module): model that generated the predictions
            x (torch.Tensor): node feature matrix
            node_index (torch.Tensor): the index of the node that the
                explanations are being generated for
            edge_index (torch.Tensor): edge_index of the input graph
            edge_weight (torch.Tensor): apply weighting to edges (Optional)
            target (torch.Tensor):  the prediction labels
            num_samples (int): num of samples of perturbations used to test
                the significance of neighbour nodes to the prediction
            max_subgraph_size:
            significance_threshold (float): threshold below which to consider
                a node has an effect on the prediction
            pred_threshold:

        Returns:
            node_mask (torch.Tensor): 1 or 0 corresponding to whether a node
                is significant in the selected node's prediction
            pgm_stats (torch.Tensor): p-values of all the nodes in the
                graph ordered by node index
        """
        logging.info(f'Explaining node: {node_index}')

        neighbors, edge_index_new, mapping, edge_mask_new = k_hop_subgraph(
            node_idx=node_index,
            num_hops=get_num_hops(model),
            edge_index=edge_index,
            relabel_nodes=False,
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
            # a subset of neighbours are selected randomly for perturbing
            seeds = np.random.choice([1, 0], size=(len(neighbors), ))
            X_perturb = self.perturb_features_on_nodes(
                feature_matrix=X_perturb,
                node_indexes=neighbors[seeds == 1],
            )

            # prediction after pertubation
            pred_perturb = model(X_perturb, edge_index, edge_weight)
            softmax_pred_perturb = torch.softmax(pred_perturb, dim=1)
            sample_bool = np.ones(shape=(len(neighbors), ))
            sample_bool[(
                (softmax_pred_perturb[neighbors, target] + pred_threshold) >=
                softmax_pred[neighbors, target]).cpu()] = 0

            samples.append(seeds)
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
                # null hypothesis is perturbing a particular
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

        pgm_stats = torch.ones(x.size(0), dtype=torch.float)
        node_mask = torch.zeros(x.size(), dtype=torch.int)

        pgm_stats[neighbors] = torch.tensor(p_values, dtype=torch.float)

        if max_subgraph_size is None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((max_subgraph_size, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [
                index_subgraph_to_original[node] for node in ind_top_p
            ]
        node_mask[pgm_nodes] = 1
        return node_mask, pgm_stats

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        target: Tensor,
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
            index (int): index of the node generating explanations for
            **kwargs:

        Returns:
            Explanation
        """
        if isinstance(index, Tensor):
            if index.numel() > 1:
                raise NotImplementedError(
                    f"'{self.__class__.__name}' only supports a single "
                    f"`index` for now")
            index = index.item()

        assert self.model_config.task_level in [
            ModelTaskLevel.graph, ModelTaskLevel.node
        ]

        edge_weight = kwargs.pop('edge_weight', None)
        num_samples = kwargs.pop('num_samples', 100)
        significance_threshold = kwargs.pop('significance_threshold', 0.05)
        max_subgraph_size = kwargs.get('max_subgraph_size', None)
        pred_threshold = kwargs.pop('pred_threshold', 0.1)
        if self.model_config.task_level == ModelTaskLevel.node:

            node_mask, pgm_stats = self.explain_node(
                model=model, x=x, node_index=index, edge_index=edge_index,
                edge_weight=edge_weight, target=target[index],
                num_samples=num_samples, max_subgraph_size=max_subgraph_size,
                significance_threshold=significance_threshold,
                pred_threshold=pred_threshold)
            return Explanation(
                x=x,
                edge_index=edge_index,
                node_mask=node_mask,
                pgm_stats=pgm_stats,
            )
        elif self.model_config.task_level == ModelTaskLevel.graph:
            node_mask, pgm_stats = self.explain_graph(
                model=model, x=x, target=target, edge_index=edge_index,
                num_samples=num_samples, max_subgraph_size=max_subgraph_size,
                significance_threshold=significance_threshold,
                pred_threshold=pred_threshold, **kwargs)
            return Explanation(
                node_mask=node_mask,
                pgm_stats=pgm_stats,
            )

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.node, ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False
        if self.explainer_config.edge_mask_type is not None:
            logging.error("Edge masks not supported by PGM explainer")
        return True
