import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from pgmpy.estimators.CITests import chi_square
from torch import Tensor

from torch_geometric.explain.config import ModelTaskLevel
from torch_geometric.explain.explanation import Explanation

from .base import ExplainerAlgorithm


class PGMExplainer(ExplainerAlgorithm):
    r"""The PGMExplainer model from the `"PGM-Explainer:
     Probabilistic Graphical Model Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/2010.05788>


    Args:

        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
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
        # edge_mask is only None if edge_mask_type is != MaskType.object

        print("Explainer settings")
        # print("Number of layers: ", self.num_layers)
        print("Perturbation mode: ", self.perturb_mode)

    def perturb_features_on_node(
            self,
            feature_matrix: torch.Tensor,
            node_idx: int,
            pertubation_mode="randint",
            is_random: bool = False,
            is_pertubation_scaled: bool = False,
            perturb_feature_list=None  # indexes of features being pertubed
    ):
        r"""
        pertub node feature matrix. This allows for checking how much influence
        neighbouring nodes has on the output
        Args:
            feature_matrix: node feature matrix of the input graph
                of shape [num_nodes, num_features]
            node_idx: index of the node we are calculating the explanation for
            pertubation_mode: how to pertube the features. must be one of
                [random, zero, mean, max]
            is_pertubation_scaled: whether to scale the pertubed matrix
                with the original feature matrix

        Returns:

        """
        # return a random perturbed feature matrix
        # random = is_random for nothing, True for random.randint.
        # is_pertubation_scaled=True, "scale" for scaling with original feature

        X_perturb = feature_matrix.detach().clone()
        perturb_array = X_perturb[node_idx].clone()
        epsilon = 0.05 * torch.max(feature_matrix, dim=0).values

        if is_random:
            if not is_pertubation_scaled:
                if pertubation_mode == "randint":
                    perturb_array = torch.randint(
                        high=2, size=X_perturb[node_idx].shape[0])
                # graph explainers
                elif pertubation_mode in ("mean", "zero", "max", "uniform"):
                    for i in range(perturb_array.shape[0]):
                        if i in perturb_feature_list:
                            if pertubation_mode == "mean":
                                perturb_array[i] = torch.mean(
                                    feature_matrix[:, i])
                            elif pertubation_mode == "zero":
                                perturb_array[i] = 0
                            elif pertubation_mode == "max":
                                perturb_array[i] = torch.max(feature_matrix[:,
                                                                            i])
                            elif pertubation_mode == "uniform":
                                perturb_array[
                                    i] = perturb_array[i] + torch.tensor(
                                        np.random.uniform(
                                            low=-epsilon[i], high=epsilon[i]))
                                if perturb_array[i] < 0:
                                    perturb_array[i] = 0
                                elif perturb_array[i] > torch.max(
                                        feature_matrix, dim=0)[i]:
                                    perturb_array[i] = torch.max(
                                        feature_matrix, dim=0)[i]

            elif is_pertubation_scaled:
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
        # edge_weight: Tensor,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        index: Optional[int] = None,  # node index
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

        assert self.model_config.task_level in [
            ModelTaskLevel.graph, ModelTaskLevel.node
        ]

        model.eval()

        edge_weight = kwargs.get('edge_weight')
        num_samples = kwargs.get('num_samples', 100)
        significance_threshold = kwargs.get('significance_threshold', 0.05)
        top_node = kwargs.get('top_node')
        pred_threshold = kwargs.get('pred_threshold', 0.1)
        if self.model_config.task_level == ModelTaskLevel.node:

            neighbors, pgm_stats = self._explain_node(
                model, x, index, edge_index, edge_weight, target[index],
                num_samples=num_samples, top_node=top_node,
                significance_threshold=significance_threshold,
                pred_threshold=pred_threshold**kwargs)
        elif self.model_config.task_level == ModelTaskLevel.graph:
            pgm_nodes, p_values, candidate_nodes = self._explain_graph(
                model=model, x=x, index=index, target=target,
                num_samples=num_samples, top_node=top_node,
                significance_threshold=significance_threshold,
                pred_threshold=pred_threshold, **kwargs)
            return Explanation(node_mask=p_values < significance_threshold)

        # if explainer_config.node_mask_type == MaskType.object:
        #     node_feat_mask = None
        # else:
        #     node_feat_mask = node_mask
        #     node_mask = None

        # build explanation
        return Explanation(
            x=x,
            edge_index=edge_index,
            markov_blanket=neighbors,
            # pgm_explation=pgm_explanation,
            # marginal_probability=marginal_prob,
            pgm_stats=pgm_stats,
        )

    def batch_perturb_features_on_node(
            self,
            num_samples,
            index_to_perturb,
            X_features,
            model,
            percentage=50,  # % time node gets pertubed
            p_threshold=0.05,
            pred_threshold=0.1,
            pertubation_mode="mean",
            snorm_node=None,
            snorm_edge=None,
            **kwargs):
        # for pertubing a batch of graphs for graph classification tasks

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
                            X_perturb, node, pertubation_mode=pertubation_mode,
                            perturb_feature_list=self.perturb_feature_list,
                            is_random=True)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)
            # if data:
            #     data.x = X_perturb.detach().clone()
            # kwargs['data'] = data
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

    def _explain_graph(
        self,
        model,
        x,
        index,
        target=None,
        num_samples=100,
        top_node=None,  # num of neightbours to consider
        significance_threshold=0.05,
        pred_threshold=0.1,
        percentage=0.5,
        pertubation_mode="mean",
        **kwargs,
    ):
        model.eval()
        num_nodes = x.shape[0]
        if not top_node:
            top_node = int(num_nodes / 20)

        #         Round 1

        Samples = self.batch_perturb_features_on_node(
            num_samples=int(num_samples / 2),
            index_to_perturb=range(num_nodes), X_features=x, model=model,
            pertubation_mode=pertubation_mode, **kwargs)

        data = pd.DataFrame(np.array(Samples.detach().cpu()))
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

        number_candidates = int(top_node * 4)
        candidate_nodes = np.argpartition(
            p_values, number_candidates)[0:number_candidates]

        #         Round 2
        Samples = self.batch_perturb_features_on_node(
            num_samples=num_samples, index_to_perturb=candidate_nodes,
            X_features=x, model=model, pertubation_mode=pertubation_mode,
            **kwargs)

        # todo the PC estimator is in the code but it does nothing??
        data = pd.DataFrame(np.array(Samples.detach().cpu()))
        # est = PC(data)

        p_values = []
        dependent_nodes = []

        target = num_nodes
        for node in range(num_nodes):
            chi2, p, _ = chi_square(node, target, [], data, boolean=False,
                                    significance_level=significance_threshold)
            p_values.append(p)
            if p < significance_threshold:
                dependent_nodes.append(node)

        top_p = np.min((top_node, num_nodes - 1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)

        return pgm_nodes, torch.tensor(p_values), candidate_nodes

    def _explain_node(
            self,
            model: torch.nn.Module,
            x: Tensor,  # [num_rows x num_features]
            node_index: int,
            edge_index: Tensor,
            edge_weight: Tensor,
            target: Tensor,
            num_samples: int = 100,
            top_node: int = None,
            significance_threshold=0.05,
            pred_threshold=0.1):
        logging.info(f'Explaining node: {node_index}')

        x_new, edge_index_new, mapping, \
            neighbors, edge_mask_new, kwargs = self.subgraph(
                model=model, node_idx=node_index, x=x, edge_index=edge_index)

        # neighbors = neighbors.cpu().detach().numpy()

        if node_index not in neighbors:
            neighbors = torch.cat([neighbors, node_index], dim=1)

        pred_model = model(x, edge_index, edge_weight)

        softmax_pred = torch.softmax(pred_model, dim=1)
        # prediction of the node we want to get explanation for
        # prediction_selected_node = pred_model[node_index].data
        # label_node = torch.argmax(prediction_selected_node)
        # softmax_prediction_selected_node = softmax_pred[node_index]

        Samples = []
        Pred_Samples = []

        for iteration in range(num_samples):
            X_perturb = x.detach().clone()

            sample = []
            for node in neighbors:
                seed = np.random.choice([1, 0])
                if seed == 1:
                    pertubation_mode = "random"
                else:
                    pertubation_mode = "zero"
                X_perturb = self.perturb_features_on_node(
                    X_perturb, node, pertubation_mode=pertubation_mode)
                sample.append(seed)

            # X_perturb = X_perturb.to(model.device)
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

            Samples.append(sample)
            Pred_Samples.append(sample_bool)

        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = (Samples * 10 + Pred_Samples) + 1

        neighbors = np.array(neighbors.detach().cpu())
        data_pgm = pd.DataFrame(Combine_Samples)
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

        if top_node is None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((top_node, len(neighbors) - 1))
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

    def supports(
        self,
        # explainer_config: ExplainerConfig,
        # model_config: ModelConfig,
    ) -> bool:
        # if model_config.task_level not in [
        #     ModelTaskLevel.node, ModelTaskLevel.graph
        # ]:
        #     logging.error("Model task level not supported.")
        #     return False

        return True
