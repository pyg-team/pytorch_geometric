import logging
from scipy.special import softmax
from typing import Optional, Tuple, Union, Dict
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import pandas as pd
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
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import k_hop_subgraph

from scipy.special import softmax
# from pgmpy.estimators import PC as ConstraintBasedEstimator
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from scipy import stats
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
            # num_layers: int = None, # part of model
            perturb_feature_list=None,
            perturb_mode="mean",  # mean, zero, max or uniform
            perturb_indicator="diff",  # diff or abs
            snorm_n=None,
            snorm_e=None,
            **kwargs,
    ):
        super().__init__()
        # self.num_layers = num_layers
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        # edge_mask is only None if edge_mask_type is != MaskType.object

        print("Explainer settings")
        # print("Number of layers: ", self.num_layers)
        print("Perturbation mode: ", self.perturb_mode)

    def perturb_features_on_node(self, feature_matrix, node_idx, is_random=False, is_pertubation_scaled=False):
        # return a random perturbed feature matrix
        # random = is_random for nothing, True for random.randint.
        # is_pertubation_scaled=True, "scale" for scaling with original feature

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
                                               torch.rand(size=X_perturb[node_idx].shape[0])) * 2
            X_perturb[node_idx] = perturb_array
        return X_perturb

    def forward(
            self,
            model: torch.nn.Module,
            x: Tensor,
            edge_index: Tensor,
            # edge_weight: Tensor,
            explainer_config: ExplainerConfig,
            model_config: ModelConfig,
            target: Tensor,
            target_index: Optional[Union[int, Tensor]] = None,
            index: Optional[int] = None, # node index
            **kwargs,
    ) -> Explanation:

        assert model_config.task_level in [
            ModelTaskLevel.graph, ModelTaskLevel.node
        ]

        model.eval()
        node_feature_mask = None
        node_mask = None
        edge_mask = None
        # edge_index = None

        edge_weight = kwargs.get('edge_weight')
        if model_config.task_level == ModelTaskLevel.node:

            node_mask, node_feature_mask = self._explain_node(
                model,x, index,
            edge_index,
            edge_weight,
            explainer_config,
            model_config,
            target,
            num_samples= 100,
                               top_node = None,
                                          significance_threshold = 0.05,
                                                                   pred_threshold = 0.1)
            print(node_feature_mask, 'here')
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
                           node_mask=node_mask, node_feat_mask=node_feature_mask)

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

    def batch_perturb_features_on_node(self, num_samples, index_to_perturb,
        X_features, edge_features, percentage, p_threshold, pred_threshold):
        X_torch = torch.tensor(X_features, dtype=torch.float)
        E_torch = torch.tensor(self.edge_features, dtype=torch.float)

        pred_torch = self.model.forward(self.graph, X_torch, E_torch, self.snorm_n, self.snorm_e)
        soft_pred = np.asarray(softmax(np.asarray(pred_torch[0].data)))
        pred_label = np.argmax(soft_pred)
        num_nodes = self.X_feat.shape[0]
        Samples = []
        for iteration in range(num_samples):
            X_perturb = self.X_feat.copy()
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(X_perturb, node, random=latent)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float)
            pred_perturb_torch = self.model.forward(self.graph, X_perturb_torch, E_torch, self.snorm_n, self.snorm_e)
            soft_pred_perturb = np.asarray(softmax(np.asarray(pred_perturb_torch[0].data)))

            pred_change = np.max(soft_pred) - soft_pred_perturb[pred_label]

            sample.append(pred_change)
            Samples.append(sample)

        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)

        top = int(num_samples / 8)
        top_idx = np.argsort(Samples[:, num_nodes])[-top:]
        for i in range(num_samples):
            if i in top_idx:
                Samples[i, num_nodes] = 1
            else:
                Samples[i, num_nodes] = 0

        return Samples

    def generalize_target(self, x):
        if x > 10:
            return x - 10
        else:
            return x

    def generalize_others(self, x):
        if x == 2:
            return 1
        elif x == 12:
            return 11
        else:
            return x

    def generate_evidence(self, evidence_list):
        return dict(zip(evidence_list, [1 for node in evidence_list]))

    def chi_square(self, X, Y, Z, data):
        """
        Modification of Chi-square conditional independence test from pgmpy
        Tests the null hypothesis that X is independent from Y given Zs.
        Parameters
        ----------
        X: int, string, hashable object
            A variable name contained in the data set
        Y: int, string, hashable object
            A variable name contained in the data set, different from X
        Zs: list of variable names
            A list of variable names contained in the data set, different from X and Y.
            This is the separating set that (potentially) makes X and Y independent.
            Default: []
        Returns
        -------
        chi2: float
            The chi2 test statistic.
        p_value: float
            The p_value, i.e. the probability of observing the computed chi2
            statistic (or an even higher value), given the null hypothesis
            that X _|_ Y | Zs.
        sufficient_data: bool
            A flag that indicates if the sample size is considered sufficient.
            As in [4], require at least 5 samples per parameter (on average).
            That is, the size of the data set must be greater than
            `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
            (c() denotes the variable cardinality).
        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.2.2.3 (page 789)
        [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
        [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4
        """
        X = str(int(X))
        Y = str(int(Y))
        if isinstance(Z, (frozenset, list, set, tuple)):
            Z = list(Z)
        Z = [str(int(z)) for z in Z]

        state_names = {
            var_name: data.loc[:, var_name].unique() for var_name in data.columns
        }

        row_index = state_names[X]
        column_index = pd.MultiIndex.from_product(
            [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
        )

        XYZ_state_counts = pd.crosstab(
            index=data[X], columns=[data[Y]] + [data[z] for z in Z],
            rownames=[X], colnames=[Y] + Z
        )

        if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
            XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
        XYZ_state_counts = XYZ_state_counts.reindex(
            index=row_index, columns=column_index
        ).fillna(0)

        if Z:
            XZ_state_counts = XYZ_state_counts.sum(axis=1, level=list(range(1, len(Z) + 1)))  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Z)  # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)
            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both

        XYZ_expected = np.zeros(XYZ_state_counts.shape)

        r_index = 0
        for X_val in XYZ_state_counts.index:
            X_val_array = []
            if Z:
                for Y_val in XYZ_state_counts.columns.levels[0]:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + list(temp.to_numpy())
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index = +1
            else:
                for Y_val in XYZ_state_counts.columns:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + [temp]
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index = +1

        observed = XYZ_state_counts.to_numpy().reshape(1, -1)
        expected = XYZ_expected.reshape(1, -1)
        observed, expected = zip(*((o, e) for o, e in zip(observed[0], expected[0]) if not (e == 0 or np.isnan(e))))
        chi2, significance_level = stats.chisquare(observed, expected)

        return chi2, significance_level

    def search_MK(self, data, target, nodes):
        target = str(int(target))
        data.columns = data.columns.astype(str)
        nodes = [str(int(node)) for node in nodes]

        MB = nodes
        while True:
            count = 0
            for node in nodes:
                evidences = MB.copy()
                evidences.remove(node)
                _, p = self.chi_square(target, node, evidences, data[nodes + [target]])
                if p > 0.05:
                    MB.remove(node)
                    count = 0
                else:
                    count = count + 1
                    if count == len(MB):
                        return MB

    def pgm_generate(self, target, data, pgm_stats, subnodes, child=None):

        subnodes = [str(int(node)) for node in subnodes]
        target = str(int(target))
        subnodes_no_target = [node for node in subnodes if node != target]
        data.columns = data.columns.astype(str)

        MK_blanket = self.search_MK(data, target, subnodes_no_target.copy())

        if child == None:
            est = HillClimbSearch(data[subnodes_no_target], scoring_method=BicScore(data))
            pgm_no_target = est.estimate()
            for node in MK_blanket:
                if node != target:
                    pgm_no_target.add_edge(node, target)

            #   Create the pgm
            pgm_explanation = BayesianModel()
            for node in pgm_no_target.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_no_target.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)
        else:
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)

            est = HillClimbSearch(data_ex, scoring_method=BicScore(data_ex))
            pgm_w_target_explanation = est.estimate()

            #   Create the pgm
            pgm_explanation = BayesianModel()
            for node in pgm_w_target_explanation.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            #   Fit the pgm
            data_ex = data[subnodes].copy()
            data_ex[target] = data[target].apply(self.generalize_target)
            for node in subnodes_no_target:
                data_ex[node] = data[node].apply(self.generalize_others)
            pgm_explanation.fit(data_ex)

        return pgm_explanation

    def pgm_conditional_prob(self, target, pgm_explanation, evidence_list):
        pgm_infer = VariableElimination(pgm_explanation)
        for node in evidence_list:
            if node not in list(pgm_infer.variables):
                print("Not valid evidence list.")
                return None
        evidences = self.generate_evidence(evidence_list)
        elimination_order = [node for node in list(pgm_infer.variables) if node not in evidence_list]
        elimination_order = [node for node in elimination_order if node != target]
        q = pgm_infer.query([target], evidence=evidences,
                            elimination_order=elimination_order, show_progress=False)
        return q.values[0]

    def get_model_layers(self, model):
        r"""
        get the number of message passing layers in the model-- in PGM only the nodes reachable by
        message passing is considered for the explanation
        Args:
            model:

        Returns:

        """
        return [module for module in model.modules() if not isinstance(module, MessagePassing)]

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
            x: Tensor, # [num_rows x num_features]
            node_index: int,
            edge_index: Tensor,
            edge_weight: Tensor,
            explainer_config: ExplainerConfig,
            model_config: ModelConfig,
            target: Tensor,
            num_samples: int = 100,
            top_node=None,
            significance_threshold=0.05,
            pred_threshold=0.1

    ) -> tuple[torch.Tensor, torch.Tensor]:
        logging.info(f'Explaining node: {node_index}')
        num_hops = self.get_model_layers(model)
        neighbors, edge_index_new, mapping, edge_mask = k_hop_subgraph(node_index, num_hops=num_hops,
                                                                   edge_index=edge_index, relabel_nodes=True)

        neighbors = neighbors.cpu().detach().numpy()

        if node_index not in neighbors:
            neighbors = np.append(neighbors, node_index)

        pred_model = target.cpu()

        softmax_pred = np.asarray([softmax(np.asarray(pred_model[node_].data)) for node_ in range(x.shape[0])])
        pred_single_node = pred_model[node_index].data
        # label_node = torch.argmax(pred_single_node)
        # soft_pred_single_node = torch.softmax(pred_single_node)

        Samples = []
        Pred_Samples = []

        for iteration in range(num_samples):

            X_perturb = x.cpu().detach().numpy()
            sample = []
            for node in neighbors:
                seed = np.random.choice([1, 0])
                if seed == 1:
                    X_perturb = self.perturb_features_on_node(X_perturb, node, is_random=True)
                sample.append(seed)

            X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float).to(edge_index.device)
            pred_perturb_torch = model(X_perturb_torch, edge_index, edge_weight).cpu()
            softmax_pred_perturb = np.asarray(
                [softmax(np.asarray(pred_perturb_torch[node_].data)) for node_ in range(x.shape[0])]
            )

            sample_bool = []
            for node in neighbors:
                if (softmax_pred_perturb[node, np.argmax(softmax_pred[node])] + pred_threshold) < np.max(softmax_pred[node]):
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            Samples.append(sample)
            Pred_Samples.append(sample_bool)

        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = Samples - Samples
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])]
            )

        data_pgm = pd.DataFrame(Combine_Samples)
        data_pgm = data_pgm.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(neighbors, list(data_pgm.columns)))
        ind_sub_to_ori = dict(zip(list(data_pgm.columns), neighbors))
        p_values = []

        dependent_neighbors = []
        dependent_neighbors_p_values = []
        for node in neighbors:
            if node == node_index:
                p = 0  # p<0.05 => we are confident that we can reject the null hypothesis (i.e. the prediction is the same after perturbing the neighbouring node
                # => this neighbour has no influence on the prediction - should not be in the explanation)
            else:
                chi2, p, _ = chi_square(
                    ind_ori_to_sub[node], ind_ori_to_sub[node_index], [],
                    data_pgm, boolean=False, significance_level=significance_threshold
                )
            p_values.append(p)
            if p < significance_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)

        pgm_stats = dict(zip(neighbors, p_values))

        if top_node == None:
            pgm_nodes = dependent_neighbors
        else:
            top_p = np.min((top_node, len(neighbors) - 1))
            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
            pgm_nodes = [ind_sub_to_ori[node] for node in ind_top_p]
        node_mask = torch.Tensor(neighbors)
        node_feature_mask = torch.Tensor(list(zip(neighbors, p_values)))

        data_pgm = data_pgm.rename(columns={"A": 0, "B": 1})
        data_pgm = data_pgm.rename(columns=ind_sub_to_ori)

        return node_mask, node_feature_mask, pgm_stats, pgm_nodes, data_pgm

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
