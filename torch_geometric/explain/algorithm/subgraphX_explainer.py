import copy
import logging
import math
from collections import Counter
from functools import partial
from os.path import isfile, join
from typing import Callable, Dict, List, Optional, Union

import networkx as nx
import torch
from torch import Tensor

from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.config import ModelTaskLevel
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import (
    k_hop_subgraph,
    remove_self_loops,
    to_networkx,
)

from .base import ExplainerAlgorithm
from .subgraphX_utils import (
    GnnNetsGC2valueFunc,
    GnnNetsNC2valueFunc,
    find_closest_node_result,
    gnn_score,
    reward_func,
    sparsity,
)


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results


class MCTSNode(object):
    def __init__(
        self,
        coalition: list = None,
        data: Data = None,
        ori_graph: nx.Graph = None,
        c_puct: float = 10.0,
        W: float = 0,
        N: int = 0,
        P: float = 0,
        load_dict: Optional[Dict] = None,
        device="cpu",
    ):
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.device = device
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)
        if load_dict is not None:
            self.load_info(load_dict)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        info_dict = {
            "data": self.data.to("cpu"),
            "coalition": self.coalition,
            "ori_graph": self.ori_graph,
            "W": self.W,
            "N": self.N,
            "P": self.P,
        }
        return info_dict

    def load_info(self, info_dict):
        self.W = info_dict["W"]
        self.N = info_dict["N"]
        self.P = info_dict["P"]
        self.coalition = info_dict["coalition"]
        self.ori_graph = info_dict["ori_graph"]
        self.data = info_dict["data"].to(self.device)
        self.children = []
        return self


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method.

    Args:
        X (:obj:`torch.Tensor`): Input node features
        edge_index (:obj:`torch.Tensor`): The edge indices.
        num_hops (:obj:`int`): Number of hops :math:`k`.
        n_rollout (:obj:`int`): Number of sequences to build the
            monte carlo tree.
        min_atoms (:obj:`int`): Number of atoms for the subgraph
            in the monte carlo tree leaf node.
        c_puct (:obj:`float`): Hyper-parameter to encourage
            exploration while searching.
        expand_atoms (:obj:`int`): Number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node
            from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract
            the neighborhood.
        score_func (:obj:`Callable`): The reward function for
            tree node, such as mc_shapely and mc_l_shapely.
    """

    def __init__(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        num_hops: int,
        n_rollout: int = 10,
        min_atoms: int = 3,
        c_puct: float = 10.0,
        expand_atoms: int = 14,
        high2low: bool = False,
        node_idx: int = None,
        score_func: Callable = None,
        device="cpu",
    ):
        self.X = X
        self.edge_index = edge_index
        self.device = device
        self.num_hops = num_hops
        self.data = Data(x=self.X, edge_index=self.edge_index)
        graph_data = Data(x=self.X, edge_index=remove_self_loops(self.edge_index)[0])
        self.graph = to_networkx(graph_data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low
        self.new_node_idx = None

        # extract the sub-graph and change the node indices.
        if node_idx is not None:
            if isinstance(node_idx, torch.Tensor):
                node_idx = node_idx.item()

            self.ori_node_idx = node_idx
            self.ori_graph = copy.copy(self.graph)
            x, edge_index, subset, edge_mask, kwargs = self.__subgraph__(
                node_idx, self.X, self.edge_index, self.num_hops
            )
            self.data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
            self.graph = self.ori_graph.subgraph(subset.tolist())
            mapping = {int(v): k for k, v in enumerate(subset)}
            self.graph = nx.relabel_nodes(self.graph, mapping)
            self.new_node_idx = torch.where(subset == self.ori_node_idx)[0].item()
            self.num_nodes = self.graph.number_of_nodes()
            self.subset = subset

        self.root_coalition = sorted([node for node in range(self.num_nodes)])
        self.MCTSNodeClass = partial(
            MCTSNode,
            data=self.data,
            ori_graph=self.graph,
            c_puct=self.c_puct,
            device=self.device,
        )
        self.root = self.MCTSNodeClass(self.root_coalition)
        self.state_map = {str(self.root.coalition): self.root}

    def set_score_func(self, score_func):
        self.score_func = score_func

    @staticmethod
    def __subgraph__(node_idx, x, edge_index, num_hops, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, subset, edge_mask, kwargs

    def mcts_rollout(self, tree_node):
        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(
                node_degree_list, key=lambda x: x[1], reverse=self.high2low
            )
            all_nodes = [x[0] for x in node_degree_list]

            if self.new_node_idx:
                expand_nodes = [node for node in all_nodes if node != self.new_node_idx]
            else:
                expand_nodes = all_nodes

            if len(all_nodes) > self.expand_atoms:
                expand_nodes = expand_nodes[: self.expand_atoms]

            for each_node in expand_nodes:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and
                # only keep the largest one
                subgraph_coalition = [node for node in all_nodes if node != each_node]

                subgraphs = [
                    self.graph.subgraph(c)
                    for c in nx.connected_components(
                        self.graph.subgraph(subgraph_coalition)
                    )
                ]

                if self.new_node_idx:
                    for sub in subgraphs:
                        if self.new_node_idx in list(sub.nodes()):
                            main_sub = sub
                else:
                    main_sub = subgraphs[0]

                    for sub in subgraphs:
                        if sub.number_of_nodes() > main_sub.number_of_nodes():
                            main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))

                # check the state map and merge the same sub-graph
                find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(
                        new_graph_coalition
                    ):
                        new_node = old_graph_node
                        find_same = True

                if not find_same:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        find_same_child = True

                if not find_same_child:
                    tree_node.children.append(new_node)

            scores = compute_scores(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(
                    f"At the {rollout_idx} rollout, {len(self.state_map)}"
                    "states that have been explored."
                )

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


class SubgraphXExplainer(ExplainerAlgorithm):
    r"""The SubgraphX explainer model from the `"On Explainability
    of Graph Neural Networks via Subgraph Explorations"

    Paper: https://arxiv.org/abs/2102.05152
    Official Implementation Repo: \
        https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/subgraphx.py

    .. note::
        For an example of using SubgraphXExplainer, see
        `examples/subgraphx_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        subgraphx_explainer.py>`_.

    Args:
        device: Device to generate the explanations on.
        local_radius(:obj:`int`): Local radius to be considered while
            evaluating subgraph importance for :obj:`l_shapley`,
            :obj:`mc_l_shapley`
        sample_num(:obj:`int`): Sampling time of monte carlo sampling
            approximation for :obj:`mc_shapley`, :obj:`mc_l_shapley`
            (default: :obj:`mc_l_shapley`)
        reward_method(:obj:`str`): Reward method to assign subgraph
            importance.
            One of [
                "gnn_score", "mc_shapley", "l_shapley",
                "mc_l_shapley", "nc_mc_l_shapley"
            ]
        subgraph_building_method(:obj:`str`): Specifies way to fill
            subgraph. One of ["zero_filling", "split"]
        save_dir(:obj:`str`, :obj:`None`): Root directory to save
        the explanation results (default: :obj:`None`)
        filename(:obj:`str`): The filename of results
        TODO: Fill this.
    """

    def __init__(
        self,
        num_classes: int,
        max_nodes: int = 5,
        num_hops: int = None,
        MCTS_info_path: str = None,
        device: str = "cpu",
        verbose: bool = True,
        local_radius: int = 4,
        sample_num=100,
        reward_method: str = "mc_l_shapley",
        subgraph_building_method: str = "zero_filling",
        rollout: int = 20,
        min_atoms: int = 5,
        c_puct: float = 10.0,
        expand_atoms: int = 14,
        high2low: bool = False,
        save_dir: str = None,
        filename: str = "mcts_results",
    ):
        super().__init__()
        self.device = device
        self.MCTS_info_path = MCTS_info_path
        self.num_hops = num_hops
        self.num_classes = num_classes
        self.verbose = verbose
        self.max_nodes = max_nodes

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        # reward function hyper-parameters
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.reward_method = reward_method
        self.subgraph_building_method = subgraph_building_method

        # saving and visualization
        self.save_dir = save_dir
        self.filename = filename
        self.save = True if self.save_dir is not None else False

    def update_num_hops(self, model, num_hops):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def read_from_MCTSInfo_list(self, MCTSInfo_list):
        """TODO: Complete Docstrings for Readability"""
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [
                MCTSNode(device=self.device).load_info(node_info)
                for node_info in MCTSInfo_list
            ]
        elif isinstance(MCTSInfo_list[0][0], dict):
            ret_list = []
            for single_label_MCTSInfo_list in MCTSInfo_list:
                single_label_ret_list = [
                    MCTSNode(device=self.device).load_info(node_info)
                    for node_info in single_label_MCTSInfo_list
                ]
                ret_list.append(single_label_ret_list)
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        """TODO: Complete Docstrings for Readability"""
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        elif isinstance(MCTSNode_list[0][0], MCTSNode):
            ret_list = []
            for single_label_MCTSNode_list in MCTSNode_list:
                single_label_ret_list = [
                    node.info for node in single_label_MCTSNode_list
                ]
                ret_list.append(single_label_ret_list)
        return ret_list

    def get_reward_func(self, value_func, node_idx=None):
        """Runs `__call__` on the `reward_method`"""
        if self.model_config.task_level == ModelTaskLevel.graph:
            node_idx = None

        return reward_func(
            reward_method=self.reward_method,
            value_func=value_func,
            node_idx=node_idx,
            local_radius=self.local_radius,
            sample_num=self.sample_num,
            subgraph_building_method=self.subgraph_building_method,
        )

    def get_mcts_class(
        self, x, edge_index, node_idx: int = None, score_func: Callable = None
    ):
        if self.model_config.task_level == ModelTaskLevel.graph:
            node_idx = None

        return MCTS(
            x,
            edge_index,
            node_idx=node_idx,
            device=self.device,
            score_func=score_func,
            num_hops=self.num_hops,
            n_rollout=self.rollout,
            min_atoms=self.min_atoms,
            c_puct=self.c_puct,
            expand_atoms=self.expand_atoms,
            high2low=self.high2low,
        )

    def explain(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        label: int,
        max_nodes: int = 5,
        node_idx: Optional[int] = None,
        saved_MCTSInfo_list: Optional[List[List]] = None,
        **kwargs,
    ):
        # get task classification probabilities
        with torch.no_grad():
            probs = model(x, edge_index, **kwargs).softmax(dim=-1)

        # load MCTSInfo_list if present
        if saved_MCTSInfo_list:
            results = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)

        if self.model_config.task_level == ModelTaskLevel.graph:
            # Explanation for Graph Classification Task
            if not saved_MCTSInfo_list:
                value_func = GnnNetsGC2valueFunc(model, target_class=label)
                payoff_func = self.get_reward_func(value_func)
                self.mcts_state_map = self.get_mcts_class(
                    x, edge_index, score_func=payoff_func
                )
                results = self.mcts_state_map.mcts(verbose=self.verbose)

            # l sharply score
            value_func = GnnNetsGC2valueFunc(model, target_class=label)

        else:
            # get label of `node_idx`
            label = label[node_idx]

            # Explanation for Node Classification Task
            self.mcts_state_map = self.get_mcts_class(x, edge_index, node_idx=node_idx)
            self.new_node_idx = self.mcts_state_map.new_node_idx
            # mcts will extract the subgraph and relabel the nodes
            value_func = GnnNetsNC2valueFunc(
                model,
                node_idx=self.mcts_state_map.new_node_idx,
                target_class=label,
            )

            if not saved_MCTSInfo_list:
                payoff_func = self.get_reward_func(
                    value_func, node_idx=self.mcts_state_map.new_node_idx
                )
                self.mcts_state_map.set_score_func(payoff_func)
                results = self.mcts_state_map.mcts(verbose=self.verbose)

        # get the top result based on our node
        tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)

        # keep the important structure
        masked_node_list = [
            node
            for node in range(tree_node_x.data.x.shape[0])
            if node in tree_node_x.coalition
        ]

        # remove the important structure, for node_classification,
        # remain the node_idx when remove the important structure
        maskout_node_list = [
            node
            for node in range(tree_node_x.data.x.shape[0])
            if node not in tree_node_x.coalition
        ]
        if not self.model_config.task_level == ModelTaskLevel.graph:
            maskout_node_list += [self.new_node_idx]

        masked_score = gnn_score(
            masked_node_list,
            tree_node_x.data,
            value_func=value_func,
            subgraph_building_method=self.subgraph_building_method,
        )

        maskout_score = gnn_score(
            maskout_node_list,
            tree_node_x.data,
            value_func=value_func,
            subgraph_building_method=self.subgraph_building_method,
        )

        sparsity_score = sparsity(
            masked_node_list,
            tree_node_x.data,
            subgraph_building_method=self.subgraph_building_method,
        )

        results = self.write_from_MCTSNode_list(results)
        node_idx = 0 if node_idx is None else node_idx
        related_pred = {
            "masked": masked_score,
            "maskout": maskout_score,
            "origin": probs[node_idx, label].item(),
            "sparsity": sparsity_score,
        }

        return results, related_pred, masked_node_list

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Optional[Tensor] = None,
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[int] = None,
        **kwargs,
    ) -> Explanation:
        """Computes explaination based on SubgraphX Explainer

        Args:
            model (torch.nn.Module): The model to explain.
            x (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The input
                node features of a homogeneous or heterogeneous graph.
            edge_index (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The
                input edge indices of a homogeneous or heterogeneous graph.
            target (Optional[torch.Tensor]): The target of the model.
            index (Union[int, Tensor], optional): The index of the model
                output to explain. Can be a single index or a tensor of
                indices. (default: :obj:`None`)
            target_index (int, optional): The index of the model outputs to
                reference in case the model returns a list of tensors, *e.g.*,
                in a multi-task learning scenario. Should be kept to
                :obj:`None` in case the model only returns a single output
                tensor. (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        # set model to eval mode
        model.eval()

        # update num_hops if not provided
        self.num_hops = self.update_num_hops(model, self.num_hops)

        # check if MCTS_info_list has been provided, load if present
        saved_results = None
        if self.save:
            # generate file_path
            file_path = join(self.save_dir, f"{self.filename}.pt")
            # if file_path exists, load the saved_results
            if isfile(file_path):
                saved_results = torch.load(file_path)

        if self.model_config.task_level == ModelTaskLevel.node:
            # check if index has been provided
            assert (
                index is not None
            ), "For Node Classification, index (node_idx) must be provided"

        # get explanation for that index and the prediction label
        results, related_pred, masked_node_list = self.explain(
            model,
            x,
            edge_index,
            label=target,
            max_nodes=self.max_nodes,
            node_idx=index,
            saved_MCTSInfo_list=saved_results,
            **kwargs,
        )

        # if self.save is True, save the explanation results
        if self.save:
            # generate file_path
            file_path = join(self.save_dir, f"{self.filename}.pt")
            torch.save(results, file_path)

        # create node_mask from masked_node_list
        node_mask = torch.zeros(size=(x.size()[0],)).float()
        node_mask[masked_node_list] = 1.0
        # squeeze() to make it 2-dimensional
        node_mask = node_mask.unsqueeze(1)

        # create edge_mask from masked_node_list
        explained_edge_list = []
        edge_mask_indices = []
        for i in range(edge_index.size()[1]):
            (n_frm, n_to) = edge_index[:, i].tolist()
            if n_frm in masked_node_list and n_to in masked_node_list:
                explained_edge_list.append((n_frm, n_to))
                edge_mask_indices.append(i)

        edge_mask = torch.zeros(size=(edge_index.size()[1],)).float()
        edge_mask[edge_mask_indices] = 1.0

        # create explanation with additional args
        explanation = Explanation(
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            results=results,
            related_pred=related_pred,
            masked_node_list=masked_node_list,
            explained_edge_list=explained_edge_list,
        )

        return explanation

    def supports(self) -> bool:
        """SubgraphXExplainer only supports Node and Graph Classification"""
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.node, ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False

        return True
