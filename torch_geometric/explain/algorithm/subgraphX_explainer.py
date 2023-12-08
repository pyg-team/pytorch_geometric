import logging
from os.path import isfile, join
from typing import Callable, Dict, List, Optional, Union, Tuple

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.config import ModelTaskLevel
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import subgraph

from .subgraphX_mcts import MCTS, MCTSNode
from .base import ExplainerAlgorithm
from .subgraphX_utils import (
    GnnNetsGC2valueFunc,
    GnnNetsNC2valueFunc,
    find_closest_node_result,
    gnn_score,
    reward_func,
    sparsity,
)


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
    """

    def __init__(
        self,
        num_classes: int,
        max_nodes: int = 5,
        num_hops: Optional[int] = None,
        MCTS_info_path: Optional[str] = None,
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
        save_dir: Optional[str] = None,
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
        self.save = self.save_dir is not None

    def update_num_hops(self, model, num_hops) -> int:
        """calculate the number of hops in the model"""
        if num_hops is not None:
            return num_hops

        k = 0
        for module in model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def read_from_MCTSInfo_list(
        self, MCTSInfo_list: List[Union[List[Dict], Dict]]
    ) -> List[Union[List[MCTSNode], MCTSNode]]:
        """
        read MCTSInfo_list from saved file
        """
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [
                MCTSNode(device=self.device, **node_info) for node_info in MCTSInfo_list
            ]
        elif isinstance(MCTSInfo_list[0][0], dict):
            ret_list = [
                list(
                    map(
                        lambda x: MCTSNode(device=self.device, **x),
                        single_label_MCTSInfo_list,
                    )
                )
                for single_label_MCTSInfo_list in MCTSInfo_list
            ]
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        """
        TODO: @Donald - I think that it should be renmaed
        write MCTSNode_list to saved file
        """
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

    def get_reward_func(self, value_func, node_idx=None) -> Callable[[...], float]:
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
    ) -> MCTS:
        if (
            self.model_config.task_level == ModelTaskLevel.graph
            and node_idx is not None
        ):
            logging.warning(
                "For Graph Classification, node_idx should not be provided to explain. node_idx will be ignored"
            )
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
        label: Optional[Tensor] = None,
        max_nodes: int = 5,
        node_idx: Optional[int] = None,
        saved_MCTSInfo_list: Optional[List[List]] = None,
        **kwargs,
    ) -> Tuple[List[Union[List[MCTSNode], MCTSNode]], Dict, List[int]]:
        with torch.no_grad():
            probs = model(x, edge_index, **kwargs).softmax(dim=-1)

        if saved_MCTSInfo_list is not None:
            # TODO : donald, should check why there are two cases
            results = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)

        if self.model_config.task_level == ModelTaskLevel.graph:
            if saved_MCTSInfo_list is None:
                value_func = GnnNetsGC2valueFunc(model, target_class=label, **kwargs)
                payoff_func = self.get_reward_func(value_func)
                self.mcts_state_map: MCTS = self.get_mcts_class(
                    x, edge_index, score_func=payoff_func
                )
                results = self.mcts_state_map.mcts(verbose=self.verbose)
            value_func = GnnNetsGC2valueFunc(model, target_class=label)
        elif self.model_config.task_level == ModelTaskLevel.node:
            if node_idx is None:
                raise ValueError("For Node task, node_idx must be provided to explain")
            label = label[node_idx]

            self.mcts_state_map: MCTS = self.get_mcts_class(
                x, edge_index, node_idx=node_idx
            )
            self.new_node_idx = self.mcts_state_map.new_node_idx

            value_func = GnnNetsNC2valueFunc(
                model,
                node_idx=self.mcts_state_map.new_node_idx,
                target_class=label,
            )

            if saved_MCTSInfo_list is None:
                payoff_func = self.get_reward_func(
                    value_func, node_idx=self.mcts_state_map.new_node_idx
                )
                self.mcts_state_map.score_func = payoff_func
                results = self.mcts_state_map.mcts(verbose=self.verbose)
        else:
            raise ValueError(
                f"Task level '{self.model_config.task_level.value}' not supported"
            )

        tree_node_x: MCTSNode = find_closest_node_result(results, max_nodes=max_nodes)

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
        if self.model_config.task_level == ModelTaskLevel.node:
            assert (
                index is not None
            ), "For Node Classification, index (node_idx) must be provided"

        model.eval()

        self.num_hops = self.update_num_hops(model, self.num_hops)

        if self.save:
            file_path = join(self.save_dir, f"{self.filename}.pt")
            # TODO: @donald
            # in my understanding, there is no additional process when loading the saved results
            if isfile(file_path):
                logging.info(f"Loading saved results from {file_path}")
                saved_results = torch.load(file_path)
        else:
            saved_results = None

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

        if self.save:
            logging.info(f"Saving results to {file_path}")
            torch.save(results, file_path)

        # create node_mask from masked_node_list
        node_mask = torch.zeros(size=(x.size()[0],)).float()
        node_mask[masked_node_list] = 1.0
        node_mask = node_mask.unsqueeze(1)

        # create edge_mask from masked_node_list
        subgraph_edge_index, subgraph_x, edge_mask = subgraph(
            masked_node_list, edge_index, relabel_nodes=False, return_edge_mask=True
        )

        explanation = Explanation(
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask.unsqueeze(1).float(),
            results=results,
            related_pred=related_pred,
            masked_node_list=masked_node_list,
            explained_edge_list=subgraph_edge_index,
        )

        return explanation

    def supports(self) -> bool:
        """SubgraphXExplainer only supports Node and Graph Classification"""
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.node, ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False

        return True
