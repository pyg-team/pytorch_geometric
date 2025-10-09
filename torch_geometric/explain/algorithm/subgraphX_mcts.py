import math
from functools import partial
from typing import Callable, List, Optional
import numpy as np

import torch

from torch_geometric.data import Data
from torch_geometric.utils import (
    k_hop_subgraph,
    remove_self_loops,
    to_undirected,
    degree,
    subgraph,
)

from .subgraphX_utils import connected_components


def compute_scores(score_func: Callable, children: List) -> List[float]:
    """
    compute the scores for each child node given the score function
    Args:
        score_func (:obj:`Callable`): The reward function for tree node
        children (:obj:`List`): The children nodes of the current node

    Returns:
        List[float]: the scores for each child node
    """
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
        data: Optional[Data] = None,
        coalition: list = [],
        c_puct: float = 10.0,
        W: float = 0,
        N: int = 0,
        P: float = 0,
        device: str = "cpu",
    ):
        """
        Args:
            coalition (:obj:`Optional[list]`): The coalition of nodes
                in the current subgraph.
            data (:obj:`Optional[Data]`, optional): The data of the current
                subgraph. Defaults to None.
            c_puct (:obj:`float`, optional): Hyper-parameter to encourage
                exploration while searching. Defaults to 10.0.
            W (:obj:`float`, optional): sum of node value. Defaults to 0.
            N (:obj:`int`, optional): times of arrival. Defaults to 0.
            P (:obj:`float`, optional): property score (reward). Defaults to 0.
            device (:obj:`str`, optional): The device to load the node
                information to. Defaults to "cpu".
        """
        self.data = data
        self.coalition = coalition
        self.device = device
        self.c_puct = c_puct
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        return {
            "data": self.data.to("cpu"),
            "coalition": self.coalition,
            "W": self.W,
            "N": self.N,
            "P": self.P,
        }


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method.

    Args:
        X (:obj:`torch.Tensor`): Input node features
        edge_index (:obj:`torch.Tensor`): The edge indices.
        num_hops (:obj:`int`): Number of hops :math:`k`.
        n_rollout (:obj:`int`): Number of sequences to build the
            monte carlo tree.
        score_func (:obj:`Callable`): The reward function for
            tree node, such as mc_shapely and mc_l_shapely.
        min_atoms (:obj:`int`): Number of atoms for the subgraph
            in the monte carlo tree leaf node.
        c_puct (:obj:`float`): Hyper-parameter to encourage
            exploration while searching.
        expand_atoms (:obj:`int`): Number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node
            from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract
            the neighborhood.
    """

    def __init__(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        num_hops: int,
        score_func: Callable,
        n_rollout: int = 10,
        min_atoms: int = 3,
        c_puct: float = 10.0,
        expand_atoms: int = 14,
        high2low: bool = False,
        node_idx: Optional[int] = None,
        device="cpu",
    ):
        self.device = device
        self.num_hops = num_hops
        self._score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        self.graph = Data(
            x=X, edge_index=to_undirected(remove_self_loops(edge_index)[0])
        )
        self.new_node_idx = node_idx
        self.subset = None

        # extract the sub-graph and change the node indices.
        if node_idx is not None:
            if isinstance(node_idx, torch.Tensor):
                node_idx = node_idx.item()
            x, edge_index, subset, mapping = self.__subgraph__(
                node_idx, X, edge_index, self.num_hops
            )
            self.graph = Data(
                x=x,
                edge_index=to_undirected(
                    remove_self_loops(
                        edge_index,
                    )[0]
                ),
            )
            self.new_node_idx = mapping.item()
            self.subset = subset.tolist()

        self.num_nodes = self.graph.num_nodes
        self.root_coalition = list(range(self.num_nodes))
        self.MCTSNodeClass = partial(
            MCTSNode,
            data=self.graph,
            c_puct=self.c_puct,
            device=self.device,
        )
        self.root = self.MCTSNodeClass(coalition=self.root_coalition)
        self.state_map = {str(self.root.coalition): self.root}

    @property
    def score_func(self):
        return self._score_func

    @score_func.setter
    def score_func(self, score_func):
        self._score_func = score_func

    @staticmethod
    def __subgraph__(
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_hops: int,
    ):
        num_nodes = x.size(0)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx,
            num_hops,
            edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        x = x[subset]
        return x, edge_index, subset, mapping

    def mcts_rollout(self, tree_node: MCTSNode) -> float:
        """ """
        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        if len(tree_node.children) == 0:
            subgraph_edge_index, _ = subgraph(
                cur_graph_coalition,
                self.graph.edge_index,
                relabel_nodes=False,
            )
            node_degree = degree(
                subgraph_edge_index[0], dtype=torch.long
            ).tolist()
            all_nodes = list(
                filter(
                    lambda x: node_degree[x] > 0,
                    sorted(
                        list(range(len(node_degree))),
                        key=lambda x: node_degree[x],
                        reverse=self.high2low,
                    ),
                )
            )

            if self.new_node_idx is not None:
                expand_nodes = [
                    node for node in all_nodes if node != self.new_node_idx
                ]
            else:
                expand_nodes = all_nodes

            for curr_node in np.random.permutation(expand_nodes)[
                : self.expand_atoms
            ]:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and
                # only keep the largest one
                subgraph_coalition = [
                    node for node in all_nodes if node != curr_node
                ]
                subgraphs = list(
                    connected_components(
                        subgraph(
                            subgraph_coalition,
                            self.graph.edge_index,
                            relabel_nodes=False,
                        )[0],
                        subgraph_coalition,
                    )
                )
                if self.new_node_idx is not None:
                    for sub in subgraphs:
                        if self.new_node_idx in sub:
                            main_sub = sub
                else:
                    main_sub = subgraphs[0]
                    for sub in subgraphs:
                        if len(sub) > len(main_sub):
                            main_sub = sub
                new_graph_coalition = sorted(main_sub.tolist())

                # check the state map and merge the same sub-graph
                find_same = False
                for old_graph_node in self.state_map.values():
                    if sorted(old_graph_node.coalition) == sorted(
                        new_graph_coalition
                    ):
                        new_node = old_graph_node
                        find_same = True
                        break

                if not find_same:
                    new_node = self.MCTSNodeClass(
                        coalition=new_graph_coalition
                    )
                    self.state_map[str(new_graph_coalition)] = new_node

                find_same_child = False
                for cur_child in tree_node.children:
                    if sorted(cur_child.coalition) == sorted(
                        new_graph_coalition
                    ):
                        find_same_child = True
                        break

                if not find_same_child:
                    tree_node.children.append(new_node)

            scores = compute_scores(self.score_func, tree_node.children)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(
            tree_node.children, key=lambda x: x.Q() + x.U(sum_count)
        )
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True) -> List[MCTSNode]:
        if verbose:
            print(f"The nodes in graph is {self.graph}")
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
