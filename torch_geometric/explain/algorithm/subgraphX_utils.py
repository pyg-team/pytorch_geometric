"""SubgraphX Explainer Utils
The functions in the file have been ported over from,
- https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/shapley.py
- https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/subgraphx.py
"""
import copy
from functools import partial
from itertools import combinations
from typing import Callable, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import comb

from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph


def find_closest_node_result(results, max_nodes: int):
    """Return the highest reward tree_node
    with its subgraph is smaller than max_nodes

    Args:
        results (list): list of TreeNode
        max_nodes (int): maximum number of nodes in the subgraph

    Returns:
        result_node (TreeNode): the highest reward TreeNode
    """
    return sorted(
        filter(lambda x: len(x.coalition) <= max_nodes, results),
        key=lambda x: x.P,
        reverse=True,
    )[0]


def connected_components(
    edge_index: torch.Tensor, nodes: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """Generate connected components from edge_index by using k_hop_subgraph"""

    if nodes is None:
        nodes = edge_index.unique().tolist()

    seen = set()
    unseen = set(nodes)
    while unseen:
        node = unseen.pop()
        if node not in edge_index:
            seen.add(node)
            yield torch.tensor([node])
            continue
        component = k_hop_subgraph([node], len(nodes), edge_index)[0]
        seen.update(component.tolist())
        unseen.difference_update(component.tolist())
        yield component


class MarginalSubgraphDataset(Dataset):
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        super(MarginalSubgraphDataset, self).__init__()
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = (
            torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        )
        self.include_mask = (
            torch.tensor(include_mask).type(torch.float32).to(self.device)
        )
        self.subgraph_build_func = subgraph_build_func

    def len(self):
        return self.exclude_mask.shape[0]

    def get(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(
            self.X, self.edge_index, self.exclude_mask[idx]
        )
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(
            self.X, self.edge_index, self.include_mask[idx]
        )
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


def marginal_contribution(
    data: Data,
    exclude_mask: np.array,
    include_mask: np.array,
    value_func,
    subgraph_build_func,
):
    """Calculate the marginal value for each pair.
    Here exclude_mask and include_mask are node mask.
    """
    marginal_subgraph_dataset = MarginalSubgraphDataset(
        data, exclude_mask, include_mask, subgraph_build_func
    )
    dataloader = DataLoader(
        marginal_subgraph_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    marginal_contribution_list = []

    for exclude_data, include_data in dataloader:
        exclude_values = value_func(exclude_data)
        include_values = value_func(include_data)
        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def l_shapley(
    coalition: list,
    data: Data,
    local_radius: int,
    value_func: str,
    subgraph_building_method="zero_filling",
):
    """shapley value where players are local neighbor nodes"""
    graph = data
    num_nodes = graph.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = list(
        set(
            coalition
            + k_hop_subgraph(coalition, local_radius - 1, graph.edge_index)[0].tolist()
        )
    )
    set_exclude_masks = []
    set_include_masks = []
    nodes_around = list(set(local_region) - set(coalition))
    num_nodes_around = len(nodes_around)

    for subset_len in range(0, num_nodes_around + 1):
        node_exclude_subsets = combinations(nodes_around, subset_len)
        for node_exclude_subset in node_exclude_subsets:
            set_exclude_mask = np.ones(num_nodes)
            set_exclude_mask[local_region] = 0.0

            if node_exclude_subset:
                set_exclude_mask[list(node_exclude_subset)] = 1.0
            set_include_mask = set_exclude_mask.copy()
            set_include_mask[coalition] = 1.0
            set_exclude_masks.append(set_exclude_mask)
            set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    num_players = num_nodes_around + 1
    num_player_in_set = (
        num_players - 1 + len(coalition) - (1 - exclude_mask).sum(axis=1)
    )
    p = num_players
    S = num_player_in_set
    coeffs = torch.tensor(1.0 / comb(p, S) / (p - S + 1e-6))

    marginal_contributions = marginal_contribution(
        data, exclude_mask, include_mask, value_func, subgraph_build_func
    )

    l_shapley_value = (marginal_contributions.squeeze().cpu() * coeffs).sum().item()
    return l_shapley_value


def mc_shapley(
    coalition: list,
    data: Data,
    value_func: str,
    subgraph_building_method="zero_filling",
    sample_num=1000,
) -> float:
    """monte carlo sampling approximation of the shapley value"""
    subset_build_func = get_graph_build_func(subgraph_building_method)

    num_nodes = data.num_nodes
    node_indices = np.arange(num_nodes)
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []

    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in node_indices if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.zeros(num_nodes)
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(
        data, exclude_mask, include_mask, value_func, subset_build_func
    )
    mc_shapley_value = marginal_contributions.mean().item()

    return mc_shapley_value


def mc_l_shapley(
    coalition: list,
    data: Data,
    local_radius: int,
    value_func: str,
    subgraph_building_method="zero_filling",
    sample_num=1000,
) -> float:
    """monte carlo sampling approximation of the l_shapley value"""

    graph = data
    num_nodes = graph.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    local_region = list(
        set(
            coalition
            + k_hop_subgraph(coalition, local_radius - 1, graph.edge_index)[0].tolist()
        )
    )
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []

    for _ in range(sample_num):
        subset_nodes_from = list(set(local_region) - set(coalition))
        random_nodes_permutation = np.random.permutation(
            subset_nodes_from + [coalition_placeholder]
        )
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(
        data, exclude_mask, include_mask, value_func, subgraph_build_func
    )

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


def get_graph_build_func(
    build_method: str,
) -> Callable[[...], Tuple[torch.Tensor, torch.Tensor]]:
    """Return the subgraph building function"""
    if build_method.lower() == "zero_filling":
        return graph_build_zero_filling
    elif build_method.lower() == "split":
        return graph_build_split
    else:
        raise NotImplementedError


def graph_build_zero_filling(
    X, edge_index, node_mask: np.array
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subgraph building through masking the
    unselected nodes with zero features
    """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(
    X, edge_index, node_mask: np.array
) -> Tuple[torch.Tensor, torch.Tensor]:
    """subgraph building through spliting the selected nodes
    from the original graph
    """
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index


def gnn_score(
    coalition: list,
    data: Data,
    value_func: Callable[[Data], torch.Tensor],
    subgraph_building_method: str = "zero_filling",
) -> float:
    """the value of subgraph with selected nodes"""
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(data.num_nodes).type(torch.float32).to(data.x.device)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = value_func(mask_data)
    return score.item()


def reward_func(
    reward_method,
    value_func,
    node_idx=None,
    local_radius=4,
    sample_num=100,
    subgraph_building_method="zero_filling",
) -> Callable[[...], float]:
    if reward_method.lower() == "gnn_score":
        return partial(
            gnn_score,
            value_func=value_func,
            subgraph_building_method=subgraph_building_method,
        )

    elif reward_method.lower() == "mc_shapley":
        return partial(
            mc_shapley,
            value_func=value_func,
            subgraph_building_method=subgraph_building_method,
            sample_num=sample_num,
        )

    elif reward_method.lower() == "l_shapley":
        return partial(
            l_shapley,
            local_radius=local_radius,
            value_func=value_func,
            subgraph_building_method=subgraph_building_method,
        )

    elif reward_method.lower() == "mc_l_shapley":
        return partial(
            mc_l_shapley,
            local_radius=local_radius,
            value_func=value_func,
            subgraph_building_method=subgraph_building_method,
            sample_num=sample_num,
        )

    elif reward_method.lower() == "nc_mc_l_shapley":
        assert node_idx is not None, " Wrong node idx input "
        return partial(
            NC_mc_l_shapley,
            node_idx=node_idx,
            local_radius=local_radius,
            value_func=value_func,
            subgraph_building_method=subgraph_building_method,
            sample_num=sample_num,
        )
    else:
        raise NotImplementedError


def NC_mc_l_shapley(
    coalition: list,
    data: Data,
    local_radius: int,
    value_func: str,
    node_idx: int = -1,
    subgraph_building_method="zero_filling",
    sample_num=1000,
) -> float:
    """monte carlo approximation of l_shapley where the
    target node is kept in both subgraph"""
    graph = data
    num_nodes = graph.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    local_region = list(
        set(
            coalition
            + k_hop_subgraph(coalition, local_radius - 1, graph.edge_index)[0].tolist()
        )
    )
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        if node_idx != -1:
            set_exclude_mask[node_idx] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0  # include the node_idx

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(
        data, exclude_mask, include_mask, value_func, subgraph_build_func
    )

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value


def sparsity(coalition: list, data: Data, subgraph_building_method="zero_filling"):
    """calculate the sparsity of the subgraph"""
    if subgraph_building_method == "zero_filling":
        return 1.0 - len(coalition) / data.num_nodes

    elif subgraph_building_method == "split":
        row, col = data.edge_index
        node_mask = torch.zeros(data.x.shape[0])
        node_mask[coalition] = 1.0
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        return 1.0 - (edge_mask.sum() / edge_mask.shape[0]).item()


def GnnNetsGC2valueFunc(gnnNets, target_class):
    """Value Function for Graph Classification (GC) Task"""

    def value_func(batch, **kwargs):
        with torch.no_grad():
            logits = gnnNets(batch.x, batch.edge_index, **kwargs)
            probs = torch.softmax(logits, dim=-1)
            score = probs[:, target_class]
        return score

    return value_func


def GnnNetsNC2valueFunc(
    gnnNets_NC, node_idx, target_class
) -> Callable[[Data], torch.Tensor]:
    """Value Function for Node Classification (NC) Task"""

    def value_func(data: Data) -> torch.Tensor:
        with torch.no_grad():
            logits = gnnNets_NC(data.x, data.edge_index)
            probs = F.softmax(logits, dim=-1)
            batch_size = data.batch.max() + 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs[:, node_idx, target_class]
            return score

    return value_func
