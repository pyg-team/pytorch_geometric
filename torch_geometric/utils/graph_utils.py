import json
from collections import deque
from typing import List, Optional, Tuple, Union

import networkx as nx
import torch

from torch_geometric.data import Data
from torch_geometric.utils import coalesce

# Load entity names
with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}


def build_pyg_graph(graph_data: List[Tuple[str, str, str]],
                    entities: Optional[List[str]] = None,
                    encrypt: bool = False) -> Data:
    """Construct a PyG Data object from a list of (head, relation,
    tail) triplets.
    """
    edges = []
    for h, r, t in graph_data:
        if encrypt:
            if entities is not None and h in names_entities and \
               names_entities[h] in entities:
                h = names_entities[h]
            if entities is not None and t in names_entities and \
               names_entities[t] in entities:
                t = names_entities[t]
        edges.append((h, r.strip(), t))

    # Build node list and mapping
    node_set = {node for edge in edges for node in (edge[0], edge[2])}
    node_list = sorted(node_set)
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    edge_index_list = []
    rel_list = []

    for h, r, t in edges:
        edge_index_list.append([node_to_idx[h], node_to_idx[t]])
        rel_list.append(r)

    edge_index = torch.tensor(edge_index_list,
                              dtype=torch.long).t().contiguous()

    # Coalesce step: remove duplicates, sort edges
    edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32)
    edge_index, edge_weights = coalesce(edge_index, edge_weights,
                                        num_nodes=len(node_list))

    # Remap relations to edges
    rel_map = {}
    for h, r, t in edges:
        u, v = node_to_idx[h], node_to_idx[t]
        rel_map[(u, v)] = r
        rel_map[(v, u)] = r

    final_rels = [
        rel_map[(edge_index[0, i].item(), edge_index[1, i].item())]
        for i in range(edge_index.size(1))
    ]

    data = Data(num_nodes=len(node_list), edge_index=edge_index,
                edge_weights=edge_weights)
    data.node_list = node_list  # Reference to original names
    data.node_to_idx = node_to_idx
    data.relations = final_rels  # List of relations, parallel to edge_index

    return data


def pyg_data_to_networkx(data: Data) -> nx.Graph:
    """Converts a PyG Data object to a NetworkX Graph."""
    G = nx.Graph()
    if data.node_list is None:
        raise ValueError("Data object does not contain node_list attribute.")
    G.add_nodes_from(data.node_list)

    if data.edge_index is not None:
        for i in range(data.edge_index.size(1)):
            u = data.node_list[data.edge_index[0, i].item()]
            v = data.node_list[data.edge_index[1, i].item()]
            rel = data.relations[i]
            G.add_edge(u, v, relation=rel)

    return G


def bfs_with_rule(data: Data, start_node: Union[str,
                                                int], target_rule: List[str],
                  max_p: int = 10) -> List[List[Tuple[str, str, str]]]:
    """Perform BFS to find paths matching a sequence of relations."""
    G = pyg_data_to_networkx(data)

    if isinstance(start_node, str) and start_node not in data.node_to_idx:
        return []
    if isinstance(start_node, int):
        start_node = data.node_list[start_node]

    result_paths = []
    queue: deque[Tuple[str, List[Tuple[str, str,
                                       str]]]] = deque([(start_node, [])])

    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            continue

        if current_node in G:
            expected_rel = target_rule[len(current_path)]
            for neighbor in G.neighbors(current_node):
                rel = G[current_node][neighbor]['relation']
                if rel == expected_rel:
                    new_path = current_path + [(current_node, rel, neighbor)]
                    queue.append((neighbor, new_path))

    return result_paths


def get_truth_paths(q_entities: List[str], a_entities: List[str],
                    data: Data) -> List[List[Tuple[str, str, str]]]:
    """Retrieves all shortest paths bw question and ans entities."""
    G = pyg_data_to_networkx(data)
    paths = []

    for q in q_entities:
        if q not in G:
            continue
        for a in a_entities:
            if a not in G:
                continue
            try:
                for path in nx.all_shortest_paths(G, q, a):
                    path_with_rels = [
                        (path[i], G[path[i]][path[i + 1]]['relation'],
                         path[i + 1]) for i in range(len(path) - 1)
                    ]
                    paths.append(path_with_rels)
            except nx.NetworkXNoPath:
                pass  # If no path exists, continue

    return paths


def verbalize_paths(paths: List[List[Tuple[str, str, str]]]) -> str:
    """Converts paths into a readable format."""
    return "\n".join(" → ".join(f"{edge[0]} → {edge[1]} → {edge[2]}"
                                for edge in path) for path in paths)
