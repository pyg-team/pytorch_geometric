import json
from collections import deque

import networkx as nx
import torch

from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected

# Load entity names
with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}


def build_pyg_graph(graph_data: list, entities: list = None,
                    encrypt: bool = False) -> Data:
    """Construct a PyG Data object from a list of (head, relation, tail) triplets.
    Uses PyG functionalities such as to_undirected and coalesce to format the graph.
    """
    edges = []
    for h, r, t in graph_data:
        if encrypt:
            if (h in names_entities) and (names_entities[h] in entities):
                h = names_entities[h]
            if (t in names_entities) and (names_entities[t] in entities):
                t = names_entities[t]
        edges.append((h, r.strip(), t))

    #build node list and mapping
    node_set = set()
    for h, r, t in edges:
        node_set.add(h)
        node_set.add(t)
    node_list = sorted(node_set)
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    edge_index_list = []
    rel_list = []

    for h, r, t in edges:  #store edges in a dict form to handle after coalescing
        edge_index_list.append([node_to_idx[h], node_to_idx[t]])
        rel_list.append(r)

    edge_index = torch.tensor(edge_index_list,
                              dtype=torch.long).t().contiguous()

    edge_index = to_undirected(
        edge_index)  #coalesce step: remove duplicates, sort edes
    edge_index, _ = coalesce(edge_index, None, len(node_list), len(node_list))

    # After coalescing, we have to remap relations to edges. We'll build a small lookup for (u,v) -> relation.
    # Since the graph is undirected, we store both (u,v) and (v,u).
    rel_map = {}
    for (h, r, t) in edges:
        u = node_to_idx[h]
        v = node_to_idx[t]
        rel_map[(u, v)] = r
        rel_map[(v, u)] = r

    final_rels = []
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        final_rels.append(rel_map[(u, v)])

    data = Data()
    data.num_nodes = len(node_list)
    data.edge_index = edge_index
    data.node_list = node_list  # ref back to original names
    data.node_to_idx = node_to_idx
    data.relations = final_rels  # list of relations, parallel to edges in edge_index

    return data


def pyg_data_to_networkx(data: Data) -> nx.Graph:
    """Converts a PyG Data object to a NetworkX Graph for easier BFS, shortest paths, etc.
    """
    G = nx.Graph()
    G.add_nodes_from(data.node_list)

    for i in range(data.edge_index.size(1)):
        u = data.node_list[data.edge_index[0, i].item()]
        v = data.node_list[data.edge_index[1, i].item()]
        rel = data.relations[i]
        G.add_edge(u, v, relation=rel)

    return G


def bfs_with_rule(data: Data, start_node, target_rule: list,
                  max_p: int = 10) -> list:
    """Perform breadth-first search (BFS) to find paths in the graph that match a given
    sequence of relations (target_rule). Uses the PyG Data object to store graph structure.
    utilizes NetworkX bfS and relation matching logic.

    """
    G = pyg_data_to_networkx(data)

    if isinstance(start_node, str):
        if start_node not in data.node_to_idx:
            return []
    if isinstance(start_node, int):
        start_node = data.node_list[start_node]

    result_paths = []
    queue = deque([(start_node, [])])

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


def get_truth_paths(q_entities: list, a_entities: list, data: Data) -> list:
    """Retrieves all shortest paths between sets of question entities (q_entities) and answer entities (a_entities).
    Uses the PyG data representation; the actual shortest path computation is done via NetworkX.
    Relations and nodes are mapped back from indices stored in the PyG Data object.
    """
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
                    path_with_rels = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        rel = G[u][v]['relation']
                        path_with_rels.append((u, rel, v))
                    paths.append(path_with_rels)
            except:
                # If no path exists, continue
                pass

    return paths


def verbalize_paths(paths):
    verbalized = []
    for path in paths:
        verbalized.append(" → ".join(f"{edge[0]} → {edge[1]} → {edge[2]}"
                                     for edge in path))
    return "\n".join(verbalized)
