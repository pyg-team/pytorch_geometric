import networkx as nx
import torch
from torch_geometric.utils import negative_sampling


def remove_node_feature(graph):
    '''set node feature to be constant'''
    graph.node_feature = torch.ones(graph.num_nodes, 1)


def ego_nets(graph, radius=2):
    '''get networks for mini batch node/graph prediction tasks'''
    # color center
    egos = []
    n = graph.num_nodes
    # A proper deepsnap.G should have nodes indexed from 0 to n-1
    for i in range(n):
        if radius > 4:
            egos.append(graph.G)
        else:
            egos.append(nx.ego_graph(graph.G, i, radius=radius))
    # relabel egos: keep center node ID, relabel other node IDs
    G = graph.G.__class__()
    id_bias = n
    for i in range(len(egos)):
        G.add_node(i, **egos[i].nodes(data=True)[i])
    for i in range(len(egos)):
        keys = list(egos[i].nodes)
        keys.remove(i)
        id_cur = egos[i].number_of_nodes() - 1
        vals = range(id_bias, id_bias + id_cur)
        id_bias += id_cur
        mapping = dict(zip(keys, vals))
        ego = nx.relabel_nodes(egos[i], mapping, copy=True)
        G.add_nodes_from(ego.nodes(data=True))
        G.add_edges_from(ego.edges(data=True))
    graph.G = G
    graph.node_id_index = torch.arange(len(egos))


def edge_nets(graph):
    '''get networks for mini batch edge prediction tasks'''
    # color center
    n = graph.num_nodes
    # A proper deepsnap.G should have nodes indexed from 0 to n-1
    # relabel egos: keep center node ID, relabel other node IDs
    G_raw = graph.G
    G = graph.G.__class__()
    for i in range(n):
        keys = list(G_raw.nodes)
        vals = range(i * n, (i + 1) * n)
        mapping = dict(zip(keys, vals))
        G_raw = nx.relabel_nodes(G_raw, mapping, copy=True)
        G.add_nodes_from(G_raw.nodes(data=True))
        G.add_edges_from(G_raw.edges(data=True))
    graph.G = G
    graph.node_id_index = torch.arange(0, n * n, n + 1)

    # change link_pred to conditional node classification task
    graph.node_label = graph.edge_label
    bias = graph.edge_label_index[0] * n
    graph.node_label_index = graph.edge_label_index[1] + bias

    graph.edge_label = None
    graph.edge_label_index = None


def path_len(graph):
    '''get networks for mini batch shortest path prediction tasks'''
    n = graph.num_nodes
    # shortest path label
    num_label = 1000
    edge_label_index = torch.randint(n, size=(2, num_label),
                                     device=graph.edge_index.device)
    path_dict = dict(nx.all_pairs_shortest_path_length(graph.G))
    edge_label = []
    index_keep = []
    for i in range(num_label):
        start, end = edge_label_index[0, i].item(), edge_label_index[
            1, i].item()
        try:
            dist = path_dict[start][end]
        except:
            continue
        edge_label.append(min(dist, 4))
        index_keep.append(i)

    edge_label = torch.tensor(edge_label, device=edge_label_index.device)
    graph.edge_label_index, graph.edge_label = edge_label_index[:,
                                               index_keep], edge_label


def get_link_label(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float,
                              device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def neg_sampling_transform(data):
    train_neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    data.train_edge_index = torch.cat(
        [data.train_pos_edge_index, train_neg_edge_index], dim=-1)
    data.train_edge_label = get_link_label(data.train_pos_edge_index,
                                           train_neg_edge_index)

    return data
