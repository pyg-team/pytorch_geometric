import os.path as osp

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from torch_geometric.contrib.nn import PanRepHetero
from torch_geometric.datasets import IMDB

# Some Notes to use this model:
# To compute NMD loss, we first need to run our defined function
# convert_to_mtx () and pass in a heteroGraph instance to get
# the .mtx representation of the heterogeneous graph. Then we
# need to run an external program written in c++
# (https://github.com/nkahmed/PGD) to acquire a .micro/.txt file
# representing the motif features for each node or edge. Finally,
# run our defined function txt_to_motif() and upload the .txt file
# to get motif features for each node as a tensor. For convenience,
# we put a motif_feature.txt file in the data folder, which is the motif
# features for the IMDB dataset.
# To compute MRW loss, we need to import the dgl library to perform metapath
# random walk.

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
dataset = IMDB(path)
imdb_graph = dataset[0]

# Following are utility functions for the PanRep model


def pyg_to_dgl(graph):
    """Convert a graph from PyTorch Geometric (PyG) format to Deep Graph
    Library (DGL) format.

    This function takes a graph in PyG format and converts it into a DGL graph.
    It iterates over
    the relations in the PyG graph and constructs a dictionary that maps each
    relation to its edge
    indices, which is then used to create the DGL graph.

    Args:
        graph (PyG Graph): The graph in PyTorch Geometric format.

    Returns:
        DGLGraph: The graph in Deep Graph Library format.

    Note:
        This function assumes that the input PyG graph has an attribute
        'edge_items' which contains
        the relational data necessary for constructing a heterogeneous graph in
        DGL.
    """
    # Dictionary to store relation to edge index mapping
    rel_dic = {}

    # Iterate over each relation and its data in the PyG graph
    for rel, dic in graph.edge_items():
        # Extract edge indices for the current relation
        cur = dic['edge_index']
        # Update relation dictionary with source and destination node indices
        rel_dic[rel] = cur[0], cur[1]

    # Create a DGL heterogeneous graph using the relation dictionary
    dgl_graph = dgl.heterograph(rel_dic)
    return dgl_graph


def get_all_node_features(data, ptr):
    """Extracts and concatenates the features of all nodes from a graph data
    structure.

    This function iterates over different types of nodes in the provided graph
    data,
    stores the index of each node type for reference, and collects their
    features.
    The features of all nodes are then concatenated into a single tensor.

    Parameters:
    - data (GraphData): A graph data structure containing multiple types of
    nodes. This structure is expected to have a method 'node_items' that yields
    node types and their corresponding data.
    - ptr (str): The key or pointer used to access node features within each
    node's data.

    Returns:
    - Tuple[Dict, torch.Tensor]:
        - The first element is a dictionary mapping each node type to its
        starting index
          in the concatenated feature tensor.
        - The second element is a torch.Tensor containing the concatenated
        features
          of all node types.

    Example:
    ```
    graph_data = GraphData(...)
    node_indices, all_features = get_all_node_features(graph_data,
    'feature_key')
    ```
    """
    node_index = {}
    node_features = []
    idx = 0
    for ntype, nx in data.node_items():
        node_index[ntype] = idx
        idx += nx[ptr].shape[0]
        node_features.append(nx[ptr])
    return node_index, torch.cat(tuple(node_features), dim=0)


def get_node_type_dict(features, index):
    """Creates a dictionary mapping each node type to its corresponding
    features/embeddings.

    This function takes a tensor of concatenated node features and a dictionary
    mapping node types to their starting indices in this tensor. It returns a
    new dictionary where each node type is associated with its respective
    feature slice from the tensor.

    Parameters:
    - features (torch.Tensor): A tensor containing concatenated features
    embeddings of various node types.
    - index (Dict): A dictionary mapping node types to their starting indices
    in the 'features' tensor.

    Returns:
    - Dict: A dictionary where each key is a node type and each value is a
    tensor of features corresponding to that node type.

    Example:
    ```
    features, index = get_all_node_features(graph_data, 'feature_key')
    node_type_dict = get_node_type_dict(features, index)
    ```
    """
    indices = sorted(list(index.values()))
    indices.append(features.shape[0])
    dic = {}
    for i in range(len(indices) - 1):
        for ntype in index:
            if index[ntype] == indices[i]:
                dic[ntype] = features[indices[i]:indices[i + 1]]
                break
    return dic


def get_complete_edge_indices(data, node_index):
    """Generates a single tensor of edge indices for a heterogeneous graph.

    This function iterates through all edge types in a graph, adjusting their
    indices
    based on a provided node index mapping. It concatenates these adjusted
    indices
    into a single tensor, which represents the complete set of edges in the
    graph.

    Parameters:
    - data (GraphData): A graph data structure with multiple edge types. It is
    expected
                        to have an 'edge_items' method that yields tuples
                        (relation, data_dict),
                        where 'relation' is a tuple representing the edge type
                        and 'data_dict'
                        contains the edge indices under the key 'edge_index'.
    - node_index (Dict): A dictionary mapping node types to their starting
    indices in a
                         concatenated node feature tensor.

    Returns:
    - torch.Tensor: A tensor containing all adjusted edge indices for the
    graph. Each column
                    in this tensor represents an edge, with the first row being
                    the index of
                    the source node and the second row the index of the target
                    node.

    Example:
    ```
    graph_data = GraphData(...)
    node_indices, _ = get_all_node_features(graph_data, 'feature_key')
    complete_edge_indices = get_complete_edge_indices(graph_data, node_indices)
    ```
    """
    edge_index = None
    for rel, dic in data.edge_items():
        cur = torch.clone(dic["edge_index"])
        cur[0] += node_index[rel[0]]
        cur[1] += node_index[rel[2]]
        if edge_index is None:
            edge_index = cur
        else:
            edge_index = torch.cat((edge_index, cur), dim=1)
    return edge_index


def get_edge_type(graph):
    """Extract and enumerate edge types from a graph.

    This function iterates over all the edge types in a heterogeneous graph and
    assigns a unique integer to each type. It then creates a tensor for each
    edge type filled with its corresponding integer identifier. Finally, it
    concatenates these tensors to form a single tensor representing all edge
    types in the graph.

    Args:
        graph: The heterogeneous graph from which to extract edge types.

    Returns:
        Tensor: A long tensor where each element represents the edge type of
        the corresponding edge in the graph. The edge types are enumerated as
        integers.

    Note:
        This function assumes that the input graph is a heterogeneous graph
        with a defined 'edge_types'
        attribute, and that each edge type in 'edge_types' has an associated
        'edge_index'.
    """
    # List to store tensors for each edge type
    all_edge_type = []

    # Iterate over all edge types in the graph
    for i in range(len(graph.edge_types)):
        edge_type = graph.edge_types[i]

        # Create a tensor filled with the current edge type's integer
        # identifier
        # The size of the tensor is equal to the number of edges of this type
        # in the graph
        edge_type_tensor = torch.full((graph[edge_type].edge_index.size(1), ),
                                      i, dtype=torch.long)

        # Append the tensor to the list
        all_edge_type.append(edge_type_tensor)

    return torch.cat(all_edge_type)


def generate_cluster(n_cluster, node_features):
    """Generate cluster labels for node features using KMeans clustering and
    convert to one-hot encoding.

    This function applies KMeans clustering to the given node features to group
    them into a specified
    number of clusters. It then converts these cluster labels into a one-hot
    encoded format.

    Args:
        n_cluster (int): The number of clusters to form.
        node_features (array-like or matrix): Node feature matrix where each
        row represents a node and each column represents a feature.

    Returns:
        Tensor: A one-hot encoded tensor of shape [num_nodes, n_cluster
        representing the cluster
                assignments of the nodes.

    Note:
        The function assumes that 'node_features' are provided in a format
        compatible with scikit-learn's
        KMeans (e.g., a NumPy array or a Pandas DataFrame).
    """
    # Compute KMeans clustering on the node features
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(node_features)

    # Retrieve the cluster labels for each node
    y_pred = kmeans.labels_

    # Convert the cluster labels to a one-hot encoding using PyTorch
    # The one-hot encoded matrix will have shape [num_nodes, n_cluster]
    y_pred_one_hot = torch.eye(n_cluster)[y_pred]

    return y_pred_one_hot


def generate_rwalks(g, metapaths, samples_per_node=10):
    """Generates random walks for nodes in a graph 'g' based on specified
    'metapaths'.

    For each node type in the graph, this function performs random walks as
    defined by the
    metapaths for that node type. It then organizes and returns the neighbors
    discovered through these random walks in a structured format.

    Parameters:
    g (DGL Graph): The graph on which random walks are to be performed.
    metapaths (dict): A dictionary where keys are node types and values are
    metapaths
                    (sequences of edge types) for random walks.
    samples_per_node (int, optional): Number of random walk samples to be
    generated per node.
                                    Defaults to 10.

    Returns:
    dict: A dictionary where keys are node types and values are dictionaries.
    Each value dictionary maps a neighboring node type to a tensor containing
    the IDs of neighboring nodes of that type.
    """
    rw_neighbors = {}
    for ntype in metapaths.keys():
        if ntype in g.ntypes:
            traces, types = dgl.sampling.random_walk(
                g,
                list(np.arange(g.number_of_nodes(ntype))) * samples_per_node,
                metapath=metapaths[ntype])
            # remove the same node id as the start of the walk!!
            traces = traces[:, 1:]
            types = types[1:]
            sampled_ntypes = list(types.numpy()) * samples_per_node
            rw_neighbors_ids = traces.reshape(
                (g.number_of_nodes(ntype), samples_per_node * traces.shape[1]))
            rw_neighbors[ntype] = (rw_neighbors_ids, sampled_ntypes)
            neighbors = rw_neighbors[ntype][0]
            neighbor_per_ntype = {}
            for id in range(len(rw_neighbors[ntype][1])):
                neighbor_type = g.ntypes[rw_neighbors[ntype][1][id]]
                if neighbor_type in neighbor_per_ntype:
                    neighbor_per_ntype[neighbor_type] = torch.cat(
                        (neighbor_per_ntype[neighbor_type],
                         neighbors[:, id].unsqueeze(0).transpose(1, 0)), dim=1)
                else:
                    neighbor_per_ntype[
                        neighbor_type] = neighbors[:,
                                                   id].unsqueeze(0).transpose(
                                                       1, 0)
            rw_neighbors[ntype] = neighbor_per_ntype

    return rw_neighbors


def txt_to_motif(file_path):
    # Read the content of the file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Number of motif types (D) is number of columns minus 2 (for src and
    # dst columns)
    num_motif_types = df.shape[1] - 2

    # Get the unique nodes
    nodes = np.unique(df[['% src', 'dst']].values)

    # Initialize a NxD matrix where N is the number of nodes
    N = len(nodes)
    D = num_motif_types
    node_features = np.zeros((N, D))

    # Map node IDs to matrix indices
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Fill the feature matrix
    for _, row in df.iterrows():
        src_idx = node_to_idx[row['% src']]
        dst_idx = node_to_idx[row['dst']]
        node_features[src_idx] += row[2:].values
        node_features[dst_idx] += row[2:].values

    # Now node_features contains the sum of motifs for each node as
    # described
    node_features_df = pd.DataFrame(node_features, index=nodes,
                                    columns=df.columns[2:])

    # Convert the DataFrame to a tensor
    node_features_tensor = torch.tensor(node_features_df.values,
                                        dtype=torch.float32)

    return node_features_tensor


# Function to convert HeteroData to MTX format
def convert_to_mtx(hetero_data):
    # Collect all edges in the graph
    edges = set()
    print(hetero_data)
    director_add = 4278
    actor_add = 4278 + 2081
    j = 0
    for edge_type in hetero_data.edge_types:
        j = j + 1
        edge_index = hetero_data[edge_type].edge_index
        print(edge_index.shape)
        if j == 1:
            src_add = 0
            dst_add = director_add
        elif j == 2:
            src_add = 0
            dst_add = actor_add
        elif j == 3:
            src_add = director_add
            dst_add = 0
        else:
            src_add = actor_add
            dst_add = 0
        for i in range(edge_index.shape[1]):
            # MTX format is 1-indexed
            src = edge_index[0][i].item() + 1 + src_add
            dst = edge_index[1][i].item() + 1 + dst_add
            real_src = max(src, dst)
            real_dst = min(src, dst)
            edges.add((real_src, real_dst))

    # Number of nodes is the maximum index since we're making it homogeneous
    num_nodes = 4278 + 2081 + 5257
    num_nodes = max(max(edge) for edge in edges)
    num_edges = len(edges)

    # Write to a .mtx file
    with open('graph.mtx', 'w') as f:
        # First line: Object type and size
        f.write('%%MatrixMarket matrix coordinate pattern symmetric\n')
        f.write(f'{num_nodes} {num_nodes} {num_edges}\n')
        # Write the edges to the file
        for edge in sorted(edges):
            f.write(f"{edge[0]} {edge[1]}\n")


num_relations = 4
feature_dim = 3066
n_cluster = 8
h_dim = 300
embed_dim = 3000
num_motifs = 8

index, node_features = get_all_node_features(imdb_graph, 'x')
true_clusters = generate_cluster(n_cluster, node_features)

metapath = {
    'movie': [('movie', 'to', 'director'), ('director', 'to', 'movie'),
              ('movie', 'to', 'actor'), ('actor', 'to', 'movie')],
    'director': [('director', 'to', 'movie'), ('movie', 'to', 'actor'),
                 ('actor', 'to', 'movie'), ('movie', 'to', 'director')],
    'actor': [('actor', 'to', 'movie'), ('movie', 'to', 'director'),
              ('director', 'to', 'movie'), ('movie', 'to', 'actor')]
}

imdb_dgl = pyg_to_dgl(imdb_graph)
rw_neighbors = generate_rwalks(imdb_dgl, metapath, samples_per_node=1)
dire = osp.dirname(osp.realpath(__file__))
motif_features = txt_to_motif(dire + '/motif_feature.txt')


def train(model, optimizer, num_epoch, g):
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        loss = model(g)
        loss.backward()
        optimizer.step()
        print('Epoch {}, loss {:.4f}'.format(epoch, loss.item()))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PanRepHetero(feature_dim, embed_dim, num_relations, n_cluster, h_dim,
                     num_motifs, True, 2, true_clusters, motif_features,
                     rw_neighbors, device).to(device)

num_epoch = 10
lr = 0.005
weight_decay = 0

optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)

train(model, optimizer, num_epoch, imdb_graph)
