# Standard library imports
import math

# PyTorch imports for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric import for geometric deep learning
from torch_geometric.nn import RGCNConv


class PanRepHetero(torch.nn.Module):
    """This model is designed for heterogeneous graph representation learning.
    It combines an RGCN encoder with multiple decoders to address various tasks
    such as cluster recovery, information maximization, node motif decoding,
    and metapath random walk decoding.

    Args:
        feature_dim (int): Dimension of the input features.
        embed_dim (int): Dimension of the node embeddings.
        num_relations (int): Number of relation types in the graph.
        n_cluster (int): Number of clusters for the Cluster Recovery Decoder.
        hidden_dim (int): Dimension of the hidden layer for the Cluster
        Recovery Decoder.
        num_motifs (int): Number of motifs for the Node Motif Decoder.
        single_layer (bool): Flag to use a single layer in the Node Motif
        Decoder.
        negative_rate (int): Negative sampling rate for metapath random walk
        decoder.
        true_clusters (Tensor): Ground truth clusters for the Cluster
        Recovery Decoder.
        motif_features (Tensor): Motif features for the Node Motif Decoder.
        rw_neighbors (dict): Neighbors data for the Metapath Random Walk
        Decoder.
        device (torch.device, optional): Device on which to run the model.
    """
    def __init__(self, feature_dim, embed_dim, num_relations, n_cluster,
                 hidden_dim, num_motifs, single_layer, negative_rate,
                 true_clusters, motif_features, rw_neighbors, device):
        super(PanRepHetero, self).__init__()
        # Store parameters and data
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.n_cluster = n_cluster
        self.hidden_dim = hidden_dim
        self.num_motifs = num_motifs
        self.single_layer = single_layer
        self.negative_rate = negative_rate
        self.true_clusters = true_clusters
        self.motif_features = motif_features
        self.rw_neighbors = rw_neighbors
        self.device = device

        # Initialize the encoder and decoders
        self.encoder = RGCNEncoder(feature_dim, embed_dim, num_relations,
                                   device=device)
        self.decoders = nn.ModuleDict({
            'crd':
            ClusterRecoverDecoderHomo(n_cluster, embed_dim, hidden_dim, device,
                                      single_layer),
            'imd':
            InformationMaximizationDecoder(embed_dim, device),
            'nmd':
            NodeMotifDecoder(embed_dim, num_motifs, device,
                             single_layer=single_layer),
            'mrw':
            MetapathRWalkDecoder(embed_dim, rw_neighbors, device,
                                 negative_rate)
        })

    def forward(self, g):
        """Forward pass of the PanRepHetero model.

        Args:
            g (Graph): The input graph.

        Returns:
            Tensor: The sum of losses from each decoder in the model.
        """
        # Extract and process node features and edge data
        index, node_features = get_all_node_features(g, 'x')
        pos_samples = node_features.to(self.device)
        neg_samples = generate_neg_samples(g, 'x')
        edge_index = get_complete_edge_indices(g, index).to(self.device)
        edge_type = get_edge_type(g).to(self.device)

        # Encode the positive and negative node samples
        pos_x = self.encoder(pos_samples, edge_index,
                             edge_type).to(self.device)

        # Compute losses for each decoder
        for decoder_name, decoder_model in self.decoders.items():
            if decoder_name == 'crd':
                crd_loss = decoder_model(self.true_clusters.to(self.device),
                                         pos_x)

            if decoder_name == 'imd':
                neg_x = self.encoder(neg_samples.to(self.device), edge_index,
                                     edge_type).to(self.device)
                pos_embed_dict = get_node_type_dict(pos_x, index)
                neg_embed_dict = get_node_type_dict(neg_x, index)
                imd_loss = decoder_model(pos_embed_dict, neg_embed_dict)

            if decoder_name == 'nmd':
                nmd_loss = decoder_model(self.motif_features.to(self.device),
                                         pos_x)

            if decoder_name == 'mrw':
                mrw_loss = decoder_model(g, 'x', pos_embed_dict)

        # Sum and return the total loss from all dec
        return crd_loss + imd_loss + nmd_loss + mrw_loss


class RGCNEncoder(torch.nn.Module):
    """This encoder is part of the PanRep model, which is designed for
    extracting universal node embeddings in heterogeneous graphs.

    Args:
    in_channels (int): Number of input features per node.
    out_channels (int): Number of output features per node.
    num_relations (int): Number of different types of relations in the graph.
    device (torch.device, optional): The device on which to run the module
    (defaults to 'cuda' if available, otherwise 'cpu').

    The encoder uses a Relational Graph Convolutional Network (RGCN) layer
    to process
    the graph data, which is effective in learning on graphs with multiple
    relation types.
    """
    def __init__(self, in_channels, out_channels, num_relations, device):
        super(RGCNEncoder, self).__init__()
        # Define an RGCN layer
        # The RGCNConv layer is crucial for handling different types of
        # relations in a graph.
        # It processes input node features (in_channels), and outputs
        # transformed node features (out_channels),
        # based on the number of relation types (num_relations) in the graph.
        self.conv = RGCNConv(in_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        """Forward pass of the RGCNEncoder.

        Args:
            x (Tensor): The input node features, shape [num_nodes, in_channels]
            edge_index (LongTensor): The edge indices, shape [2, num_edges].
            edge_type (Tensor): The type of each edge in the graph, shape
            [num_edges].

        Returns:
            Tensor: The transformed node features, shape [num_nodes,
            out_channels].

        The method applies the defined RGCN layer to the input graph data. The
        RGCN layer
        uses the node features, the edge connectivity (edge_index), and edge
        types (edge_type)
        to perform the convolution operation and returns the updated node
        features.
        """
        # Apply the RGCN layer
        # The convolution operation is applied to the node features, along with
        # the edge connectivity
        # and edge types, to produce the updated node features.
        x = self.conv(x, edge_index, edge_type)
        return x


class ClusterRecoverDecoderHomo(torch.nn.Module):
    """This decoder is part of the PanRep model and is designed
    for the task of cluster recovery.
    It aims to reconstruct cluster assignments from node embeddings.

    Args:
    n_cluster (int): The number of clusters.
    in_size (int): The size of the input features for each node.
    h_dim (int): The size of the hidden dimension
    (used when single_layer is False).
    device (torch.device, optional): The device on which to run the module
    (defaults to 'cuda' if available, otherwise 'cpu').
    activation (torch.nn.Module, optional): The activation function
    to use (default: torch.nn.ReLU()).
    single_layer (bool, optional): Whether to use a single linear layer
    or a multi-layer architecture for the decoder (default: True).

    The decoder uses a neural network architecture that can be either
    a single linear layer or a multi-layer perceptron, depending
    on the `single_layer` flag.
    """
    def __init__(self, n_cluster, in_size, h_dim, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'),
                 activation=torch.nn.ReLU(), single_layer=True):
        super(ClusterRecoverDecoderHomo, self).__init__()

        self.activation = activation
        self.h_dim = h_dim
        self.n_cluster = n_cluster
        self.device = device
        self.single_layer = single_layer

        layers = []
        if self.single_layer:
            layers.append(nn.Linear(in_size, n_cluster, bias=False))
        else:
            layers.append(nn.Linear(in_size, self.h_dim))
            layers.append(activation)
            layers.append(nn.Linear(self.h_dim, n_cluster, bias=False))

        # Assemble the layers into a sequential model and move to the specified
        # device
        self.weight = nn.Sequential(*layers).to(self.device)

    def loss_function(self, predicted_clusters, true_clusters):
        """Compute the loss for the cluster recovery.

        Args:
            predicted_clusters (Tensor): The predicted cluster assignments,
            shape [num_nodes, n_cluster].
            true_clusters (Tensor): The ground truth cluster assignments,
            shape [num_nodes, n_cluster].

        Returns:
            Tensor: The computed loss value.

        The function applies softmax to the predicted clusters and computes
        the cross-entropy loss
        against the true cluster assignments. The loss reflects how well
        the model performs in reconstructing the correct cluster assignments.
        """
        predicted_clusters = F.softmax(predicted_clusters, dim=1)
        return -torch.sum(true_clusters * torch.log(predicted_clusters + 1e-15)
                          ) / true_clusters.shape[0]

    def forward(self, true_clusters, node_embed):
        """Forward pass of the ClusterRecoverDecoderHomo.

        Args:
            true_clusters (Tensor): The ground truth cluster assignments, shape
            [num_nodes, n_cluster].
            node_embed (Tensor): The node embeddings, shape [num_nodes,
            in_size].

        Returns:
            Tensor: The computed loss for the reconstructed clusters.

        The method applies the decoder network to the node embeddings to
        reconstruct the cluster
        assignments and then calculates the loss using the true cluster
        assignments.
        """
        reconstructed = self.weight(node_embed)
        loss = self.loss_function(reconstructed, true_clusters)
        return loss


class InformationMaximizationDecoder(torch.nn.Module):
    """This decoder is part of the PanRep model, designed for maximizing
    information in node embeddings
    using a contrastive learning approach. It learns to distinguish between
    positive and negative
    node pairs.

    Args:
    n_hidden (int): The size of the hidden layer, which also corresponds to the
    size of node embeddings.
    device (torch.device, optional): The device on which to run the module
    (defaults to 'cuda' if available, otherwise 'cpu').

    The decoder uses a learnable weight matrix `w_him` to transform node
    embeddings for the
    contrastive learning task. It also employs a binary cross-entropy loss for
    training.
    """
    def __init__(self, n_hidden, device):

        super(InformationMaximizationDecoder, self).__init__()
        self.w_him = nn.Parameter(torch.Tensor(n_hidden, n_hidden)).to(device)
        self.loss = nn.BCEWithLogitsLoss()
        self.reset_parameters()

    def uniform(self, size, tensor):
        """Initialize a tensor uniformly.

        Args:
            size (int): The size parameter to determine the bounds of the
            uniform distribution.
            tensor (Tensor): The tensor to be initialized.

        This method initializes the tensor with values uniformly distributed
        between -1/sqrt(size) and 1/sqrt(size).
        """
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        """Reset the parameters of the decoder.

        This method initializes the learnable weight matrix `w_him` using the
        uniform method.
        """
        size = self.w_him.size(0)
        self.uniform(size, self.w_him)

    def forward(self, positives, negatives):
        """Forward pass of the InformationMaximizationDecoder.

        Args:
            positives (dict): A dictionary of positive node embeddings, keyed
            by node type.
            negatives (dict): A dictionary of negative node embeddings, keyed
            by node type.

        Returns:
            Tensor: The total loss computed from the positive and negative
            samples.

        The method computes the contrastive loss for each node type by applying
        a sigmoid function
        on the transformed node embeddings and calculating the binary
        cross-entropy loss with respect
        to the positive and negative labels.
        """
        l1 = 0
        l2 = 0
        for node_type in positives.keys():
            if positives[node_type].shape[0] > 0:
                pos = positives[node_type]
                neg = negatives[node_type]
                glo = pos.mean(dim=0)

                pos = torch.sigmoid(
                    torch.matmul(pos, torch.matmul(self.w_him, glo)))
                neg = torch.sigmoid(
                    torch.matmul(neg, torch.matmul(self.w_him, glo)))

                l1 += self.loss(pos, torch.ones_like(pos))
                l2 += self.loss(neg, torch.zeros_like(neg))
        return l1 + l2


class MetapathRWalkDecoder(torch.nn.Module):
    """This decoder is a part of the PanRep model, designed to handle decoding
    tasks based on relational walks
    using metapaths in heterogeneous graphs.

    Args:
    in_dim (int): The dimensionality of the input node embeddings.
    mrw (dict): A dictionary defining the metapath-based random walks for each
    node type.
    device (torch.device, optional): The device on which to run the module
    (defaults to 'cuda' if available, otherwise 'cpu').
    negative_rate (int, optional): The rate at which negative samples are
    generated compared
    to positive samples (default: 5).

    The decoder utilizes a parameter dictionary `w_relation` for each relation
    type, which aids in
    computing the relational scores between nodes in the graph.
    """
    def __init__(self, in_dim, mrw, device, negative_rate=5):

        super(MetapathRWalkDecoder, self).__init__()
        self.in_dim = in_dim
        self.negative_rate = negative_rate
        self.mrw = mrw
        self.device = device

        # Initialize weight parameters for each relation
        w_relation = {}
        self.w_relation = nn.ParameterDict().to(device)
        for node_type in mrw.keys():
            for neighbor_type in mrw[node_type].keys():
                relation = node_type + neighbor_type
                w_relation[relation] = nn.Parameter(torch.Tensor(in_dim, 1))
                nn.init.xavier_uniform_(w_relation[relation],
                                        gain=nn.init.calculate_gain('relu'))
        self.w_relation.update(w_relation)

    def cal_score(self, head_embed, tail_embed, relation):
        """Calculate the relational score between two nodes.

        Args:
            head_embed (Tensor): The embedding of the head node.
            tail_embed (Tensor): The embedding of the tail node.
            relation (str): The relation type between the head and tail nodes.

        Returns:
            Tensor: The computed relational score.

        The method computes a relational score based on the embeddings of the
        head and tail nodes and
        the specific relation type, using the relation-specific weight
        parameter.
        """
        r = self.w_relation[relation].squeeze().to(self.device)
        score = torch.sum(head_embed * r * tail_embed, dim=1)
        return score

    def forward(self, g, ptr, node_embed_dict):
        """Forward pass of the MetapathRWalkDecoder.

        Args:
            g (Graph): The input graph.
            ptr (str): A pointer to the feature of graph.
            node_embed_dict (dict): A dictionary containing node embeddings,
            keyed by node type.

        Returns:
          Tensor: The total loss computed for the metapath-based random walks.

        The method computes loss based on the binary cross-entropy for both
        positive and negative
        samples, using the relational scores derived from the metapath-based
        random walks.
        """
        loss = 0
        for node_type in self.mrw.keys():
            cur_node_type_neighbors = self.mrw[node_type]
            for neighbor_type in cur_node_type_neighbors.keys():
                cur_node_ids = torch.arange(
                    0,
                    g.get_node_store(node_type)[ptr].shape[0])
                neighbor_ids = cur_node_type_neighbors[neighbor_type][
                    cur_node_ids]
                all_neighbor_ids = torch.arange(
                    0,
                    g.get_node_store(neighbor_type)[ptr].shape[0])

                sampled_neighbors_ids = (torch.nonzero(
                    neighbor_ids[..., None] == all_neighbor_ids))

                if sampled_neighbors_ids.shape[0] > 0:
                    head_nodes = sampled_neighbors_ids[:, 0]
                    tail_nodes = sampled_neighbors_ids[:, 2]
                    head_embed = node_embed_dict[node_type][head_nodes].to(
                        self.device)
                    tail_embed = node_embed_dict[neighbor_type][tail_nodes].to(
                        self.device)
                    relation = node_type + neighbor_type
                    pos_score = self.cal_score(head_embed, tail_embed,
                                               relation)
                    loss += F.binary_cross_entropy_with_logits(
                        pos_score,
                        torch.ones(tail_embed.shape[0]).to(self.device))

                    neg_head_nodes = head_nodes.repeat(self.negative_rate)
                    neg_tail_nodes = torch.randint(
                        node_embed_dict[neighbor_type].shape[0],
                        (tail_embed.shape[0] * self.negative_rate, ))
                    neg_head_embed = node_embed_dict[node_type][
                        neg_head_nodes].to(self.device)
                    neg_tail_embed = node_embed_dict[neighbor_type][
                        neg_tail_nodes].to(self.device)
                    neg_score = self.cal_score(neg_head_embed, neg_tail_embed,
                                               relation)
                    loss += F.binary_cross_entropy_with_logits(
                        neg_score,
                        torch.zeros(neg_tail_embed.shape[0]).to(self.device))
        return loss


class NodeMotifDecoder(torch.nn.Module):
    """This decoder is a part of the PanRep model, intended for transforming
    node embeddings into motif features.
    It aims to reconstruct motif representations from the node embeddings.

    Args:
    in_dim (int): The dimensionality of the input node embeddings.
    motif_dim (int): The dimensionality of the motif features to be
    reconstructed.
    device (torch.device, optional): The device on which to run the module
    (defaults to 'cuda' if available, otherwise 'cpu').
    activation (torch.nn.Module, optional): The activation function to use
    (default: nn.ReLU()).
    single_layer (bool, optional): Whether to use a single linear layer or a
    multi-layer architecture for the decoder (default: False).

    The decoder uses a neural network architecture that can be either a single
    linear layer
    or a multi-layer perceptron, depending on the `single_layer` flag. It aims
    to minimize the
    mean squared error between the reconstructed and actual motif features.
    """
    def __init__(self, in_dim, motif_dim, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'), activation=nn.ReLU(),
                 single_layer=False):
        super(NodeMotifDecoder, self).__init__()
        layers = []
        self.device = device

        if single_layer:
            layers.append(nn.Linear(in_dim, motif_dim, bias=False))
        else:
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(activation)
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(activation)
            layers.append(nn.Linear(in_dim, motif_dim, bias=False))

        # Assemble the layers into a sequential model and move to the specified
        # device
        self.weight = nn.Sequential(*layers).to(device)

    def forward(self, motif_features, node_embed):
        """Forward pass of the NodeMotifDecoder.

        Args:
            motif_features (Tensor): The actual motif features of the nodes,
            shape [num_nodes, motif_dim].
            node_embed (Tensor): The node embeddings.

        Returns:
            Tensor: The computed loss for the reconstructed motif features.

        The method applies the decoder network to the node embeddings to
        reconstruct the motif features
        and then calculates the mean squared error loss using the actual motif
        features.
        """
        reconstructed = self.weight(node_embed).to(self.device)
        # Compute the mean squared error loss normalized by the number of nodes
        loss = F.mse_loss(reconstructed, motif_features) / node_embed.shape[0]
        return loss


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


def shuffle_node_attributes(nodes):
    """Shuffles node attributes among nodes of the same type.

    This function randomly permutes the rows of the input tensor. Each row in
    the tensor represents the attributes of a node, so this operation
    effectively shuffles the attributes among all the nodes.

    Parameters:
    nodes (Tensor): A tensor containing node attributes. Each row corresponds
    to a node.

    Returns:
    Tensor: A tensor with the same shape as `nodes`, but with attributes
    shuffled among nodes.
    """
    # Number of nodes
    num_nodes = nodes.size(0)

    # Generate a random permutation of indices
    shuffle_indices = torch.randperm(num_nodes)

    # Apply the permutation to shuffle node attributes
    shuffled_nodes = nodes[shuffle_indices]

    return shuffled_nodes


def generate_neg_samples(data, ptr):
    """Generates negative samples by shuffling node attributes
    within each node type.

    This function iterates through each node type in the graph
    data, shuffles the node attributes, and then concatenates
    these shuffled attributes for all node types into a single tensor.
    This is useful in graph learning tasks, particularly for generating
    negative samples for contrastive learning or other unsupervised
    learning tasks.

    Parameters:
    - data (GraphData): A graph data structure containing
    multiple types of nodes. The structure is expected to have a method
    'node_items' that yields node types and their corresponding data.
    - ptr (str): The key or pointer used to access node attributes within each
    node type's data.

    Returns:
    - torch.Tensor: A tensor containing the concatenated shuffled node
    attributes of all node types.
    """
    node_features = []
    for ntype, nx in data.node_items():
        shuffled_nodes = shuffle_node_attributes(nx[ptr])
        node_features.append(shuffled_nodes)
    return torch.cat(tuple(node_features), dim=0)
